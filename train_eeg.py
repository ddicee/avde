import warnings
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")
warnings.filterwarnings("ignore", message="Reliance on distutils from stdlib is deprecated")

import gc
import os
# os.environ["WANDB_MODE"] = 'disabled'
import shutil
import sys
import time
from functools import partial

import wandb
import torch
from torch.utils.data import DataLoader

import dist
from utils import arg_util, misc
from eeg_data.dataset import EEGDataset
from utils.data_sampler import DistInfiniteBatchSampler, EvalDistributedSampler
from utils.misc import auto_resume


def build_everything(args: arg_util.Args):
    # resume
    # auto_resume_info, start_ep, start_it, trainer_state, args_state = auto_resume(args, 'ar-ckpt*.pth')
    start_ep, start_it = 0, 0
    trainer_state, args_state = None, None
    # create tensorboard logger
    wandb_lg: misc.WandbLogger
    with_wandb_lg = dist.is_master()
    if with_wandb_lg:
        # Initialize wandb
        wandb.init(
            project="avde",  # Replace with your project name
            name=f'run_{misc.time_str("%m%d_%H%M")}',
            config=vars(args)
        )
        # noinspection PyTypeChecker
        wandb_lg = misc.DistLogger(misc.WandbLogger(), verbose=True)
        wandb_lg.flush()
    else:
        # noinspection PyTypeChecker
        wandb_lg = misc.DistLogger(None, verbose=False)
    dist.barrier()
    
    # log args
    print(f'global bs={args.glb_batch_size}, local bs={args.batch_size}')
    print(f'initial args:\n{str(args)}')
    
    # build data
    print(f'[build PT data] ...\n')
    if args.cross_subject:
        dataset_train = EEGDataset(exclude_subject=args.sub, subjects=[f"sub-{i:02d}" for i in range(1, 11)], train=True)
        dataset_val = EEGDataset(exclude_subject=args.sub, subjects=[f"sub-{i:02d}" for i in range(1, 11)], train=False)
    else:
        dataset_train = EEGDataset(subjects=[args.sub], train=True)
        dataset_val = EEGDataset(subjects=[args.sub], train=False)
    types = str((type(dataset_train).__name__, type(dataset_val).__name__))
        
    ld_val = DataLoader(
        dataset_val, num_workers=0, pin_memory=True,
        batch_size=round(args.batch_size*1.5), sampler=EvalDistributedSampler(dataset_val, num_replicas=dist.get_world_size(), rank=dist.get_rank()),
        shuffle=False, drop_last=False,
    )
    del dataset_val
    
    ld_train = DataLoader(
        dataset=dataset_train, num_workers=args.workers, pin_memory=True,
        generator=args.get_different_generator_for_each_rank(), # worker_init_fn=worker_init_fn,
        batch_sampler=DistInfiniteBatchSampler(
            dataset_len=len(dataset_train), glb_batch_size=args.glb_batch_size, same_seed_for_all_ranks=args.same_seed_for_all_ranks,
            shuffle=True, fill_last=True, rank=dist.get_rank(), world_size=dist.get_world_size(), start_ep=start_ep, start_it=start_it,
        ),
    )
    del dataset_train
    
    # [print(line) for line in auto_resume_info]
    print(f'[dataloader multi processing] ...', end='', flush=True)
    stt = time.time()
    iters_train = len(ld_train)
    ld_train = iter(ld_train)
    # noinspection PyArgumentList
    print(f'     [dataloader multi processing](*) finished! ({time.time()-stt:.2f}s)', flush=True, clean=True)
    print(f'[dataloader] gbs={args.glb_batch_size}, lbs={args.batch_size}, iters_train={iters_train}, types(tr, va)={types}')
        

    
    # build models
    from torch.nn.parallel import DistributedDataParallel as DDP
    from models import VAREEG, VQVAE, NeuralTransformer, build_vae_var_eeg, load_labram_ckpt
    from trainer_eeg import VAREEGTrainer
    from utils.amp_sc import AmpOptimizer
    from utils.lr_control import filter_params
    
    vae_local, labram_local, var_wo_ddp = build_vae_var_eeg(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,        # hard-coded VQVAE hyperparameters
        device=dist.get_device(), patch_nums=args.patch_nums,
        depth=args.depth, shared_aln=args.saln, attn_l2_norm=args.anorm,
        flash_if_available=args.fuse, fused_if_available=args.fuse,
        init_adaln=args.aln, init_adaln_gamma=args.alng, init_head=args.hd, init_std=args.ini,
    )
    
    vae_ckpt = 'pretrained/vae_ch160v4096z32.pth'
    var_ckpt = f'pretrained/var_d{args.depth}.pth'
    # labram_ckpt = 'pretrained/labram-base.pth'
    if dist.is_local_master():
        if not os.path.exists(vae_ckpt):
            os.system(f'wget https://huggingface.co/FoundationVision/var/resolve/main/{vae_ckpt}')
    dist.barrier()
    vae_local.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
    var_wo_ddp.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=False)
    load_labram_ckpt(labram_local, torch.load(args.labram_ckpt, map_location='cpu'))
    if not args.unfreeze_labram:
        for param in labram_local.parameters():
            param.requires_grad = False
    
    vae_local: VQVAE = args.compile_model(vae_local, args.vfast)
    var_wo_ddp: VAREEG = args.compile_model(var_wo_ddp, args.tfast)
    labram_local: NeuralTransformer = args.compile_model(labram_local, args.tfast)
    var: DDP = (DDP if dist.initialized() else NullDDP)(var_wo_ddp, device_ids=[dist.get_local_rank()], find_unused_parameters=False, broadcast_buffers=False)
    
    print(f'[INIT] VAR model = {var_wo_ddp}\n\n')
    count_p = lambda m: f'{sum(p.numel() for p in m.parameters())/1e6:.2f}'
    print(f'[INIT][#para] ' + ', '.join([f'{k}={count_p(m)}' for k, m in (('VAE', vae_local), ('VAE.enc', vae_local.encoder), ('VAE.dec', vae_local.decoder), ('VAE.quant', vae_local.quantize))]))
    print(f'[INIT][#para] ' + ', '.join([f'{k}={count_p(m)}' for k, m in (('VAR', var_wo_ddp),)]) + '\n\n')
    
    # build optimizer
    names, paras, para_groups = filter_params(var_wo_ddp, nowd_keys={
        'cls_token', 'start_token', 'task_token', 'cfg_uncond',
        'pos_embed', 'pos_1LC', 'pos_start', 'start_pos', 'lvl_embed',
        'gamma', 'beta',
        'ada_gss', 'moe_bias',
        'scale_mul',
    })
    opt_clz = {
        'adam':  partial(torch.optim.AdamW, betas=(0.9, 0.95), fused=args.afuse),
        'adamw': partial(torch.optim.AdamW, betas=(0.9, 0.95), fused=args.afuse),
    }[args.opt.lower().strip()]
    opt_kw = dict(lr=args.tlr, weight_decay=0)
    print(f'[INIT] optim={opt_clz}, opt_kw={opt_kw}\n')
    
    var_optim = AmpOptimizer(
        mixed_precision=args.fp16, optimizer=opt_clz(params=para_groups, **opt_kw), names=names, paras=paras,
        grad_clip=args.tclip, n_gradient_accumulation=args.ac
    )
    del names, paras, para_groups
    
    # build trainer
    if not args.noise_debug:
        trainer = VAREEGTrainer(
            device=args.device, patch_nums=args.patch_nums, resos=args.resos,
            vae_local=vae_local, var_wo_ddp=var_wo_ddp, var=var, labram_local=labram_local,
            var_opt=var_optim, label_smooth=args.ls,
        )
    if trainer_state is not None and len(trainer_state):
        trainer.load_state_dict(trainer_state, strict=False, skip_vae=True) # don't load vae again
    del vae_local, var_wo_ddp, var, var_optim
    
    dist.barrier()
    return (
        wandb_lg, trainer, start_ep, start_it,
        iters_train, ld_train, ld_val
    )


def main_training():
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    
    (
        wandb_lg, trainer,
        start_ep, start_it,
        iters_train, ld_train, ld_val
    ) = build_everything(args)
        
    # train
    start_time = time.time()
    best_L_mean, best_L_tail, best_acc_mean, best_acc_tail = 999., 999., -1., -1.
    best_val_loss_mean, best_val_loss_tail, best_val_acc_mean, best_val_acc_tail = 999, 999, -1, -1
    
    L_mean, L_tail = -1, -1
    for ep in range(start_ep, args.ep):
        if hasattr(ld_train, 'sampler') and hasattr(ld_train.sampler, 'set_epoch'):
            ld_train.sampler.set_epoch(ep)
            if ep < 3:
                # noinspection PyArgumentList
                print(f'[{type(ld_train).__name__}] [ld_train.sampler.set_epoch({ep})]', flush=True, force=True)
        wandb_lg.set_step(ep * iters_train)
        
        stats, (sec, remain_time, finish_time) = train_one_ep(
            ep, ep == start_ep, start_it if ep == start_ep else 0, args, wandb_lg, ld_train, iters_train, trainer
        )
        
        L_mean, L_tail, acc_mean, acc_tail, grad_norm = stats['Lm'], stats['Lt'], stats['Accm'], stats['Acct'], stats['tnm']
        best_L_mean, best_acc_mean = min(best_L_mean, L_mean), max(best_acc_mean, acc_mean)
        if L_tail != -1: best_L_tail, best_acc_tail = min(best_L_tail, L_tail), max(best_acc_tail, acc_tail)
        args.L_mean, args.L_tail, args.acc_mean, args.acc_tail, args.grad_norm = L_mean, L_tail, acc_mean, acc_tail, grad_norm
        args.cur_ep = f'{ep+1}/{args.ep}'
        args.remain_time, args.finish_time = remain_time, finish_time
        
        AR_ep_loss = dict(L_mean=L_mean, L_tail=L_tail, acc_mean=acc_mean, acc_tail=acc_tail)
        is_val_and_also_saving = (ep + 1) % 1 == 0 or (ep + 1) == args.ep
        if is_val_and_also_saving:
            val_loss_mean, val_loss_tail, val_acc_mean, val_acc_tail, tot, cost = trainer.eval_ep(ld_val)
            best_updated = best_val_loss_mean > val_loss_mean
            best_val_loss_mean, best_val_loss_tail = min(best_val_loss_mean, val_loss_mean), min(best_val_loss_tail, val_loss_tail)
            best_val_acc_mean, best_val_acc_tail = max(best_val_acc_mean, val_acc_mean), max(best_val_acc_tail, val_acc_tail)
            AR_ep_loss.update(vL_mean=val_loss_mean, vL_tail=val_loss_tail, vacc_mean=val_acc_mean, vacc_tail=val_acc_tail)
            args.vL_mean, args.vL_tail, args.vacc_mean, args.vacc_tail = val_loss_mean, val_loss_tail, val_acc_mean, val_acc_tail
            print(f' [*] [ep{ep}]  (val {tot})  Lm: {L_mean:.4f}, Lt: {L_tail:.4f}, Acc m&t: {acc_mean:.2f} {acc_tail:.2f},  Val cost: {cost:.2f}s')
            
            if dist.is_local_master():
                local_out_ckpt = os.path.join(args.local_out_dir_path, 'ar-ckpt-last.pth')
                local_out_ckpt_best = os.path.join(args.local_out_dir_path, 'ar-ckpt-best.pth')
                print(f'[saving ckpt] ...', end='', flush=True)
                torch.save({
                    'epoch':    ep+1,
                    'iter':     0,
                    'trainer':  trainer.state_dict(),
                    'args':     args.state_dict(),
                }, local_out_ckpt)
                if best_updated:
                    shutil.copy(local_out_ckpt, local_out_ckpt_best)
                print(f'     [saving ckpt](*) finished!  @ {local_out_ckpt}', flush=True, clean=True)
            dist.barrier()
        
        print(    f'     [ep{ep}]  (training )  Lm: {best_L_mean:.3f} ({L_mean:.3f}), Lt: {best_L_tail:.3f} ({L_tail:.3f}),  Acc m&t: {best_acc_mean:.2f} {best_acc_tail:.2f},  Remain: {remain_time},  Finish: {finish_time}', flush=True)
        wandb_lg.update(head='AR_ep_loss', step=ep+1, **AR_ep_loss)
        wandb_lg.update(head='AR_z_burnout', step=ep+1, rest_hours=round(sec / 60 / 60, 2))
        args.dump_log(); wandb_lg.flush()
    
    total_time = f'{(time.time() - start_time) / 60 / 60:.1f}h'
    print('\n\n')
    print(f'  [*] [PT finished]  Total cost: {total_time},   Lm: {best_L_mean:.3f} ({L_mean}),   Lt: {best_L_tail:.3f} ({L_tail})')
    print('\n\n')
    
    del stats
    del iters_train, ld_train
    time.sleep(3), gc.collect(), torch.cuda.empty_cache(), time.sleep(3)
    
    args.remain_time, args.finish_time = '-', time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() - 60))
    print(f'final args:\n\n{str(args)}')
    args.dump_log(); wandb_lg.flush(); wandb_lg.close()
    dist.barrier()


def train_one_ep(ep: int, is_first_ep: bool, start_it: int, args: arg_util.Args, wandb_lg: misc.WandbLogger, ld_or_itrt, iters_train: int, trainer):
    # import heavy packages after Dataloader object creation
    from utils.lr_control import lr_wd_annealing
    from trainer_eeg import VAREEGTrainer
    trainer: VAREEGTrainer
    
    step_cnt = 0
    me = misc.MetricLogger(delimiter='  ')
    me.add_meter('tlr', misc.SmoothedValue(window_size=1, fmt='{value:.2g}'))
    me.add_meter('tnm', misc.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    [me.add_meter(x, misc.SmoothedValue(fmt='{median:.3f} ({global_avg:.3f})')) for x in ['Lm', 'Lt']]
    [me.add_meter(x, misc.SmoothedValue(fmt='{median:.2f} ({global_avg:.2f})')) for x in ['Accm', 'Acct']]
    header = f'[Ep]: [{ep:4d}/{args.ep}]'
    
    if is_first_ep:
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        warnings.filterwarnings('ignore', category=UserWarning)
    g_it, max_it = ep * iters_train, args.ep * iters_train
    
    for it, (eeg, img, label, text, text_features, img_features) in me.log_every(start_it, iters_train, ld_or_itrt, 30 if iters_train > 8000 else 5, header):
        g_it = ep * iters_train + it
        if it < start_it: continue
        if is_first_ep and it == start_it: warnings.resetwarnings()
        
        eeg = eeg.to(args.device, non_blocking=True)
        img = img.to(args.device, non_blocking=True)
        label = label.to(args.device, non_blocking=True)
        text_features = text_features.to(args.device, non_blocking=True)
        img_features = img_features.to(args.device, non_blocking=True)
        args.cur_it = f'{it+1}/{iters_train}'
        
        wp_it = args.wp * iters_train
        min_tlr, max_tlr, min_twd, max_twd = lr_wd_annealing(args.sche, trainer.var_opt.optimizer, args.tlr, args.twd, args.twde, g_it, wp_it, max_it, wp0=args.wp0, wpe=args.wpe)
        args.cur_lr, args.cur_wd = max_tlr, max_twd
        
        if args.pg: # default: args.pg == 0.0, means no progressive training, won't get into this
            if g_it <= wp_it: prog_si = args.pg0
            elif g_it >= max_it*args.pg: prog_si = len(args.patch_nums) - 1
            else:
                delta = len(args.patch_nums) - 1 - args.pg0
                progress = min(max((g_it - wp_it) / (max_it*args.pg - wp_it), 0), 1) # from 0 to 1
                prog_si = args.pg0 + round(progress * delta)    # from args.pg0 to len(args.patch_nums)-1
        else:
            prog_si = -1
        
        stepping = (g_it + 1) % args.ac == 0
        step_cnt += int(stepping)
        
        grad_norm, scale_log2 = trainer.train_step(
            it=it, g_it=g_it, stepping=stepping, metric_lg=me, logger=wandb_lg,
            inp_B3HW=img, inp_eeg=eeg, label_B=label, text_features=text_features, img_features=img_features,
            prog_si=prog_si, prog_wp_it=args.pgwp * iters_train,
        )
        
        me.update(tlr=max_tlr)
        wandb_lg.set_step(step=g_it)
        wandb_lg.update(head='AR_opt_lr/lr_min', sche_tlr=min_tlr)
        wandb_lg.update(head='AR_opt_lr/lr_max', sche_tlr=max_tlr)
        wandb_lg.update(head='AR_opt_wd/wd_max', sche_twd=max_twd)
        wandb_lg.update(head='AR_opt_wd/wd_min', sche_twd=min_twd)
        wandb_lg.update(head='AR_opt_grad/fp16', scale_log2=scale_log2)
        
        if args.tclip > 0:
            wandb_lg.update(head='AR_opt_grad/grad', grad_norm=grad_norm)
            wandb_lg.update(head='AR_opt_grad/grad', grad_clip=args.tclip)
    
    me.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in me.meters.items()}, me.iter_time.time_preds(max_it - (g_it + 1) + (args.ep - ep) * 15)  # +15: other cost


class NullDDP(torch.nn.Module):
    def __init__(self, module, *args, **kwargs):
        super(NullDDP, self).__init__()
        self.module = module
        self.require_backward_grad_sync = False
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


if __name__ == '__main__':
    try: main_training()
    finally:
        dist.finalize()
        if isinstance(sys.stdout, misc.SyncPrint) and isinstance(sys.stderr, misc.SyncPrint):
            sys.stdout.close(), sys.stderr.close()
