## Download Dataset
You can download the raw THINGS-EEG dataset from [osf](https://osf.io/3jk45/).

## Environment setup
Create a conda environment with python 3.10
```
conda create -n avde python=3.10
```
Install the required packages via pip
```
pip install -r requirements.txt
```

## Data Preprocessing
Provide a path for storing the pre-processed data and run the script
```
python eeg_data/preprocess.py --project_dir "/path/to/data/"
```

## Fine-tune pre-trained LaBraM via contrastive learning
```
torchrun --nnodes=1 --nproc_per_node=2 --master_port=222 run_labram_finetuning.py \
        --output_dir /path/to/output/dir \
        --description "" \
        --model labram_base_patch200_200 \
        --finetune ./pretrained/labram-base.pth \
        --weight_decay 0.05 \
        --batch_size 128 \
        --lr 5e-4 \
        --update_freq 1 \
        --warmup_epochs 5 \
        --epochs 50 \
        --layer_decay 0.8 \
        --drop_path 0.1 \
        --dist_eval \
        --save_ckpt_freq 5 \
        --disable_rel_pos_bias \
        --abs_pos_emb \
        --disable_qkv_bias \
        --seed 0 \
        --sub "sub-08" \
        --no_auto_resume \
```

## Train the visual autoregressive transformer
```
OMP_NUM_THREADS=1 torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 train_eeg.py \
    --depth=16 --bs=128 --workers=5 --ep=100 --fp16=1 --alng=1e-3 --wpe=0.1 --tblr=2e-5 \
    --labram_ckpt="/path/to/the/labram/checkpoint" \
    --local_out_dir_path="" \
    --sub="sub-08"
```

## Acknowledgement

This repository builds upon the following projects. We are grateful for their contributions.

- [Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI](https://github.com/935963004/LaBraM)
- [Visual Decoding and Reconstruction via EEG Embeddings with Guided Diffusion](https://github.com/dongyangli-del/EEG_Image_decode)
- [Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction](https://github.com/FoundationVision/VAR)