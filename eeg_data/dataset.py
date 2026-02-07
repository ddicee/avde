import os
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode, transforms
from torch.utils.data import Dataset
from einops import rearrange
import open_clip
import dist
import torch.nn.functional as F

data_root = "/path/to/data/"
data_path = os.path.join(data_root, "preprocessed_bc")
# data_path = os.path.join(data_root, "preprocessed_whitened")
img_directory_training = os.path.join(data_root, "image_set/training_images")
img_directory_test = os.path.join(data_root, "image_set/test_images")
# features_path = os.path.join(data_root, "Generation/variables")

class EEGDataset(Dataset):
    def __init__(self, subjects, classes=None, pictures=None, train=True, exclude_subject=None):
        self.subjects = subjects
        self.classes = classes
        self.pictures = pictures
        self.train = train
        self.exclude_subject = exclude_subject
        self.n_cls = 1654 if train else 200
        self.data, self.labels, self.text, self.img = load_data(subjects, classes, pictures, train, exclude_subject)
        # self.text_features = self.Textencoder(self.text)
        # self.image_features = self.ImageEncoder(self.img)
        model_type = 'ViT-H-14'
        features_filename = os.path.join(f'{model_type}_features_train.pt') if self.train else os.path.join(f'{model_type}_features_test.pt')
        saved_features = torch.load(features_filename)
        self.text_features = saved_features['text_features']
        self.img_features = saved_features['img_features']
    def Textencoder(self, text):
        model_type = 'ViT-H-14'
        device = dist.get_device()
        vlmodel, preprocess_train, feature_extractor = open_clip.create_model_and_transforms(
            model_type, pretrained='/path/to/pretrained/model', precision='fp32', device=device)
        tokenizer = open_clip.get_tokenizer(model_type)
        text_inputs = torch.cat([tokenizer(t) for t in text]).to(device)

        with torch.no_grad():
            text_features = vlmodel.encode_text(text_inputs)
        
        text_features = F.normalize(text_features, dim=-1).detach()
        text_features = text_features.cpu()
    
        return text_features
    
    def ImageEncoder(self,images):
        model_type = 'ViT-H-14'
        device = dist.get_device()
        vlmodel, preprocess_train, feature_extractor = open_clip.create_model_and_transforms(
            model_type, pretrained='/path/to/pretrained/model', precision='fp32', device=device)
        
        batch_size = 1024
        image_features_list = []
      
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            image_inputs = torch.stack([preprocess_train(Image.open(img).convert("RGB")) for img in batch_images]).to(device)

            with torch.no_grad():
                batch_image_features = vlmodel.encode_image(image_inputs)
                # batch_image_features /= batch_image_features.norm(dim=-1, keepdim=True)   #remove code when reconstruction 


            image_features_list.append(batch_image_features)

        image_features = torch.cat(image_features_list, dim=0)
        print("image_features shape", image_features.shape)
        
        return image_features
    
    def prepare_image(
        self,
        img_path: str, final_reso: int,
        hflip=False, mid_reso=1.125,
    ):
        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        # build augmentations
        mid_reso = round(mid_reso * final_reso)  # first resize to mid_reso, then crop to final_reso
        train_aug, val_aug = [
            transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS), # transforms.Resize: resize the shorter edge to mid_reso
            transforms.RandomCrop((final_reso, final_reso)),
            transforms.ToTensor(), normalize_01_into_pm1,
        ], [
            transforms.Resize(final_reso, interpolation=InterpolationMode.BILINEAR), # transforms.Resize: resize the shorter edge to mid_reso
            transforms.ToTensor(), normalize_01_into_pm1,
        ]
        if hflip: train_aug.insert(0, transforms.RandomHorizontalFlip())
        train_aug, val_aug = transforms.Compose(train_aug), transforms.Compose(val_aug)
        
        if self.train:
            img = train_aug(img)
        else:
            img = val_aug(img)
        
        return img

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        # Get the data and label corresponding to "index"
        # index: (subjects * classes * 10 * 4)
        x = self.data[index]
        label = self.labels[index]
        
        if self.pictures is None:
            if self.classes is None:
                index_n_sub_train = self.n_cls * 10 * 4
                index_n_sub_test = self.n_cls * 1 * 80
            else:
                index_n_sub_test = len(self.classes)* 1 * 80
                index_n_sub_train = len(self.classes)* 10 * 4
            # text_index: classes
            if self.train:
                text_index = (index % index_n_sub_train) // (10 * 4)
            else:
                text_index = (index % index_n_sub_test) 
            # img_index: classes * 10
            if self.train:
                img_index = (index % index_n_sub_train) // (4)
            else:
                img_index = (index % index_n_sub_test) 
        else:
            if self.classes is None:
                index_n_sub_train = self.n_cls * 1 * 4
                index_n_sub_test = self.n_cls * 1 * 80
            else:
                index_n_sub_test = len(self.classes)* 1 * 80
                index_n_sub_train = len(self.classes)* 1 * 4
            # text_index: classes
            if self.train:
                text_index = (index % index_n_sub_train) // (1 * 4)
            else:
                text_index = (index % index_n_sub_test) 
            # img_index: classes * 10
            if self.train:
                img_index = (index % index_n_sub_train) // (4)
            else:
                img_index = (index % index_n_sub_test) 
                
        x = x.float() / 100
        x = rearrange(x, 'N (A T) -> N A T', T=200)
        text = self.text[text_index]
        text_features = self.text_features[text_index]
        
        img = self.prepare_image(self.img[img_index], final_reso=256)
        img_features = self.img_features[img_index]
        
        return x, img, label, text, text_features, img_features


def load_data(subjects, classes=None, pictures=None, train=True, exclude_subject=None):
    data_list = []
    label_list = []
    texts = []
    images = []
    
    if train:
        directory = img_directory_training
    else:
        directory = img_directory_test

    dirnames = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    dirnames.sort()
    
    if classes is not None:
        dirnames = [dirnames[i] for i in classes]

    for dir in dirnames:

        try:
            idx = dir.index('_')
            description = dir[idx+1:]
        except ValueError:
            print(f"Skipped: {dir} due to no '_' found.")
            continue
            
        # new_description = f"This picture is {description}"
        texts.append(description)

    if train:
        img_directory = img_directory_training
    else:
        img_directory = img_directory_test
    
    all_folders = [d for d in os.listdir(img_directory) if os.path.isdir(os.path.join(img_directory, d))]
    all_folders.sort()

    if classes is not None and pictures is not None:
        images = []
        for i in range(len(classes)):
            class_idx = classes[i]
            pic_idx = pictures[i]
            if class_idx < len(all_folders):
                folder = all_folders[class_idx]
                folder_path = os.path.join(img_directory, folder)
                all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                all_images.sort()
                if pic_idx < len(all_images):
                    images.append(os.path.join(folder_path, all_images[pic_idx]))
    elif classes is not None and pictures is None:
        images = []
        for i in range(len(classes)):
            class_idx = classes[i]
            if class_idx < len(all_folders):
                folder = all_folders[class_idx]
                folder_path = os.path.join(img_directory, folder)
                all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                all_images.sort()
                images.extend(os.path.join(folder_path, img) for img in all_images)
    elif classes is None:
        images = []
        for folder in all_folders:
            folder_path = os.path.join(img_directory, folder)
            all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
            all_images.sort()  
            images.extend(os.path.join(folder_path, img) for img in all_images)
    else:

        print("Error")
        
    for subject in subjects:
        if train:
            if subject == exclude_subject:
                continue            
            # print("subject:", subject)    
            file_name = 'preprocessed_eeg_training.npy'

            file_path = os.path.join(data_path, subject, file_name)
            data = np.load(file_path, allow_pickle=True)
            
            preprocessed_eeg_data = torch.from_numpy(data['preprocessed_eeg_data']).float().detach()                
            times = torch.from_numpy(data['times']).detach()[50:]
            ch_names = data['ch_names']

            n_classes = 1654
            samples_per_class = 10
            
            if classes is not None and pictures is not None:
                for c, p in zip(classes, pictures):
                    start_index = c * 1 + p
                    if start_index < len(preprocessed_eeg_data):
                        preprocessed_eeg_data_class = preprocessed_eeg_data[start_index: start_index+1]
                        labels = torch.full((1,), c, dtype=torch.long).detach()
                        data_list.append(preprocessed_eeg_data_class)
                        label_list.append(labels)

            elif classes is not None and pictures is None:
                for c in classes:
                    start_index = c * samples_per_class
                    preprocessed_eeg_data_class = preprocessed_eeg_data[start_index: start_index+samples_per_class]
                    labels = torch.full((samples_per_class,), c, dtype=torch.long).detach()
                    data_list.append(preprocessed_eeg_data_class)
                    label_list.append(labels)

            else:
                for i in range(n_classes):
                    start_index = i * samples_per_class
                    # if self.exclude_subject==None:
                    #     preprocessed_eeg_data_class = preprocessed_eeg_data[start_index: start_index+samples_per_class]
                    # else:
                    preprocessed_eeg_data_class = preprocessed_eeg_data[start_index: start_index+samples_per_class]
                    # print("preprocessed_eeg_data_class", preprocessed_eeg_data_class.shape)
                    # preprocessed_eeg_data_class = torch.mean(preprocessed_eeg_data_class, 1)
                    # preprocessed_eeg_data_class = torch.mean(preprocessed_eeg_data_class, 0)
                    # print("preprocessed_eeg_data_class", preprocessed_eeg_data_class.shape)
                    labels = torch.full((samples_per_class,), i, dtype=torch.long).detach()
                    data_list.append(preprocessed_eeg_data_class)
                    label_list.append(labels)

                
        else:
            if subject == exclude_subject or exclude_subject==None:
                file_name = 'preprocessed_eeg_test.npy'
                file_path = os.path.join(data_path, subject, file_name)
                data = np.load(file_path, allow_pickle=True)
                preprocessed_eeg_data = torch.from_numpy(data['preprocessed_eeg_data']).float().detach()
                times = torch.from_numpy(data['times']).detach()[50:]
                ch_names = data['ch_names']
                n_classes = 200  # Each class contains 1 images
                
                samples_per_class = 1

                for i in range(n_classes):
                    if classes is not None and i not in classes:  # If we've defined specific classes and the current class is not in the list, skip
                        continue
                    start_index = i * samples_per_class  # Update start_index for each class
                    preprocessed_eeg_data_class = preprocessed_eeg_data[start_index:start_index+samples_per_class]
                    # print("preprocessed_eeg_data_class", preprocessed_eeg_data_class.shape)
                    labels = torch.full((samples_per_class,), i, dtype=torch.long).detach()  # Add class labels
                    preprocessed_eeg_data_class = torch.mean(preprocessed_eeg_data_class.squeeze(0), 0)
                    # print("preprocessed_eeg_data_class", preprocessed_eeg_data_class.shape)
                    data_list.append(preprocessed_eeg_data_class)
                    label_list.append(labels)  # Add labels to the label list
            else:
                continue
    # datalist: (subjects * classes) * (10 * 4 * 17 * 100)
    # data_tensor: (subjects * classes * 10 * 4) * 17 * 100
    # data_list = np.mean(data_list, )
    # print("data_list", len(data_list))
    if train:
        # print("data_list", *data_list[0].shape[1:])            
        data_tensor = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape[2:])                 
        # data_tensor = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape[1:])
        # data_tensor = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape)   
        # print("label_tensor", label_tensor.shape)
        print("data_tensor", data_tensor.shape)
    else:           
        data_tensor = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape)   
        # label_tensor = torch.cat(label_list, dim=0)
        # print("label_tensor", label_tensor.shape)
        # data_tensor = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape[2:])
    # print("data_tensor", data_tensor.shape)
    # label_list: (subjects * classes) * 10
    # label_tensor: (subjects * classes * 10)
    # print("label_tensor = torch.cat(label_list, dim=0)")
    # print(label_list)
    label_tensor = torch.cat(label_list, dim=0)
    # label_tensor = torch.cat(label_list, dim=0)
    # print(label_tensor[:300])
    if train:
        # label_tensor: (subjects * classes * 10 * 4)
        label_tensor = label_tensor.repeat_interleave(4)
        if classes is not None:
            unique_values = list(label_tensor.numpy())
            lis = []
            for i in unique_values:
                if i not in lis:
                    lis.append(i)
            unique_values = torch.tensor(lis)        
            mapping = {val.item(): index for index, val in enumerate(unique_values)}   
            label_tensor = torch.tensor([mapping[val.item()] for val in label_tensor], dtype=torch.long)

    else:
        # label_tensor = label_tensor.repeat_interleave(80)
        # if self.classes is not None:
        #     unique_values = torch.unique(label_tensor, sorted=False)
        
        #     mapping = {val.item(): index for index, val in enumerate(torch.flip(unique_values, [0]))}
        #     label_tensor = torch.tensor([mapping[val.item()] for val in label_tensor], dtype=torch.long)
        pass      

                
    times = times
    ch_names = ch_names

    print(f"Data tensor shape: {data_tensor.shape}, label tensor shape: {label_tensor.shape}, text length: {len(texts)}, image length: {len(images)}")
    
    return data_tensor, label_tensor, texts, images

def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)



# trainset = EEGDataset(subjects=['sub-01'], train=True)
# testset = EEGDataset(subjects=['sub-01'], train=False)