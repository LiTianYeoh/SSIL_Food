from torchvision.datasets import VisionDataset
from torch.utils.data import Dataset, DataLoader
import json
import PIL.Image
import os
import yaml
from .transform_utils import train_trans, test_trans, pure_trans

######### Classes
class ftr_dataset(Dataset):

    def __init__(self, ftrvecs, label):
        super(ftr_dataset, self).__init__()
        self.ftrvecs = ftrvecs
        self.label = label
    
    def __getitem__(self, idx):
        return self.ftrvecs[idx], self.label[idx]

    def __len__(self):
        return len(self.ftrvecs)

class DoubleAugmentedDataset(Dataset):

    def __init__(self, dataset, train_trans, test_trans):
        super(DoubleAugmentedDataset, self).__init__()
        self.dataset = dataset 
        self.train_trans = train_trans
        self.test_trans = test_trans

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        orig_img, label = self.dataset[idx]
        img = self.test_trans(orig_img)
        aug1 = self.train_trans(orig_img)
        aug2 = self.train_trans(orig_img)
        return {"index": idx, "img": img, "aug_1": aug1, "aug_2": aug2, "label": label}

class F1Subset(VisionDataset):
    def __init__(self, root, split, transform, label_list, start_label_idx):
        super().__init__(root, transform=transform)
        self.label_list = label_list
        self.start_label_idx = start_label_idx
        self.split = split

        self.num_label = len(self.label_list)
        self.end_label_idx = self.start_label_idx + self.num_label -1
        
        self.meta_dir = os.path.join(root, 'meta')
        self.image_dir = os.path.join(root, 'images')
    
        self.labels = []
        self.image_files = []

        json_path = os.path.join(self.meta_dir, f'{split}.json')
        json_file = open(json_path, 'r')
        metadata = json.loads(json_file.read())

        self.classes = sorted(metadata.keys())
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))

        for class_name, im_rel_paths in metadata.items():
            if self.class_to_idx[class_name] in self.label_list:
                label = self.label_list.index(self.class_to_idx[class_name]) + self.start_label_idx
                self.labels += [label] * len(im_rel_paths)
                self.image_files += [
                    os.path.join(self.image_dir, f'{im_rel_path}.jpg') for im_rel_path in im_rel_paths
                ]
            else:
                pass
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file, label = self.image_files[idx], self.labels[idx]
        image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label




########## Functions for getting dloader

def get_ftr_dloader(ftrvec, label, batch_size):
    ftr_ds = ftr_dataset(ftrvec, label)
    ftr_dloader = DataLoader(ftr_ds, batch_size = batch_size, num_workers=8)

    return ftr_dloader


def get_f1_ord_dloader(f1_root_dir, off_inc='off', batch_size=64):
    #### read split yaml file
    ds_split_path = os.path.join(f1_root_dir, 'food101_split.yaml')
    split_file = open(ds_split_path, "r")
    food_ds_split = yaml.safe_load(split_file)
    split_file.close()

    off_label, inc_label = food_ds_split['off'], food_ds_split['inc']
    num_off_class = len(off_label)
    num_inc_class = len(inc_label)

    if off_inc == 'off':
        lab_list = off_label
        start_index = 0
        dloader_num_class = num_off_class
        set_text = "Offline"
    elif off_inc == 'inc':
        lab_list = inc_label
        start_index = num_off_class
        dloader_num_class = num_inc_class
        set_text = "Incremental"

    f1_train_ds = F1Subset(f1_root_dir, split='train', transform = train_trans, label_list=lab_list, start_label_idx=start_index)
    f1_train_loader = DataLoader(f1_train_ds, batch_size = batch_size, shuffle = True, num_workers = 8)
    print(f'{set_text} learning ordinary train dataloader: ' +
        f'{f1_train_ds.num_label} classess from ' +
        f'{f1_train_ds.start_label_idx} to {f1_train_ds.end_label_idx}, ' +
        f'Number of images = {len(f1_train_ds)}'
    )

    f1_test_ds = F1Subset(f1_root_dir, split='test', transform = train_trans, label_list=lab_list, start_label_idx=start_index)
    f1_test_loader = DataLoader(f1_test_ds, batch_size = batch_size, shuffle = False, num_workers = 8)
    print(f'{set_text} learning ordinary test dataloader: ' +
        f'{f1_test_ds.num_label} classess from ' +
        f'{f1_test_ds.start_label_idx} to {f1_test_ds.end_label_idx}, ' +
        f'Number of images = {len(f1_test_ds)}'
    )

    return f1_train_loader, f1_test_loader, dloader_num_class

def get_f1_off_relic_dloader(f1_root_dir, batch_size=20):
    #### read split yaml file
    ds_split_path = os.path.join(f1_root_dir, 'food101_split.yaml')
    split_file = open(ds_split_path, "r")
    food_ds_split = yaml.safe_load(split_file)
    split_file.close()

    off_label = food_ds_split['off']
    num_off_class = len(off_label)

    f1_relic_train_ds = F1Subset(f1_root_dir, split='train', transform = None,
    label_list=off_label, start_label_idx=0)
    
    relic_off_train_ds = DoubleAugmentedDataset(f1_relic_train_ds, train_trans=train_trans, test_trans=test_trans)
    relic_off_train_loader = DataLoader(relic_off_train_ds, batch_size = batch_size, shuffle = True, num_workers = 8)  

    print(f'Offline learning ReLIC train dataloader: ' +
        f'{f1_relic_train_ds.num_label} classess from ' +
        f'{f1_relic_train_ds.start_label_idx} to {f1_relic_train_ds.end_label_idx}, ' +
        f'Number of images = {len(f1_relic_train_ds)}'
    )

    return relic_off_train_loader

def get_f1_eval_train_dloader(f1_root_dir, batch_size =64):

    ds_split_path = os.path.join(f1_root_dir, 'food101_split.yaml')
    split_file = open(ds_split_path, "r")
    food_ds_split = yaml.safe_load(split_file)
    split_file.close()

    off_label, _ = food_ds_split['off'], food_ds_split['inc']
    num_off_class = len(off_label)

    f1_eval_train_ds = F1Subset(f1_root_dir, split='train', transform = test_trans, label_list=off_label, start_label_idx=0)
    f1_eval_train_loader = DataLoader(f1_eval_train_ds, batch_size = batch_size, shuffle = False, num_workers = 8)

    return f1_eval_train_loader, num_off_class

def get_f1_pure_dloader(f1_root_dir, off_inc='off', split = 'train', batch_size=64):
    #### read split yaml file
    ds_split_path = os.path.join(f1_root_dir, 'food101_split.yaml')
    split_file = open(ds_split_path, "r")
    food_ds_split = yaml.safe_load(split_file)
    split_file.close()

    off_label, inc_label = food_ds_split['off'], food_ds_split['inc']
    num_off_class = len(off_label)
    num_inc_class = len(inc_label)

    if off_inc == 'off':
        lab_list = off_label
        start_index = 0
        dloader_num_class = num_off_class
    elif off_inc == 'inc':
        lab_list = inc_label
        start_index = num_off_class
        dloader_num_class = num_inc_class

    f1_pure_ds = F1Subset(f1_root_dir, split=split, transform = pure_trans, label_list=lab_list, start_label_idx=start_index)
    f1_pure_loader = DataLoader(f1_pure_ds, batch_size = batch_size, shuffle = True, num_workers = 8)

    return f1_pure_loader, dloader_num_class

