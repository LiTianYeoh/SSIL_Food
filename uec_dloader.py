import torch
from torchvision import transforms
import os
import pandas as pd
import PIL.Image
from torchvision.datasets import VisionDataset
from utils.img_utils import view_sample_img
from utils.transform_utils import train_trans, test_trans, pure_trans
from torch.utils.data import DataLoader


main_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(main_dir,'data')
uec_base_dir = os.path.join(data_dir, 'uec256')

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Found GPU device: {}".format(torch.cuda.get_device_name(0)))
else:
    device = torch.device("cpu")
    print("Using CPU")

class UECSubset(VisionDataset):
    def __init__(self, root_dir, ltype, split, transform):
        super().__init__(root_dir, transform=transform)
        self.meta_dir = os.path.join(root_dir, 'meta')
        self.image_dir = os.path.join(root_dir, 'UECFOOD256')
        
        #read img list for type and split specified
        csv_name = ltype + '_' + split + '.csv'
        csv_path = os.path.join(self.meta_dir, csv_name)
        self.img_list = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        cur_img_info = self.img_list.loc[idx]
        img_class = int(cur_img_info['manual_class'].item())
        img_act_id = int(cur_img_info['actual_id'].item())
        img_jpg_name = str(int(cur_img_info['img'].item())) + '.jpg'

        image_file = os.path.join(self.image_dir, str(img_act_id), img_jpg_name)
        base_image = PIL.Image.open(image_file).convert("RGB")

        #crop and transform
        x1 = cur_img_info['x1'].item()
        y1 = cur_img_info['y1'].item()
        x2 = cur_img_info['x2'].item()
        y2 = cur_img_info['y2'].item()

        bb_crop_image = base_image.crop((x1, y1, x2, y2))

        if self.transform:
            return_img = self.transform(bb_crop_image)

        return return_img, img_class


uec_ds = UECSubset(uec_base_dir, 'off', 'train', transform = pure_trans)
uec_dloader = DataLoader(uec_ds, batch_size = 9, shuffle=True, num_workers=8)


norm_trans = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])


# read category.txt as tsv
class_info_path = os.path.join(uec_base_dir, 'meta', 'class_list.csv')
class_info = pd.read_csv(class_info_path)

num_class = len(class_info)
class_map = class_info['name'].tolist()


def model_pred(img_gpu):  
    num_img = img_gpu.shape[0]  
    return [1]*num_img

view_sample_img(uec_dloader, model_pred, device = device, label_map=class_map)
