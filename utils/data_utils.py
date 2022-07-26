import torch
from torch.utils.data import Dataset
import torchvision
from torchvision.datasets import Food101, VisionDataset
from torchvision import transforms
import os

main_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_dir = os.path.join(main_dir,'data')

input_size = [224, 224]
data_aug_train = transforms.Compose([
    transforms.RandomApply(
        torch.nn.ModuleList([transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4,hue=0.1)]),
        p=0.8
    ),
    transforms.RandomResizedCrop(input_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
])

data_aug_test = transforms.Compose([
    transforms.CenterCrop([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
])

f1_train = Food101(root=data_dir, split='train', transform=data_aug_train, download=True)
f1_test = Food101(root=data_dir, split='test', transform=data_aug_test, download=True)

off_label = range(81)
inc_label = range(81,101)


off_train_indices = [i for i in range(len(f1_train)) if (f1_train[i][1] in off_label)]
off_test_indices = [j for j in range(len(f1_test)) if (f1_test[j][1] in off_label)]

class F1Subset(VisionDataset):
    def __init__(self, root, split, transform, label_list):
        super().__init__(root, transform=transform)
        self.label_list = label_list
        self.split = split

    def __getitem__(self, idx):
        

        return super().__getitem__(idx)

    def __len__(self):
        return 5


