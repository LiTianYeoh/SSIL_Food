from torch.nn import ModuleList
from torchvision import transforms

input_size = [224, 224]

train_trans = transforms.Compose([
    transforms.RandomApply(
        ModuleList([transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4,hue=0.1)]),
        p=0.8
    ),
    transforms.RandomResizedCrop(input_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
])

test_trans = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
])

pure_trans = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor(),
])