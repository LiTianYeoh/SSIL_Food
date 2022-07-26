import torch
from torch.utils.data import DataLoader
from torchvision.datasets import Food101
from torchvision import transforms
import matplotlib.pyplot as plt
import os
from transform_util import pure_trans

main_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_dir = os.path.join(main_dir,'data')

input_size = [224, 224]



pure_ds = Food101(root=data_dir, split='train', transform = pure_trans, download=True)
f1_train_loader = DataLoader(pure_ds, batch_size = 8, shuffle = True, num_workers = 8)

batch_load = iter(f1_train_loader)
img, label = batch_load.next()

for i in range(8):
    plt.subplot(2, 4, i+1)
    rgb_image = img[i].permute(1,2,0)
    plt.imshow(rgb_image.numpy())
    plt.title(f'Label: {label[i]}')
plt.show()