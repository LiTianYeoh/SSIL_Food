import numpy as np
import torch
from torch import nn 
from torch.utils.data import DataLoader, Subset
from torch.optim import SGD
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
import os
import yaml
import matplotlib.pyplot as plt

main_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

## parameter
#state_path = None
state_path = 'supervised_offline_e012_wosched.pt'

to_train = False
eval_perf = True
show_train_loss = True

max_epoch = 100
wu_epoch = 10
batch_s = 64


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Found GPU device: {}".format(torch.cuda.get_device_name(0)))
else:
    device = torch.device("cpu")
    print("Using CPU")





# 0 Prepare data
print('Preparing food101 dataset...')

data_dir = os.path.join(main_dir,'data')

print('Checking dataset...')

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


food_train = torchvision.datasets.Food101(root=data_dir, split='train', transform=data_aug_train, download=True)
food_test = torchvision.datasets.Food101(root=data_dir, split='test', transform=data_aug_test, download=True)

## Seperate into 80% offline learning class and 20% incremental learning class
print('Splitting data for offline learning...')

#### read split yaml file
ds_split_rel_path = 'data/food-101/food101_split.yaml'
ds_split_path = os.path.join(main_dir, ds_split_rel_path)
split_file = open(ds_split_path, "r")
food_ds_split = yaml.safe_load(split_file)
split_file.close()

off_label, inc_label = food_ds_split['off'], food_ds_split['inc']
#num_off_class = len(off_label)
num_off_class = 101

'''
off_train_indices = [i for i in range(len(food_train)) if (food_train[i][1] in off_label)]
off_test_indices = [j for j in range(len(food_test)) if (food_test[j][1] in off_label)]

off_train_ds = Subset(food_train, off_train_indices)
off_test_ds = Subset(food_test, off_test_indices)
'''

off_train_ds = food_train
off_test_ds = food_test




off_train_loader = DataLoader(off_train_ds, batch_size = batch_s, shuffle = True, num_workers = 8)
off_test_loader = DataLoader(off_test_ds, batch_size = batch_s, shuffle = True, num_workers = 8)


# 1 model and training
class sup_food_rec_model():

    def __init__(self, train_loader, num_class, max_epoch, warmup_epoch, 
    device, out_dir):

        self.train_loader = train_loader
        self.model = resnet18(weights = ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_class)
        self.model.to(device)
        
        self.train_prog = np.empty((0,2))
        self.next_epoch = 1
        self.warmup_epoch = warmup_epoch
        self.warmup_rate = 0.2/warmup_epoch
        self.max_epoch = max_epoch

        self.loss_fn = nn.CrossEntropyLoss()
        self.optim = SGD(self.model.parameters(), lr=1e-12, momentum=0.9, nesterov=True)
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optim, max_epoch - warmup_epoch, eta_min=0.0, last_epoch=-1)

        self.device = device
        self.out_dir = out_dir

    def adj_optim_lr(self):
        if self.next_epoch <= self.warmup_epoch:
            next_lr = 1e-12 + self.next_epoch * self.warmup_rate
            print(f'lr at epoch {self.next_epoch} = {next_lr}')
            for group in self.optim.param_groups:
                group["lr"] = next_lr
        else:
            self.scheduler.step()
            next_lr = self.scheduler.get_last_lr().item()
            print(f'lr at epoch {self.next_epoch} = {next_lr}')

    def train_step(self, batch):
        img, act_label = batch[0].to(self.device), batch[1].to(self.device)

        # forward
        pred_label = self.model(img)
        loss = self.loss_fn(pred_label, act_label)

        # backward
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss.item()

    def train(self):
        print(f'Training with max epoch {self.max_epoch} ...')

        for epoch in range(self.next_epoch, self.max_epoch+1):

            self.adj_optim_lr()
            epoch_step_loss = np.array([])

            for step, batch in enumerate(self.train_loader):
                step_loss = self.train_step(batch)
                epoch_step_loss = np.append(epoch_step_loss, step_loss)
                if (step+1) % 100 == 0:
                    with torch.no_grad():
                        print(f'Epoch: {epoch}, Step: {step+1}, CELoss = {step_loss:.4f}')
                        
            epoch_mean_loss = np.mean(epoch_step_loss)
            self.train_prog = np.append(self.train_prog, np.array([[epoch, epoch_mean_loss]]), axis=0)

            self.next_epoch += 1
            self.save_state()

            
            


    def save_state(self):
        state = {
            "next_epoch": self.next_epoch,
            "cnn_model": self.model.state_dict(),
            "optim": self.optim.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "train_prog": self.train_prog
        }
        output_name = 'supervised_offline.pt'
        output_path = os.path.join(self.out_dir, output_name)
        torch.save(state, output_path)
        print("Saved state at " + output_path)
    
    def load_state(self, state_file):
        file_path = os.path.join(self.out_dir, state_file)
        state = torch.load(file_path)
        self.model.load_state_dict(state['cnn_model'])
        self.optim.load_state_dict(state['optim'])
        if 'scheduler' in state.keys():
            self.scheduler.load_state_dict(state['scheduler'])
        self.next_epoch = state['next_epoch']
        self.train_prog = state['train_prog']
        print('Loaded state from ' + str(file_path))

    def predict(self, img_batch):
        return(self.model(img_batch))

    @torch.no_grad()
    def eval_perf(self, test_loader):
        print('Evaluating performance...')
        total_good_pred = 0
        total_loss = 0

        for step, batch in enumerate(test_loader):
            img, act_label = batch[0].to(device), batch[1].to(device)
            pred_label_prob = self.model(img)

            TCELoss = nn.CrossEntropyLoss(reduction='sum')
            step_loss = TCELoss(pred_label_prob, act_label)
            total_loss += step_loss.item()

            _, pred_label = torch.max(pred_label_prob, 1)
            step_good_pred = (pred_label==act_label).sum().item()
            total_good_pred += step_good_pred

            if (step+1) % 100 == 0 :
                print('Done up to step ' + str(step+1) + '...')

        num_test_samples = len(test_loader.dataset)
        print(num_test_samples)
        
        acc = total_good_pred / num_test_samples
        ACELoss = total_loss / num_test_samples

        print(f'After {self.next_epoch-1} epochs:')
        print(f'Average Cross Entropy Loss = {ACELoss:.4f}.')
        print(f'Accuracy = {acc:.4f}.') 
        
    def show_train_prog(self):
        x_step = self.train_prog[:,0]
        y_loss = self.train_prog[:,1]
        plt.plot(x_step, y_loss, 'ro')
        plt.show()


output_dir = os.path.join(main_dir, 'output_model')
model = sup_food_rec_model(off_train_loader, num_off_class,
max_epoch, wu_epoch, device, output_dir)

if state_path is not None:
    model.load_state(state_path)

if to_train:
    model.train()




### Evaluate performance on test set
if eval_perf:
    model.eval_perf(off_test_loader)

### Train (epoch) loss graph
if show_train_loss:
    model.show_train_prog()
