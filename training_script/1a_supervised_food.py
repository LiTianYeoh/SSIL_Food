import torch
import numpy as np
from torch import nn 
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision.models import resnet18, ResNet18_Weights
from torch.optim import SGD
from torchvision import transforms
import os
import yaml
import matplotlib.pyplot as plt

main_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

## parameter
#state_path = None
state_path = 'supervised_offline_e008.pt'

to_train = False
eval_perf = True
show_train_loss = True

max_epoch = 100
wu_epoch = 10
batch_s = 64
num_class = 101
opt_param = {
    'name': 'sgd',
    'lr': 0.2,
    'momentum': 0.9,
    'nesterov': True
}


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

input_size = 224
img_preproc = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

food_train = torchvision.datasets.Food101(root=data_dir, split='train', transform=img_preproc, download=True)
food_test = torchvision.datasets.Food101(root=data_dir, split='test', transform=img_preproc, download=True)

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




off_train_loader = DataLoader(off_train_ds, batch_size = batch_s, shuffle = True, num_workers = 4)
off_test_loader = DataLoader(off_test_ds, batch_size = batch_s, shuffle = True, num_workers = 4)

# 1 model and training
class sup_food_reg_model():

    def __init__(self, train_loader, num_class, optim_param, max_epoch, warmup_epoch, 
    device, out_dir):

        self.train_loader = train_loader
        self.model = resnet18(weights = ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_class)
        self.model.to(device)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optim_param = optim_param
        
        self.train_prog = np.empty((0,2))
        self.next_epoch = 1
        self.warmup_epoch = warmup_epoch
        self.max_epoch = max_epoch
        self.device = device
        self.out_dir = out_dir

        self.update_optim(optim_param, 1)

    def update_optim(self, optim_param, epoch):
        if epoch <= self.warmup_epoch:
            curr_lr = optim_param['lr']*epoch/self.warmup_epoch
            if optim_param['name'] == 'sgd':
                self.optim = SGD(self.model.parameters(), lr = curr_lr, 
                momentum = optim_param['momentum'], nesterov = optim_param['nesterov'])

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
            self.update_optim(self.optim_param, self.next_epoch)
            self.save_state()


    def save_state(self):
        state = {
            "next_epoch": self.next_epoch,
            "cnn_model": self.model.state_dict(),
            "optim": self.optim.state_dict(),
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

            pred_label = [np.argmax(prob_dist) for prob_dist in pred_label_prob.cpu().numpy()]
            step_good_pred = np.equal(pred_label, act_label.cpu().numpy()).sum()
            total_good_pred += step_good_pred

            if (step+1) % 100 == 0 :
                print('Done up to step ' + str(step+1) + '...')

        num_test_samples = len(test_loader.dataset)
        
        acc = total_good_pred / num_test_samples
        ACELoss = total_loss / num_test_samples

        print(f'After {self.next_epoch-1} epochs:')
        print(f'Average Cross Entropy Loss = {ACELoss:.4f}.')
        print(f'Accuracy = {acc:.4f}.') 
        
    def show_train_prog(self):
        x_step = self.train_prog[:,0]
        y_loss = self.train_prog[:,1]
        print(self.train_prog.size)
        plt.plot(x_step, y_loss, 'ro')
        plt.show()


output_dir = os.path.join(main_dir, 'output_model')
model = sup_food_reg_model(off_train_loader, num_off_class, opt_param, 
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
