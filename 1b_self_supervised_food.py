import torch
from torchvision.datasets import Food101
import os

from utils.data_utils import get_f1_ord_dloader, get_f1_off_relic_dloader
from models.offline_models import ReLIC_off

main_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(main_dir,'data')
f1_root_dir = os.path.join(data_dir, 'food-101')
vf_root_dir = os.path.join(data_dir, 'vf172')

## parameter
state_path = None
state_path = 'relic_offline_e004.pt'

to_train = False
eval_perf = True
show_train_loss = True

max_epoch = 100
wu_epoch = 10
batch_s = 20
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




# Prepare data
print('Preparing food101 dataset...')

data_dir = os.path.join(main_dir,'data')

print('Checking dataset...')

check_f101 = Food101(root=data_dir, split='train', download=True)

print('Obtaining dataloader for supervised offline learning...')
f1_off_train_loader, f1_off_test_loader, num_off_class = get_f1_ord_dloader(f1_root_dir=f1_root_dir, off_inc = "off", batch_size=batch_s)
f1_off_relic_train_loader = get_f1_off_relic_dloader(f1_root_dir, batch_size=batch_s)


relic_train_loader = f1_off_relic_train_loader
train_loader = f1_off_train_loader
test_loader = f1_off_test_loader



# Model and training
output_dir = os.path.join(main_dir, 'output_model')
model = ReLIC_off(relic_train_loader, max_epoch, wu_epoch, device, output_dir)

if state_path is not None:
    model.load_state(state_path)

if to_train:
    model.train_model()




### Evaluate performance on test set
if eval_perf:
    model.linear_eval_perf(test_loader, 101)

### Train (epoch) loss graph
if show_train_loss:
    model.show_train_prog()