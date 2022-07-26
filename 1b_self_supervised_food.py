import torch
import os

from utils.data_utils import \
    get_f1_ord_dloader, get_f1_off_relic_dloader, get_f1_eval_train_dloader, \
    get_uec_dloader, get_uec_off_relic_dloader, get_uec_eval_train_loader

from models.offline_models import ReLIC_off

main_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(main_dir,'data')
f1_root_dir = os.path.join(data_dir, 'food-101')
uec_root_dir = os.path.join(data_dir, 'uec256')

## parameter
ds = 'uec'
#state_path = None
state_path = 'relic_offline_e100.pt'

to_train = False
eval_perf = True
show_train_loss = False

max_epoch = 100
wu_epoch = 10
batch_s = 20
lr = 0.2


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Found GPU device: {}".format(torch.cuda.get_device_name(0)))
else:
    device = torch.device("cpu")
    print("Using CPU")




# Prepare data
if ds == 'f1':
    print('Preparing food101 dataset...')

    data_dir = os.path.join(main_dir,'data')

    print('Checking dataset...')

    if os.path.exists(f1_root_dir):

        print('Obtaining dataloader for ReLIC offline learning...')
        _, f1_off_test_loader, num_off_class = get_f1_ord_dloader(f1_root_dir=f1_root_dir, ltype = "off", batch_size=batch_s)
        f1_off_relic_train_loader = get_f1_off_relic_dloader(f1_root_dir, batch_size=batch_s)

        relic_train_loader = f1_off_relic_train_loader
        test_loader = f1_off_test_loader

    else:
        print('Could not locate food101 data!')

elif ds == 'uec':
    print('Preparing uec dataset...')

    print('Checking dataset...')
    if os.path.exists(uec_root_dir):
        print('Obtaining dataloader for ReLIC offline learning...')
        relic_train_loader = get_uec_off_relic_dloader(uec_root_dir=uec_root_dir, batch_size=batch_s)
        test_loader, num_off_class = get_uec_dloader(uec_root_dir=uec_root_dir, ltype='off', split ='test', batch_size=batch_s)
    else:
        print('Could not locate uec256 data!')


# Model and training
output_dir = os.path.join(main_dir, 'output_model', ds)
model = ReLIC_off(relic_train_loader, max_epoch, wu_epoch, lr, device, output_dir)

if state_path is not None:
    model.load_state(state_path)

if to_train:
    model.train_model()




### Evaluate performance on test set
if eval_perf:
    if ds == 'f1':
        eval_train_loader = get_f1_eval_train_dloader(f1_root_dir=f1_root_dir, batch_size=batch_s)
    elif ds == 'uec':
        eval_train_loader = get_uec_eval_train_loader(uec_root_dir=uec_root_dir, batch_size=batch_s)
    
    print('Evaluating performance on train set...')
    model.linear_eval_perf(eval_train_loader, num_off_class)
    print('Evaluating performance on test set...')
    model.linear_eval_perf(test_loader, num_off_class)

### Train (epoch) loss graph
if show_train_loss:
    model.show_train_prog(ds)