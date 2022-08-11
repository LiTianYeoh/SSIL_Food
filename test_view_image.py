import torch
from torchvision import transforms
import os

from utils.data_utils import get_f1_pure_dloader
from models.offline_models import SUP_off
from utils.img_utils import view_sample_img

main_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(main_dir,'data')
f1_root_dir = os.path.join(data_dir, 'food-101')
vf_root_dir = os.path.join(data_dir, 'vf172')

## parameter
#state_path = None
state_path = 'supervised_offline_e100.pt'
dloader_type = 'off'
dloader_split = 'test'
batch_s = 64


## unused
max_epoch = 100
wu_epoch = 10

lr = 0.2

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Found GPU device: {}".format(torch.cuda.get_device_name(0)))
else:
    device = torch.device("cpu")
    print("Using CPU")


dloader, num_off_class = get_f1_pure_dloader(f1_root_dir=f1_root_dir, off_inc = dloader_type, split = dloader_split, batch_size = batch_s)
print('Obtained pure data loader for image viewing purpose...')

output_dir = os.path.join(main_dir, 'output_model')
model = SUP_off(dloader, num_off_class, max_epoch, wu_epoch, lr, device, output_dir)
model.load_state(state_path)

norm_trans = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])

def model_pred(img_gpu):
    trans_img = norm_trans(img_gpu)
    pred_label_prob = model.predict(trans_img)
    _, pred_label = torch.max(pred_label_prob, 1)
    
    return pred_label.cpu().numpy()

view_sample_img(dloader, model_pred, device = device)