import torch
from torchvision.datasets import Food101
import os

from utils.data_utils import get_f1_ord_dloader
from utils.inc_utils import ModelWrapper, pool_feat, slda_eval_acc
from models.offline_models import ReLIC_off
from models.inc_slda import StreamingLDA

main_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(main_dir,'data')
f1_root_dir = os.path.join(data_dir, 'food-101')
vf_root_dir = os.path.join(data_dir, 'vf172')

## parameter

off_state_path = 'relic_offline_e100.pt'
inc_state_name = None #'relic_inc'

output_layers = ['encoder.layer4.1']
batch_s = 64

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Found GPU device: {}".format(torch.cuda.get_device_name(0)))
else:
    device = torch.device("cpu")
    print("Using CPU")

# Prepare data
print('Preparing food101 dataset...')

print('Checking dataset...')
check_f101 = Food101(root=data_dir, split='train', download=True)

print('Obtaining dataloader for base class (offline learning)...')
f1_off_train_loader, f1_off_test_loader, num_off_class = get_f1_ord_dloader(f1_root_dir=f1_root_dir, off_inc = "off", batch_size=batch_s)

off_train_loader = f1_off_train_loader
off_test_loader = f1_off_test_loader

print('Obtaining dataloader for new class (incremental learning)...')
f1_inc_train_loader, f1_inc_test_loader, num_inc_class = get_f1_ord_dloader(f1_root_dir=f1_root_dir, off_inc = "inc", batch_size=batch_s)

inc_train_loader = f1_inc_train_loader
inc_test_loader = f1_inc_test_loader


output_dir = os.path.join(main_dir, 'output_model')
off_model = ReLIC_off(off_train_loader, 100, 10, 0.2, device, output_dir)
off_model.load_state(off_state_path)



## prepare slda
ftr_size = 512
ftr_extraction_wrapper = ModelWrapper(off_model.model.eval(), output_layers, return_single=True).eval()
num_tot_class = num_off_class + num_inc_class
slda_classifier = StreamingLDA(input_shape=ftr_size, num_classes=num_tot_class, streaming_update_sigma=False)


if inc_state_name is not None:
    slda_classifier.load_model(save_path=output_dir, save_name=inc_state_name)
    slda_path = os.path.join(output_dir, inc_state_name + '.pth')
    print('Loaded SLDA classifier state from ' + str(slda_path))
else:
    # Fit for base (offline) data
    print('Initialising SLDA with base class...')

    base_init_data = torch.empty((len(off_train_loader.dataset), ftr_size))
    base_init_labels = torch.empty(len(off_train_loader.dataset)).long()

    print('Generating features for base class...')

    start = 0
    for step, batch in enumerate(off_train_loader):

        batch_x = batch[0].to(device)
        batch_y = batch[1]

        batch_x_feat = ftr_extraction_wrapper(batch_x)
        batch_x_feat = pool_feat(batch_x_feat)

        end = start + batch_x_feat.shape[0]
        base_init_data[start:end] = batch_x_feat.cpu()
        base_init_labels[start:end] = batch_y.squeeze()
        start = end

        if (step+1) % 100 == 0:
            print(f'Done step {step+1}...')

    print('Fitting slda with features from base class...')
    slda_classifier.fit_base(base_init_data, base_init_labels)

    print('Evaluating accuracy before introducing new classes:')
    print('Evaluating accuracy on base (off) test set...')
    off_test_acc = slda_eval_acc(slda_classifier, ftr_extraction_wrapper, off_test_loader)
    print(f'Accuracy = {off_test_acc:.4f}.') 

    # Train SLDA incrementally
    print('Training SLDA with new class...')

    for step, batch in enumerate(inc_train_loader):

        batch_x = batch[0].to(device)
        batch_y = batch[1]

        batch_x_feat = ftr_extraction_wrapper(batch_x)
        batch_x_feat = pool_feat(batch_x_feat)

        for x, y in zip(batch_x_feat, batch_y):
            slda_classifier.fit(x.cpu(), y.view(1, ))

        if (step+1) % 100 == 0:
            print(f'Done step {step+1}...')


    # Save
    print('Saving SLDA classifier...')
    slda_classifier.save_model(save_path=output_dir, save_name='relic_inc')

# Evaluate accuracy
print('Evaluating final accuracy:')
print('Evaluating accuracy on base (off) test set...')
off_test_acc = slda_eval_acc(slda_classifier, ftr_extraction_wrapper, off_test_loader)
print(f'Accuracy = {off_test_acc:.4f}.') 

print('Evaluating accuracy on new (inc) test set...')
inc_test_acc = slda_eval_acc(slda_classifier, ftr_extraction_wrapper, inc_test_loader)
print(f'Accuracy = {inc_test_acc:.4f}.') 