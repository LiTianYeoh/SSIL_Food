from multiprocessing import pool
import torch
import torch.nn as nn
import numpy as np


def get_name_to_module(model):
    name_to_module = {}
    for m in model.named_modules():
        name_to_module[m[0]] = m[1]
    return name_to_module


def get_activation(all_outputs, name):
    def hook(model, input, output):
        all_outputs[name] = output.detach()

    return hook


def add_hooks(model, outputs, output_layer_names):
    """
    :param model:
    :param outputs: Outputs from layers specified in `output_layer_names` will be stored in `output` variable
    :param output_layer_names:
    :return:
    """
    name_to_module = get_name_to_module(model)
    for output_layer_name in output_layer_names:
        name_to_module[output_layer_name].register_forward_hook(get_activation(outputs, output_layer_name))


class ModelWrapper(nn.Module):
    def __init__(self, model, output_layer_names, return_single=False):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.output_layer_names = output_layer_names
        self.outputs = {}
        self.return_single = return_single
        add_hooks(self.model, self.outputs, self.output_layer_names)

    def forward(self, x):
        self.model(x)
        output_vals = [self.outputs[output_layer_name] for output_layer_name in self.output_layer_names]
        if self.return_single:
            return output_vals[0]
        else:
            return output_vals


def pool_feat(features):
    feat_size = features.shape[-1]
    num_channels = features.shape[1]
    features2 = features.permute(0, 2, 3, 1)  # 1 x feat_size x feat_size x num_channels
    features3 = torch.reshape(features2, (features.shape[0], feat_size * feat_size, num_channels))
    feat = features3.mean(1)  # mb x num_channels
    return feat

def slda_predict(slda, ftr_ext, img):
    img_ftr = ftr_ext(img)
    img_ftr = pool_feat(img_ftr)

    probas = slda.predict(img_ftr.cuda(), return_probas=True)

    return probas

def slda_eval_acc(slda, ftr_ext, test_loader):
    total_good_pred = 0
    num_test_samples = len(test_loader.dataset)

    for step, batch in enumerate(test_loader):
        batch_x = batch[0].cuda()
        batch_y = batch[1]

        pred_y_prob = slda_predict(slda, ftr_ext, batch_x)
        _, pred_y = torch.max(pred_y_prob, 1)
        step_good_pred = (pred_y.numpy()==batch_y.numpy()).sum()
        total_good_pred += step_good_pred

        if (step+1) % 100 == 0 :
            print('Done up to step ' + str(step+1) + '...')
    
    acc = total_good_pred / num_test_samples
    print(f'Accuracy = {acc:.4f}.') 


