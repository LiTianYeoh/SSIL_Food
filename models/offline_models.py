import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler
from torchvision.models import resnet18, ResNet18_Weights
import os.path
import matplotlib.pyplot as plt
from utils.data_utils import get_ftr_dloader


class MLP(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc2(self.relu(self.bn1(self.fc1(x))))

class RelicLoss(nn.Module):

    def __init__(self, normalize=True, temperature=1.0, alpha=0.5):
        super(RelicLoss, self).__init__()
        self.normalize = normalize
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, zi, zj, z_orig):
        bs = zi.shape[0]
        labels = torch.zeros((2*bs,)).long().to(zi.device)
        mask = torch.ones((bs, bs), dtype=bool).fill_diagonal_(0)

        if self.normalize:
            zi_norm = F.normalize(zi, p=2, dim=-1)
            zj_norm = F.normalize(zj, p=2, dim=-1)
            zo_norm = F.normalize(z_orig, p=2, dim=-1)
        else:
            zi_norm = zi
            zj_norm = zj
            zo_norm = z_orig

        logits_ii = torch.mm(zi_norm, zi_norm.t()) / self.temperature
        logits_ij = torch.mm(zi_norm, zj_norm.t()) / self.temperature
        logits_ji = torch.mm(zj_norm, zi_norm.t()) / self.temperature
        logits_jj = torch.mm(zj_norm, zj_norm.t()) / self.temperature

        logits_ij_pos = logits_ij[torch.logical_not(mask)]                                          # Shape (N,)
        logits_ji_pos = logits_ji[torch.logical_not(mask)]                                          # Shape (N,)
        logits_ii_neg = logits_ii[mask].reshape(bs, -1)                                             # Shape (N, N-1)
        logits_ij_neg = logits_ij[mask].reshape(bs, -1)                                             # Shape (N, N-1)
        logits_ji_neg = logits_ji[mask].reshape(bs, -1)                                             # Shape (N, N-1)
        logits_jj_neg = logits_jj[mask].reshape(bs, -1)                                             # Shape (N, N-1)

        pos = torch.cat((logits_ij_pos, logits_ji_pos), dim=0).unsqueeze(1)                         # Shape (2N, 1)
        neg_i = torch.cat((logits_ii_neg, logits_ij_neg), dim=1)                                    # Shape (N, 2N-2)
        neg_j = torch.cat((logits_ji_neg, logits_jj_neg), dim=1)                                    # Shape (N, 2N-2)
        neg = torch.cat((neg_i, neg_j), dim=0)                                                      # Shape (2N, 2N-2)

        logits = torch.cat((pos, neg), dim=1)                                                       # Shape (2N, 2N-1)
        contrastive_loss = F.cross_entropy(logits, labels)

        logits_io = torch.mm(zi_norm, zo_norm.t()) / self.temperature
        logits_jo = torch.mm(zj_norm, zo_norm.t()) / self.temperature
        probs_io = F.softmax(logits_io[torch.logical_not(mask)], -1)
        probs_jo = F.log_softmax(logits_jo[torch.logical_not(mask)], -1)
        kl_div_loss = F.kl_div(probs_io, probs_jo, log_target=True, reduction="sum")
        return contrastive_loss + self.alpha * kl_div_loss

class OnlineNetwork(nn.Module):
    
    def __init__(self, encoder, encoder_dim, projection_dim):
        super(OnlineNetwork, self).__init__()
        self.encoder = encoder 
        self.proj_head = MLP(encoder_dim, projection_dim)
        self.pred_head = MLP(projection_dim, projection_dim)

    def forward(self, x):
        x = self.pred_head(self.proj_head(self.encoder(x)))
        return F.normalize(x, dim=-1, p=2)




class SUP_off():

    def __init__(self, train_loader, num_class, max_epoch, warmup_epoch, lr,
    device, out_dir):

        self.train_loader = train_loader
        self.model = resnet18(weights = ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_class)
        self.model.to(device)
        
        self.train_prog = np.empty((0,2))
        self.next_epoch = 1
        self.warmup_epoch = warmup_epoch
        self.warmup_rate = lr/warmup_epoch
        self.max_epoch = max_epoch

        self.loss_fn = nn.CrossEntropyLoss()
        self.optim = SGD(self.model.parameters(), lr=1e-12, momentum=0.9, nesterov=True)
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optim, max_epoch - warmup_epoch, eta_min=0.0, last_epoch=-1)

        self.device = device
        self.out_dir = out_dir

    def adj_optim_lr(self):
        if self.next_epoch <= self.warmup_epoch:
            next_lr = 1e-12 + self.next_epoch * self.warmup_rate
            print(f'lr at epoch {self.next_epoch} = {next_lr:.6f}')
            for group in self.optim.param_groups:
                group["lr"] = next_lr
        else:
            self.scheduler.step()
            next_lr = self.scheduler.get_last_lr()[0]
            print(f'lr at epoch {self.next_epoch} = {next_lr:.6f}')

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

    def train_model(self):
        self.model.train()
        print(f'Training with max epoch {self.max_epoch} ...')

        for epoch in range(self.next_epoch, self.max_epoch+1):

            self.adj_optim_lr()
            epoch_step_loss = np.array([])

            for step, batch in enumerate(self.train_loader):
                step_loss = self.train_step(batch)
                epoch_step_loss = np.append(epoch_step_loss, step_loss)
                if (step+1) % 100 == 0:
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
        self.model.eval()
        
        total_good_pred = 0
        total_loss = 0

        for step, batch in enumerate(test_loader):
            img, act_label = batch[0].to(self.device), batch[1].to(self.device)
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
        
        acc = total_good_pred / num_test_samples
        ACELoss = total_loss / num_test_samples

        print(f'After {self.next_epoch-1} epochs:')
        print(f'Average Cross Entropy Loss = {ACELoss:.4f}.')
        print(f'Accuracy = {acc:.4f}.') 

        self.model.train()
        
    def show_train_prog(self):
        x_step = self.train_prog[:,0]
        y_loss = self.train_prog[:,1]
        plt.plot(x_step, y_loss, 'ro')
        plt.show()



class ReLIC_off():

    def __init__(self, train_loader, max_epoch, warmup_epoch, lr, device, out_dir):

        self.train_loader = train_loader
        encoder = resnet18(weights = ResNet18_Weights.IMAGENET1K_V1)
        encoder_dim = encoder.fc.in_features
        encoder.fc = nn.Flatten(start_dim=1)
        self.model = OnlineNetwork(encoder, encoder_dim, 128)
        self.model.to(device)

        self.train_prog = np.empty((0,2))
        self.next_epoch = 1
        self.warmup_epoch = warmup_epoch
        self.warmup_rate = lr/warmup_epoch
        self.max_epoch = max_epoch

        self.loss_fn = RelicLoss()
        self.optim = SGD(self.model.parameters(), lr=1e-12, momentum=0.9, nesterov=True)
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optim, max_epoch - warmup_epoch, eta_min=0.0, last_epoch=-1)

        self.device = device
        self.out_dir = out_dir

    def adj_optim_lr(self):
        if self.next_epoch <= self.warmup_epoch:
            next_lr = 1e-12 + self.next_epoch * self.warmup_rate
            print(f'lr at epoch {self.next_epoch} = {next_lr:.6f}')
            for group in self.optim.param_groups:
                group["lr"] = next_lr
        else:
            self.scheduler.step()
            next_lr = self.scheduler.get_last_lr()[0]
            print(f'lr at epoch {self.next_epoch} = {next_lr:.6f}')

    def train_step(self, batch):
        img_orig, img_1, img_2 = batch["img"].to(self.device), batch["aug_1"].to(self.device), batch["aug_2"].to(self.device)

        # forward
        orig_ftr = self.model(img_orig)
        img_1_ftr = self.model(img_1)
        img_2_ftr = self.model(img_2)
        loss = self.loss_fn(img_1_ftr, img_2_ftr, orig_ftr)

        # backward
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss.item()

    def train_model(self):
        self.model.train()
        print(f'Training with max epoch {self.max_epoch} ...')

        for epoch in range(self.next_epoch, self.max_epoch+1):

            self.adj_optim_lr()
            epoch_step_loss = np.array([])

            for step, batch in enumerate(self.train_loader):
                step_loss = self.train_step(batch)
                epoch_step_loss = np.append(epoch_step_loss, step_loss)
                if (step+1) % 100 == 0:
                    print(f'Epoch: {epoch}, Step: {step+1}, ReLIC_Loss = {step_loss:.4f}')
                        
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
        output_name = 'relic_offline.pt'
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

    @torch.no_grad()
    def build_ftrs_dataloader(self, data_loader):
        print('Building feature data loader...')
        ftrvecs, act_label = [], []
        for step, batch in enumerate(data_loader):
            img, lab = batch[0].to(self.device), batch[1].detach().cpu().numpy()
            ftr = self.model(img).detach().cpu().numpy()
            ftrvecs.append(ftr), act_label.append(lab)
            if (step+1) % 200 == 0:
                print(f'Done feature up to step {step+1}...')
        
        ftrvecs, act_label = np.concatenate(ftrvecs, axis=0), np.concatenate(act_label, axis=0)
        ftr_data_loader = get_ftr_dloader(ftrvec=ftrvecs, label=act_label, batch_size=512)

        return ftr_data_loader

    def linear_eval_perf(self, test_loader, num_class):
        self.model.eval()

        print('Evaluating performance of features with linear classifier...')

        ftr_dloader = self.build_ftrs_dataloader(test_loader)
        clf_head = nn.Linear(128, num_class).to(self.device)
        clf_optim = torch.optim.SGD(params = clf_head.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
        loss_fn = nn.CrossEntropyLoss()

        print('Training linear classifier... ')
        for epoch in range(1,201):
            for step, batch in enumerate(ftr_dloader):
                ftrvecs, lab = batch[0].to(self.device), batch[1].to(self.device)
                pred_lab = clf_head(ftrvecs)
                loss = loss_fn(pred_lab, lab)

                clf_optim.zero_grad()
                loss.backward()
                clf_optim.step()
            
            if (epoch+1) % 10 == 0:
                print(f'Done training LC Epoch {epoch+1}/200. CELoss = {loss.item():.4f}')

        print('Evaluating accuracy of linear classfier...')
        with torch.no_grad():
            total_good_pred = 0
            for step, batch in enumerate(ftr_dloader):
                ftrvecs, act_lab = batch[0].to(self.device), batch[1].to(self.device)
                pred_lab_prob = clf_head(ftrvecs)

                _, pred_label = torch.max(pred_lab_prob, 1)
                step_good_pred = (pred_label==act_lab).sum().item()
                total_good_pred += step_good_pred

                if (step+1) % 100 == 0 :
                    print('Done up to step ' + str(step+1) + '...')

            num_test_samples = len(ftr_dloader.dataset)
            acc = total_good_pred / num_test_samples

        print(f'After {self.next_epoch-1} epochs:')
        print(f'Linear Classifier Accuracy = {acc:.4f}.') 

        self.model.train()

        
    def show_train_prog(self):
        x_step = self.train_prog[:,0]
        y_loss = self.train_prog[:,1]
        plt.plot(x_step, y_loss, 'ro')
        plt.show()