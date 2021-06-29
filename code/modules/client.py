from torch import optim
import numpy as np
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from utils.AverageMeter import *

device = torch.device('cuda:0')


class Client:
    def __init__(self, num_classes, con, lambda_a, lambda_i):
        """
       Client init method

       Parameters:
           args: List containing arguments to configure the training

       Returns:
           None
       """
        self.num_classes = num_classes
        self.client_id = None
        self.local_model = self.build_models()
        self.train_loader = None
        self.clss_weights = []
        self.train_loader_u = None
        self.val_loader = None
        self.steps = 0
        self.con = con
        self.lambda_a = lambda_a
        self.lambda_i = lambda_i
        self.mu = 1e-2
        self.criterion = nn.CrossEntropyLoss()
        self.kl_divergence = nn.KLDivLoss()
        self.mse = torch.nn.MSELoss()
        self.peer = self.build_models()

    def build_models(self):
        """
          create an efficientnet pretrained model

          Parameters:
              None

          Returns:
              None
        """
        model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=self.num_classes)
        num_ftrs = model._fc.in_features
        model._fc = nn.Linear(num_ftrs, self.num_classes)
        model = nn.DataParallel(model)
        model = model.cuda()
        model = model.to(device)
        return model

    def set(self, client_id, global_model, train_loader, val_loader, train_loader_u, class_weights):
        """
         initials client

         Parameters:
             client_id: client id
             global_model: global model weights
             train_loader: client training dataloader
             val_loader: client validation dataloader
             train_loader_u: client unlabeled dataloader
             class_weights: class weights

         Returns:
             None
       """
        self.client_id = client_id
        self.local_model.load_state_dict(global_model)
        self.clss_weights = class_weights
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_loader_u = train_loader_u

    def un_supervised_loss(self, x, x_aug, peers_wghts):
        """
         unsupervised loss

         Parameters:
             x: input batch
             x_aug: augment input batch
             peers_wghts: similar peers

         Returns:
             loss
        """
        loss_u = 0
        ce_lss = 0
        ls_u = 0
        if not self.peers_found:
            with torch.no_grad():
                y_pred = self.local_model(x)
            prob = torch.softmax(y_pred.detach(), dim=-1)
            mx_prob, mx_ind = torch.max(prob, axis=1)
            mx_prob = (mx_prob >= self.con) * mx_prob
            conf = [i for i in range(len(mx_ind)) if mx_prob[i] != 0]
            if len(conf) > 0:
                x_conf = x_aug[conf]
                y_hard = self.local_model(x_conf)
                pseudo_label = mx_ind[conf]
                loss_u += self.criterion(y_hard, pseudo_label) * self.lambda_a
            return loss_u, len(conf), 0, 0
        else:
            with torch.no_grad():
                y_pred = self.local_model(x)
                prob = torch.softmax(y_pred.detach(), dim=-1)
            probs = []
            for wght in peers_wghts:
                self.peer.load_state_dict(wght)
                self.peer.eval()
                with torch.no_grad():
                    peer_pred = self.peer(x)
                probs.append(torch.softmax(peer_pred.detach(), dim=-1))

            sum_prob = np.sum([prob, np.sum(probs)], axis=0) / (len(peers_wghts) + 1)
            sum_prob, mx_ind = torch.max(sum_prob, axis=1)
            sum_prob = (sum_prob >= self.con) * sum_prob
            # sum_prob = (sum_prob >=0.6)*sum_prob
            conf = [i for i in range(len(mx_ind)) if sum_prob[i] != 0]
            if len(conf) > 0:
                x_conf = x_aug[conf]
                y_hard = self.local_model(x_conf)
                pseudo_label = mx_ind[conf]
                loss_u += self.criterion(y_hard, pseudo_label) * self.lambda_a
                ce_lss = loss_u.item()
                for i in range(len(peers_wghts)):
                    loss_u += (self.mse(probs[i], prob) / len(peers_wghts)) * self.lambda_i
                ls_u = loss_u.item()
            return loss_u, len(conf), ce_lss, (ls_u - ce_lss)

    def train(self, client_id, curr_round, steps, curr_lr, peers_wghts, fed_prox, global_model):
        """
         trains client locally

         Parameters:
             client_id:
             curr_round:
             steps:
             curr_lr:
             peers_wghts:
             fed_prox:
             global_model

         Returns:
             client state
       """

        self.steps = steps
        self.local_model.train()
        if peers_wghts != None:
            self.peers_found = True
        else:
            self.peers_found = False

        optimizer = optim.Adam(self.local_model.parameters(), lr=curr_lr)
        criterion = nn.CrossEntropyLoss(weight=torch.cuda.FloatTensor(self.clss_weights))
        train_loss = AverageMeter()
        train_acc = AverageMeter()
        trainloader_l_iter = enumerate(self.train_loader)
        trainloader_u_iter = enumerate(self.train_loader_u)

        conf_cont = []
        ce_lsses = []
        mse_lsses = []

        for i in range(self.steps):
            # Check if the label loader has a batch available
            try:
                _, sample_batched = next(trainloader_l_iter)
            except:
                # Curr loader doesn't have data, then reload data
                del trainloader_l_iter
                trainloader_l_iter = enumerate(self.train_loader)
                _, sample_batched = next(trainloader_l_iter)

            try:
                _, sample_batched_u = next(trainloader_u_iter)
            except:
                del trainloader_u_iter
                trainloader_u_iter = enumerate(self.train_loader_u)
                _, sample_batched_u = next(trainloader_u_iter)

            x = sample_batched[0].type(torch.cuda.FloatTensor)
            y = sample_batched[1].type(torch.cuda.LongTensor)
            x_u = sample_batched_u[0].type(torch.cuda.FloatTensor)
            x_u_aug = sample_batched_u[2].type(torch.cuda.FloatTensor)
            n = x.size(0)

            with torch.set_grad_enabled(True):
                output = self.local_model(x)
                loss = criterion(output, y)
                loss_u, conf, ce_lss, mse_lss = self.un_supervised_loss(x_u, x_u_aug, peers_wghts)
                loss += loss_u

                conf_cont.append(conf)
                ce_lsses.append(ce_lss)
                mse_lsses.append(mse_lss)

                #########################we implement FedProx Here###########################
                # referring to https://github.com/IBM/FedMA/blob/4b586a5a22002dc955d025b890bc632daa3c01c7/main.py#L819
                if fed_prox == 'True' and i > 0 and curr_round > 0:
                    w_diff = torch.tensor(0., device=device)
                    for w, w_t in zip(global_model.parameters(), self.local_model.parameters()):
                        w_diff += torch.pow(torch.norm(w - w_t), 2)
                    loss += self.mu / 2. * w_diff
                #############################################################################

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                prediction = output.cpu().max(1, keepdim=True)[1]
                train_acc.update(prediction.cpu().eq(y.cpu().view_as(prediction.cpu())).sum().item() / n)
                train_loss.update(loss.item())

        print('rnd:{}, clnt{}: trlss:{}, tracc:{}'.format(curr_round, client_id, round(train_loss.avg, 4),
                                                          round(train_acc.avg, 4)))
        return (self.steps, self.local_model.state_dict(), train_loss, train_acc, np.sum(conf_cont), np.mean(ce_lsses),
                np.mean(mse_lsses))
        # return (len(train_loader)*batch_size, local_model.state_dict(), train_loss, train_acc)

    def validate(self, global_model, val_loader, clss_weights):
        """
        validate global mode on client local data

        Parameters:
            global_model: global mode weight
            val_loader:
            clss_weights:

        Returns:
            validation loss
      """
        self.local_model.load_state_dict(global_model)
        self.local_model.eval()
        val_loss = AverageMeter()
        val_acc = AverageMeter()
        criterion = nn.CrossEntropyLoss(weight=torch.cuda.FloatTensor(clss_weights))
        for batch_idx, sample_batched in enumerate(val_loader):
            x = sample_batched[0].type(torch.cuda.FloatTensor)
            y = sample_batched[1].type(torch.cuda.LongTensor)
            n = x.size(0)
            output = self.local_model(x)
            prediction = output.cpu().max(1, keepdim=True)[1]
            loss = criterion(output, y)
            val_loss.update(loss.item())
            val_acc.update(prediction.cpu().eq(y.cpu().view_as(prediction.cpu())).sum().item() / n)

        return val_loss.avg, val_acc.avg

