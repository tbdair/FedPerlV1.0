import argparse
import copy
import os
import time

import numpy as np
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from torch import optim
from torch.optim import *
from torch.utils.data import DataLoader

from data.data_FL import SkinData
from utils.AverageMeter import *

SEED = 1
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
device = torch.device('cuda:0')

parser = argparse.ArgumentParser(description='FedPerl')

parser.add_argument('--data_path', default='', type=str, metavar='PATH',
                    help='path to the dataset(validation or testing)')
parser.add_argument('--clients_path', default='', type=str, metavar='PATH',
                    help='path to clients')
parser.add_argument('--check_folder', default='', type=str, metavar='PATH',
                    help='path to the check points model')
parser.add_argument('--num_rounds', type=int, default=200)
parser.add_argument('--steps', type=int, default=100, help='number of iteration')
parser.add_argument('--curr_lr', type=int, default=0.00005)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_classes', type=int, default=8)
parser.add_argument('--weight_decay', type=int, default=0.002)
parser.add_argument('--img_channel', type=int, default=1)
parser.add_argument('--img_width', type=int, default=256)
parser.add_argument('--img_height', type=int, default=256)
parser.add_argument('--alpha', type=int, default=1)
parser.add_argument('--con', type=int, default=0.9, help='confidence treashold')
parser.add_argument('--lambda_a', type=int, default=0.5, help='unlabeled coefficient')
parser.add_argument('--save_check', default='True', help='save check points')
parser.add_argument('--calculate_val', default='True',
                    help='calculate validation accuracy for clients after each round')
parser.add_argument('--settting', default='SSL', help='Lower, Upper, SSL')


class Trainer:
    def __init__(self, args):
        """
        Trainer init method

        Parameters:
            args: List containing arguments to configure the training

        Returns:
            None
        """
        self.args = args
        self.data = SkinData(self.args.data_path, self.args.clients_path)

    def configure(self):
        """
        configure trainer class from user argument

        Parameters:
            None

        Returns:
            None
        """
        self.num_classes = self.args.num_classes
        self.batch_size = self.args.batch_size
        self.num_rounds = self.args.num_rounds
        self.curr_lr = self.args.curr_lr
        self.lambda_a = self.args.lambda_a
        self.con = self.args.con
        self.check_folder = self.args.check_folder  # '/home/tariq/code/UsedData/local_models/check_log/'
        self.models_folder = self.args.models_folder  # '/home/tariq/code/UsedData/Models_500/local_models/'
        self.save_check = self.args.save_check
        self.calculate_val = self.args.calculate_val
        self.client_id = self.args.client_id
        self.settting = self.args.settting

    def set_client(self, client_id):
        """
        set client data

        Parameters:
        client_id:

        Returns:
        None
        """
        self.client_id = client_id
        self.name = 'Client{}_{}_{}_Con{}_LR{}_Lu{}'.format(self.client_id, self.settting, self.num_rounds, self.con,
                                                            self.curr_lr, self.lambda_a)

    def build_model(self):
        """
        create a client with its efficientnet pretrained model

        Parameters:
        None

        Returns:
        None
        """
        self.model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=self.num_classes)
        num_ftrs = self.model._fc.in_features
        self.model._fc = nn.Linear(num_ftrs, self.num_classes)
        self.model = nn.DataParallel(self.model)
        self.model = self.model.cuda()
        self.model = self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.curr_lr)
        self.criterion_l = nn.CrossEntropyLoss(weight=torch.cuda.FloatTensor(self.clss_weights))
        self.criterion_u = nn.CrossEntropyLoss()

    def delete_model(self):
        """
        delete the model after finish training

        Parameters:
        None

        Returns:
        None
        """
        del self.model
        del self.optimizer
        del self.criterion_l
        del self.criterion_u
        if self.settting == 'SSL':
            del self.train_loader_u
        del self.train_loader
        del self.val_loader
        del self.clss_weights

    def un_supervised_loss(self, x, x_aug):
        """
        unsupervised loss

        Parameters:
        x: input batch
        x_aug: augment input batch

        Returns:
        loss
        """
        loss_u = 0
        with torch.no_grad():
            y_pred = self.model(x)
        prob = torch.softmax(y_pred.detach(), dim=-1)
        mx_prob, mx_ind = torch.max(prob, axis=1)
        mx_prob = (mx_prob >= self.con) * mx_prob
        conf = [i for i in range(len(mx_ind)) if mx_prob[i] != 0]
        if len(conf) > 0:
            x_conf = x_aug[conf]
            y_hard = self.model(x_conf)
            pseudo_label = mx_ind[conf]
            loss_u += self.criterion_u(y_hard, pseudo_label) * self.lambda_a
        return loss_u, len(conf)

    def train_fully_supervised(self, curr_round):
        """
        train_fully_supervised

        Parameters:
        curr_round

        Returns:
        None
        """
        train_loss = AverageMeter()
        train_acc = AverageMeter()
        train_loader_l_iter = enumerate(self.train_loader)

        for i in range(self.steps):
            # Check if the label loader has a batch available
            try:
                _, sample_batched = next(train_loader_l_iter)
            except:
                # Curr loader doesn't have data, then reload data
                del train_loader_l_iter
                train_loader_l_iter = enumerate(self.train_loader)
                _, sample_batched = next(train_loader_l_iter)

            x = sample_batched[0].type(torch.cuda.FloatTensor)
            y = sample_batched[1].type(torch.cuda.LongTensor)
            n = x.size(0)
            # y = torch.nn.functional.one_hot(y)              

            with torch.set_grad_enabled(True):
                output = self.model(x)
                loss = self.criterion_l(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                prediction = output.cpu().max(1, keepdim=True)[1]
                train_acc.update(prediction.cpu().eq(y.cpu().view_as(prediction.cpu())).sum().item() / n)
                train_loss.update(loss.item())

        print('Training  : Epoch:{}, clnt{}: trlss:{}, tracc:{}'.format(curr_round, self.client_id,
                                                                        round(train_loss.avg, 4),
                                                                        round(train_acc.avg, 4)))

    def train_ssl(self, curr_round):
        """
        train_ssl

        Parameters:
        curr_round

        Returns:
        None
        """

        train_loss = AverageMeter()
        train_acc = AverageMeter()

        train_loader_l_iter = enumerate(self.train_loader)
        train_loader_u_iter = enumerate(self.train_loader_u)

        # for batch_idx, sample_batched in enumerate(train_loader):
        for i in range(self.steps):
            # Check if the label loader has a batch available
            try:
                _, sample_batched = next(train_loader_l_iter)
            except:
                # Curr loader doesn't have data, then reload data
                del train_loader_l_iter
                train_loader_l_iter = enumerate(self.train_loader)
                _, sample_batched = next(train_loader_l_iter)

            try:
                _, sample_batched_u = next(train_loader_u_iter)
            except:
                del train_loader_u_iter
                train_loader_u_iter = enumerate(self.train_loader_u)
                _, sample_batched_u = next(train_loader_u_iter)

            x = sample_batched[0].type(torch.cuda.FloatTensor)
            y = sample_batched[1].type(torch.cuda.LongTensor)
            x_u = sample_batched_u[0].type(torch.cuda.FloatTensor)
            x_u_aug = sample_batched_u[2].type(torch.cuda.FloatTensor)
            n = x.size(0)
            # y = torch.nn.functional.one_hot(y)              

            with torch.set_grad_enabled(True):
                output = self.model(x)
                loss = self.criterion_l(output, y)
                loss_u, conf = self.un_supervised_loss(x_u, x_u_aug)
                loss += loss_u
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                prediction = output.cpu().max(1, keepdim=True)[1]
                train_acc.update(prediction.cpu().eq(y.cpu().view_as(prediction.cpu())).sum().item() / n)
                train_loss.update(loss.item())

        print('Training  : Epoch:{}, clnt{}: trlss:{}, tracc:{}'.format(curr_round, self.client_id,
                                                                        round(train_loss.avg, 4),
                                                                        round(train_acc.avg, 4)))

    def validate(self):
        """
        validate model

        Parameters:
        None

        Returns:
        validate loss
        """
        self.model.eval()
        val_loss = AverageMeter()
        val_acc = AverageMeter()
        for batch_idx, sample_batched in enumerate(self.val_loader):
            x = sample_batched[0].type(torch.cuda.FloatTensor)
            y = sample_batched[1].type(torch.cuda.LongTensor)
            n = x.size(0)
            output = self.model(x)
            prediction = output.cpu().max(1, keepdim=True)[1]
            loss = self.criterion_l(output, y)
            val_loss.update(loss.item())
            val_acc.update(prediction.cpu().eq(y.cpu().view_as(prediction.cpu())).sum().item() / n)

        return val_loss.avg, val_acc.avg

    def prepare_data(self):
        """
        load data to the model

        Parameters:
        None

        Returns:
        None
        """
        if self.settting == 'SSL':
            train_ds, train_dsu, val_ds, self.clss_weights = self.data.load_clients_ssl(self.client_id)
            self.train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle='True', num_workers=4,
                                           pin_memory=True)
            self.train_loader_u = DataLoader(train_dsu, batch_size=self.batch_size, shuffle='True', num_workers=4,
                                             pin_memory=True)
            self.val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle='False', num_workers=4,
                                         pin_memory=True)
            self.steps = len(self.train_loader_u)
        elif self.settting == 'Lower':
            train_ds, val_ds, self.clss_weights = self.data.load_clients_lower(self.client_id)
            self.train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle='True', num_workers=4,
                                           pin_memory=True)
            self.val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle='False', num_workers=4,
                                         pin_memory=True)
            self.steps = len(self.train_loader)
        elif self.settting == 'Upper':
            train_ds, val_ds, self.clss_weights = self.data.load_clients_upper(self.client_id)
            self.train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle='True', num_workers=4,
                                           pin_memory=True)
            self.val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle='False', num_workers=4,
                                         pin_memory=True)
            self.steps = len(self.train_loader)

    def run(self):
        """
        run training

        Parameters:
        None

        Returns:
        None
        """
        best_acc = 0
        start_time = time.time()

        for curr_round in range(self.num_rounds):
            # print('<----Training---->')  
            if self.settting == 'SSL':
                self.train_ssl(curr_round)
            else:
                self.train_fully_supervised(curr_round)
            # print('<----Validation---->')  
            vlss, vacc = self.validate()
            print('Validation: Epoch:{}, clnt{}: vlss:{}, vacc:{}'.format(curr_round, self.client_id, round(vlss, 4),
                                                                          round(vacc, 4)))
            if vacc >= best_acc:
                best_acc = vacc
                best_model_wts = copy.deepcopy(self.model.state_dict())
                torch.save(best_model_wts, os.path.join(self.models_folder, '{}.pt'.format(self.name)))
                del best_model_wts

            if self.save_check:
                if (curr_round + 1) % 10 == 0:
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                    torch.save(best_model_wts,
                               os.path.join(self.check_folder, '{}_Epoch{}.pt'.format(self.name, curr_round)))
                    del best_model_wts

        model_wts = copy.deepcopy(self.model.state_dict())
        torch.save(model_wts, os.path.join(self.models_folder, 'lst_{}.pt'.format(self.name)))
        del model_wts

        print('Client{} done'.format(self.client_id))
        print('Total time. ({}s)'.format(time.time() - start_time))


if __name__ == '__main__':
    args = parser.parse_args()
    trainer = Trainer(args)
    trainer.configure()
    for clnt in range(10):
        print("************** Client{}**********".format(clnt))
        trainer.set_client(clnt)
        trainer.prepare_data()
        trainer.build_model()
        trainer.run()
        trainer.delete_model()
