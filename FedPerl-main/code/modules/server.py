import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from scipy.stats import truncnorm
from data.data_FL import SkinData
from modules.client import *
from modules.similarity_manager import *
from scipy.spatial import KDTree
from torch.utils.data import DataLoader
import copy
import os
import time
device = torch.device('cuda:0')


class Server:
    def __init__(self, args):
        """
        Server init method

        Parameters:
            args: List containing arguments to configure the training

        Returns:
            None
        """

        self.args = args
        self.global_model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=self.args.n_classes)
        num_ftrs = self.global_model._fc.in_features
        self.global_model._fc = nn.Linear(num_ftrs, self.args.n_classes)
        self.global_model = nn.DataParallel(self.global_model)
        self.global_model = self.global_model.cuda()
        self.global_model = self.global_model.to(device)

        self.data = SkinData(self.args.data_path, self.args.clients_path)
        self.train_loaders = []
        self.val_loaders = []
        self.test_loaders = []
        self.client_clss_weights = []
        self.server_loader = None
        self.server_clss_weights = []

        'create a dumy image used to measure the similarity between the clients'
        mu, std, lower, upper = 0, 1, 0, 255
        self.dumyImg = torch.from_numpy(
            (truncnorm((lower - mu) / std, (upper - mu) / std, loc=mu, scale=std).rvs((1, 3, 224, 224))) / 255).type(
            torch.cuda.FloatTensor)

    def build_clients(self):
        """
        build_clients: create client class and peer from a pretrained efficientnet

        Parameters:
            None

        Returns:
            None
        """
        self.client = Client(self.num_classes, self.con, self.lambda_a, self.lambda_i)
        self.peer = self.build_models()
        self.peer.eval()

    def configure(self):
        """
        configure server class from user argument

        Parameters:
            None

        Returns:
            None
        """
        self.steps = self.args.steps
        self.num_clients = self.args.num_clients
        self.num_classes = self.args.num_classes
        self.connected_clients = self.args.connected_clients
        self.batch_size = self.args.batch_size
        self.num_rounds = self.args.num_rounds
        self.curr_lr = self.args.curr_lr
        self.lambda_a = self.args.lambda_a
        self.lambda_i = self.args.lambda_i
        self.con = self.args.con
        self.check_folder = self.args.check_folder
        self.clients_state = self.args.clients_state
        self.models_folder = self.args.models_folder
        self.num_peers = self.args.num_peers
        self.trained_clients = []
        self.vid_to_cid = {}
        self.client_pred = {}
        self.clnts_bst_acc = [0] * self.args.num_clients
        self.client_peers = {
            'client0': [],
            'client1': [],
            'client2': [],
            'client3': [],
            'client4': [],
            'client5': [],
            'client6': [],
            'client7': [],
            'client8': [],
            'client9': []
        }
        num_features = 622
        self.method = self.args.method
        self.is_normalized = self.args.is_normalized
        self.include_acc = self.args.include_acc
        if self.include_acc == 'True':
            num_features += 2
        if 'Perl' in self.method:
            self.similarity_manager = SimilarityManager(self.args.num_clients, num_features)
        self.save_check = self.args.save_check
        self.calculate_val = self.args.calculate_val
        self.is_PA = self.args.is_PA
        self.include_C8 = self.args.include_C8
        self.fed_prox = self.args.fed_prox
        self.name = self.method + '_c8' + str(self.include_C8) + '_avg' + str(self.is_PA) + '_prox' + str(self.fed_prox)


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

    def set_configuration(self, lambda_a=0.5, con=0.6, curr_lr=0.00005, num_rounds=200):
        """
        modify some server configurations

        Parameters:
            lambda_a: unlabeled coefficient of type float
            con: confidence treashold of type float
            curr_lr: learning rate of type float
            num_rounds: number of round of type int

        Returns:
            None
        """
        self.num_rounds = num_rounds
        self.curr_lr = curr_lr
        self.lambda_a = lambda_a
        self.con = con

    def build_client_similarity(self, updates):
        """
        build similarities based on the clients predictions

        Parameters:
            updates: tuple contains clients

        Returns:
            KDTree clients similarities in KDTree format
        """
        for client_id, _, model in updates:
            self.peer.load_state_dict(model)
            self.peer.eval()
            with torch.no_grad():
                self.client_pred[client_id] = np.squeeze(self.peer(self.dumyImg).cpu().numpy())
        self.vid_to_cid = list(self.client_pred.keys())
        self.vectors = list(self.client_pred.values())
        self.tree = KDTree(self.vectors)

    def get_peers(self, client_id, curr_round):
        """
        find top T similar peers (T = args.num_peers)

        Parameters:
            client_id: client id of type int
            curr_round: current round of type int

        Returns:
            weights of top T similar peers when is_PA = False
            or the average weights of top T similar peers when is_PA = True
        """
        weights = []
        if self.include_C8:
            cd2 = -1
        else:
            cd2 = 8
        if client_id in self.trained_clients and self.method != 'Random':
            # Get peers based on the best validation accuracy on the server validation data
            if self.method == 'Acc':
                peers = self.get_best_acc_client(client_id, self.num_peers, cid2=cd2)
                for pid in peers:
                    self.client_peers['client' + str(client_id)].append(pid)
                    w = self.load_client_weights(pid)
                    if self.is_PA:
                        weights.append((pid, 0.5, copy.deepcopy(w)))
                    else:
                        weights.append(copy.deepcopy(w))

            # Get peers based on the FedPerl similarities (i.e. cosine similarities on models weight first two
            # statistical moments)
            if self.method == 'Perl':  # Perl
                sims = self.similarity_manager.get_similar_clients(client_id, self.num_peers, cd2)
                for pid in sims:
                    self.client_peers['client' + str(client_id)].append(pid)
                    w = self.load_client_weights(pid)
                    if self.is_PA:
                        weights.append((pid, 0.5, copy.deepcopy(w)))
                    else:
                        weights.append(copy.deepcopy(w))
            # Get peers based on the FedMatch similarities (KDTree on models predictions)
            elif self.method == 'FedMatch':  # FedMatch
                cout = self.client_pred[client_id]
                sims = self.tree.query(cout, self.num_peers + 1)
                for vid in sims[1]:
                    pid = self.vid_to_cid[vid]
                    if pid == client_id or pid == cd2:
                        continue
                    self.client_peers['client' + str(client_id)].append(pid)
                    w = self.load_client_weights(pid)
                    if self.is_PA:
                        weights.append((pid, 0.5, copy.deepcopy(w)))
                    else:
                        weights.append(copy.deepcopy(w))

        # Random peers
        else:
            if len(self.trained_clients) != 0 and curr_round != 0:
                # Get peers based on the best validation accuracy on the server validation data
                if self.method == 'Acc':
                    peers = self.get_best_acc_client(client_id, self.num_peers, cid2=cd2)
                    for pid in peers:
                        self.client_peers['client' + str(client_id)].append(pid)
                        w = self.load_client_weights(pid)
                        if self.is_PA:
                            weights.append((pid, 0.5, copy.deepcopy(w)))
                        else:
                            weights.append(copy.deepcopy(w))
                else:
                    pids = np.random.choice(self.trained_clients, self.num_peers, replace=False)
                    for pid in pids:
                        if pid == client_id or pid == cd2:
                            continue
                        self.client_peers['client' + str(client_id)].append(pid)
                        w = self.load_client_weights(pid)
                        if self.is_PA:
                            weights.append((pid, 0.5, copy.deepcopy(w)))
                        else:
                            weights.append(copy.deepcopy(w))
            else:
                return None

        if self.is_PA:
            return [self.average(weights)]
        else:
            return weights[:self.num_peers]

    def get_best_acc_client(self, client_id, n, cid2=-1):
        """
       Get indexes for n peers best validation accuracy on the server validation data

       Parameters:
           client_id: client id of type int
           n: number of returned indexes of type int
           cid2: excluded from the search (i.e. when it is already included)

       Returns:
           list of n peers indexes
       """
        arg_sort = reversed(np.argsort(self.clnts_bst_acc))
        clients_idx = []
        cnt = 0
        if self.include_C8:
            c8 = -1
        else:
            c8 = 8
        for i in arg_sort:
            if i != client_id and i != cid2 and i != c8:
                clients_idx.append(i)
                cnt += 1
                if n == 1:
                    return i
                if cnt == n:
                    break
        return clients_idx

    def save_client_weights(self, client_id, weights):
        """
       Saves client weights

       Parameters:
           client_id: client id of type int
           weights: client weights

       Returns:
           None
       """
        torch.save({
            'client': client_id, 'model_state_dict': weights
        }, os.path.join(self.clients_state, 'Client{}.pt'.format(client_id)))

    def load_client_weights(self, client_id):
        """
       Loads client weights

       Parameters:
           client_id: client id of type int

       Returns:
           client
       """
        checkpoint = torch.load(os.path.join(self.clients_state, 'Client{}.pt'.format(client_id)))
        return checkpoint['model_state_dict']

    def aggregate(self, w_locals):
        """
        FedAvg method based on the samples in each client

        Parameters:
            w_locals: List containing <sample_numbers, local  parameters> pairs of all local clients

        Returns:
            averaged_params: aggregated model
        """
        training_num = 0
        for idx in range(len(w_locals)):
            (_, sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num

        (_, sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                _, local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params

    def average(self, w_locals):
        """
        Averages top T peers weights

        Parameters:
            w_locals: List containing <sample_numbers, local  parameters> pairs of all local clients

        Returns:
            averaged_params: top T similar peers averaged model
        """
        training_num = 0
        for idx in range(len(w_locals)):
            (_, sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num

        (_, sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                _, local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params

    def validate(self):
        """
        Validates global model based on the global validation dataset on the server

        Parameters:
            None

        Returns:
            val_loss.avg: validation loss
            val_acc.avg: validation accuracy
        """
        self.global_model.eval()
        val_loss = AverageMeter()
        val_acc = AverageMeter()
        criterion = nn.CrossEntropyLoss(weight=torch.cuda.FloatTensor(self.server_clss_weights))
        for batch_idx, sample_batched in enumerate(self.server_loader):
            X = sample_batched[0].type(torch.cuda.FloatTensor)
            y = sample_batched[1].type(torch.cuda.LongTensor)
            N = X.size(0)
            output = self.global_model(X)
            prediction = output.cpu().max(1, keepdim=True)[1]
            loss = criterion(output, y)
            val_loss.update(loss.item())
            val_acc.update(prediction.cpu().eq(y.cpu().view_as(prediction.cpu())).sum().item() / N)

        return val_loss.avg, val_acc.avg

    def validate_client(self, weights):
        """
        Validates global model based on the global validation dataset on the server

        Parameters:
            weights: client weights

        Returns:
            val_loss.avg: client validation loss
            val_acc.avg: client validation accuracy
        """
        self.peer.load_state_dict(weights)
        self.peer.eval()
        val_loss = AverageMeter()
        val_acc = AverageMeter()
        criterion = nn.CrossEntropyLoss(weight=torch.cuda.FloatTensor(self.server_clss_weights))
        for batch_idx, sample_batched in enumerate(self.server_loader):
            X = sample_batched[0].type(torch.cuda.FloatTensor)
            y = sample_batched[1].type(torch.cuda.LongTensor)
            N = X.size(0)
            output = self.peer(X)
            prediction = output.cpu().max(1, keepdim=True)[1]
            loss = criterion(output, y)
            val_loss.update(loss.item())
            val_acc.update(prediction.cpu().eq(y.cpu().view_as(prediction.cpu())).sum().item() / N)

        return val_loss.avg, val_acc.avg

    def get_features_vec(self, wgts):
        """
       Calculates first two statistical moments of clients' weights

       Parameters:
           wgts: client weights

       Returns:
           vec: vector contains first two statistical moments of clients' weights
       """
        vec = []
        for k in wgts.keys():
            if 'num_batches_tracked' not in k:
                vec.append(torch.mean(wgts[k]))
                vec.append(torch.std(wgts[k]))
        return vec

    def prepare_data(self):
        """
          Loads data to the server and set number of iteration based on smallest dataset (as in FedVC)

          Parameters:
              None

          Returns:
              None
          """
        self.train_loaders = []
        self.train_loaders_u = []
        self.val_loaders = []
        self.client_clss_weights = []

        server_val, self.server_clss_weights = self.data.load_server()
        self.server_loader = DataLoader(server_val, batch_size=self.batch_size, shuffle='False', num_workers=4,
                                        pin_memory=True)

        for clnt in range(self.num_clients):
            train_ds, train_dsu, val_ds, weights = self.data.load_clients_ssl(clnt)
            train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle='True', num_workers=4,
                                      pin_memory=True)
            train_loader_u = DataLoader(train_dsu, batch_size=self.batch_size, shuffle='True', num_workers=4,
                                       pin_memory=True)
            val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle='False', num_workers=4, pin_memory=True)
            self.train_loaders.append(train_loader)
            self.train_loaders_u.append(train_loader_u)
            self.val_loaders.append(val_loader)
            self.client_clss_weights.append(weights)
            if self.steps > len(train_loader_u):
                self.steps = len(train_loader_u)

    def append_client(self, client_id):
        """
          Appends current client to the trained clients list

          Parameters:
              client_id: client id of type int

          Returns:
              None
        """
        if client_id not in self.trained_clients:
            self.trained_clients.append(client_id)

    def run_fed(self):
        """
          Runs federated learning (FedPerl)

          Parameters:
              None

          Returns:
              None
        """

        clnts_bst_acc_ind = [0] * self.num_clients
        best_acc_glob = 0
        clnts_acc_log = np.zeros((self.num_clients, self.num_rounds))
        clnts_acc = [0] * self.num_clients
        start_time = time.time()

        conf = np.zeros((self.num_clients, self.num_rounds))
        ce_ls = np.zeros((self.num_clients, self.num_rounds))
        mse_ls = np.zeros((self.num_clients, self.num_rounds))

        for curr_round in range(self.num_rounds):
            clnts_updates = []
            clnts_tr_loss = []
            clnts_tr_acc = []

            # include or exclude client 8 fron the training
            if self.include_C8:
                clients = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            else:
                clients = [0, 1, 2, 3, 4, 5, 6, 7, 9]

            # select random client for the training in the current round
            connected_ids = np.random.choice(clients, self.connected_clients,
                                             replace=False).tolist()  # pick clients
            print('<----Training---->')
            print('training clients (round:{}, connected:{})'.format(curr_round, connected_ids))

            for client_id in connected_ids:
                # set initial weights from global model
                w_global = self.global_model.state_dict()

                # initials client
                self.client.set(client_id, w_global, self.train_loaders[client_id], None,
                                self.train_loaders_u[client_id], self.client_clss_weights[client_id])

                # get top T similar peers for the client
                peers = None
                if self.method != 'FixMatch':
                    peers = self.get_peers(client_id, curr_round)

                # train client locally
                sz, wgt, lss, acc, con, ce, mse = self.client.train(client_id, curr_round, self.steps, self.curr_lr,
                                                                    peers, self.fed_prox, self.global_model)

                # save client weights
                self.save_client_weights(client_id, copy.deepcopy(wgt))
                clnts_updates.append((client_id, sz, copy.deepcopy(wgt)))
                clnts_tr_loss.append(copy.deepcopy(lss))
                clnts_tr_acc.append(copy.deepcopy(acc))

                conf[client_id, curr_round] = con
                ce_ls[client_id, curr_round] = ce
                mse_ls[client_id, curr_round] = mse

                # generate FedPerl features
                if 'Perl' in self.method:
                    features_vec = self.get_features_vec(copy.deepcopy(wgt))
                    if self.include_acc:
                        lss, acc = self.validate_client(copy.deepcopy(wgt))
                        features_vec.append(acc)
                        features_vec.append(lss)
                    self.similarity_manager.update_plain_features(client_id, features_vec)

            # update trained clients list
            for client_id in connected_ids:
                self.append_client(client_id)

            trlss, count_lss, tracc, count_acc = 0, 0, 0, 0
            for i in range(len(clnts_tr_loss)):
                trlss += clnts_tr_loss[i].sum
                count_lss += clnts_tr_loss[i].count

            for i in range(len(clnts_tr_acc)):
                tracc += clnts_tr_acc[i].sum
                count_acc += clnts_tr_acc[i].count
            print('rnd:{}, AVG: trlss:{}, tracc:{}'.format(curr_round, round(trlss / count_lss, 4),
                                                           round(tracc / count_acc, 4)))
            # models aggregation
            w_global = self.aggregate(clnts_updates)

            # update the global model weights for the next round of the training
            self.global_model.load_state_dict(w_global)

            if self.calculate_val == 'True':
                print('<----Validation---->')
                # global model validation:
                vlss, vacc = self.validate()

                print('rnd:{}, glob_vlss:{}, glob_vacc:{}'.format(curr_round, round(vlss, 4), round(vacc, 4)))
                if vacc >= best_acc_glob:
                    best_acc_glob = vacc
                    best_model_wts = copy.deepcopy(self.global_model.state_dict())
                    torch.save(best_model_wts, os.path.join(self.models_folder, 'Glob{}.pt'.format(self.name)))
                    del best_model_wts

                # save check point every 10 rounds
                if self.save_check == 'True':
                    if (curr_round + 1) % 10 == 0:
                        best_model_wts = copy.deepcopy(self.global_model.state_dict())
                        torch.save(best_model_wts,
                                   os.path.join(self.check_folder, 'Glob{}_Round{}.pt'.format(self.name, curr_round)))
                        del best_model_wts

                # clients validation
                for cid in range(self.num_clients):
                    vlss_clnt, vacc_clnt = self.client.validate(self.global_model.state_dict(), self.val_loaders[cid],
                                                                self.client_clss_weights[cid])
                    clnts_acc_log[cid, curr_round] = vacc_clnt
                    print('rnd:{}, clinet{}: vlss:{}, vacc:{}'.format(curr_round, cid, round(vlss_clnt, 4),
                                                                      round(vacc_clnt, 4)))
                    if vacc_clnt >= clnts_acc[cid]:
                        clnts_acc[cid] = vacc_clnt
                        clnts_bst_acc_ind[cid] = curr_round
                        best_model_wts = copy.deepcopy(self.global_model.state_dict())
                        torch.save(best_model_wts,
                                   os.path.join(self.models_folder, 'Client{}_{}.pt'.format(cid, self.name)))
                        del best_model_wts

            if self.method != 'FixMatch':
                # FedPerl
                if 'Perl' in self.method:
                    if self.is_normalized == 'True':
                        self.similarity_manager.update_similarity_matrix(1)
                    else:
                        self.similarity_manager.update_similarity_matrix(0)

                    # save similarity matrix every 5 rounds
                    if (curr_round + 1) % 5 == 0:
                        np.save(os.path.join(self.check_folder, 'similarity__Round{}_{}'.format(curr_round, self.name)),
                                self.similarity_manager.SimilarityMatrix)
                # FedMatch
                elif 'FedMatch' in self.method:
                    self.build_client_similarity(clnts_updates)

        model_wts = copy.deepcopy(self.global_model.state_dict())
        torch.save(model_wts, os.path.join(self.models_folder, 'lst_Glob{}.pt'.format(self.name)))
        del model_wts

        # save some logs
        np.save('clnts_bstaccind_{}'.format(self.name), clnts_bst_acc_ind)
        np.save('clnts_acclog_{}'.format(self.name), clnts_acc_log)
        np.save('client_peers_{}'.format(self.name), self.client_peers)

        np.save('clnts_conf_{}'.format(self.name), conf)
        np.save('clnts_ce_{}'.format(self.name), ce_ls)
        np.save('clnts_mse_{}'.format(self.name), mse_ls)

        print('all clients done')
        print('server done. ({}s)'.format(time.time() - start_time))
