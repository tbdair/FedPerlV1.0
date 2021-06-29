import argparse
import torch
from torch.optim import *
from modules.server import *

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
parser.add_argument('--clients_state', default='', type=str, metavar='PATH',
                    help='path to save the clients state temporarly')
parser.add_argument('--models_folder', default='', type=str, metavar='PATH',
                    help='path to save the trained models')
parser.add_argument('--model_name', default='FedPerl', type=str)
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
parser.add_argument('--con', type=int, default=0.6, help='confidence treashold')
parser.add_argument('--lambda_a', type=int, default=0.5, help='unlabeled coefficient')
parser.add_argument('--lambda_i', type=int, default=0.01, help='consistency coefficient')
parser.add_argument('--num_clients', type=int, default=10, help='num of clients')
parser.add_argument('--connected_clients', type=int, default=10, help='connected clients')
parser.add_argument('--num_peers', type=int, default=2, help='number of peers used in peer learning')
parser.add_argument('--method', default='Perl', type=str,
                    help='current options Perl, Random, FedMatch, FixMatch')
parser.add_argument('--is_normalized', default='True', help='normalize the features on the similairty matrix')
parser.add_argument('--include_acc', default='True',
                    help='include clients accuarcy in the similarity calculation for FedPerl')
parser.add_argument('--save_check', default='True', help='save check points')
parser.add_argument('--calculate_val', default='True',
                    help='calculate validation accuracy for clients after each round')
parser.add_argument('--is_PA', default='True', help='apply peer anonymization')
parser.add_argument('--include_C8', default='True', help='include client 8 in the training')
parser.add_argument('--fed_prox', default='False', help='apply fedprox')


if __name__ == '__main__':
    args = parser.parse_args()
    server = Server(args)
    server.configure()
    server.build_clients()
    server.prepare_data()
    server.run_fed()
