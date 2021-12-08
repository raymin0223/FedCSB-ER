import argparse

__all__ = ['parse_args']
        
        
class StoreDictKeyPair(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values.split(","):
            k,v = kv.split("=")
            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)
        

def parse_args():
    parser = argparse.ArgumentParser()

    ############ experimental settings ###############
    parser.add_argument('--device', 
                    help='gpu device;', 
                    type=str,
                    default='cuda:0')
    parser.add_argument('--seed',
                    help='seed for experiment',
                    type=int,
                    default=0)
    parser.add_argument('--exp-name',
                    help='name for experiment',
                    type=str,
                    default='')
    
    #################### dataset ####################
    # basic settings
    parser.add_argument('--dataset',
                    help='name of dataset;',
                    type=str,
                    required=True,
                    choices=['dirichlet_mnist', 'dirichlet_cifar10', 'dirichlet_cifar100', 'dirichlet_fashion_mnist', 'femnist', 'federated_cifar100', 'synthetic', 'landmark_g23k', 'landmark_g160k'])
    parser.add_argument('--data-dir', 
                    help='dir for dataset;',
                    type=str,
                    default='./data/benchmarks')
    parser.add_argument('--batch-size',
                    help='batch size of local data on each client;',
                    type=int,
                    default=128)
    parser.add_argument('--pin-memory', 
                    help='argument of pin memory on DataLoader;',
                    action='store_true')
    parser.add_argument('--num-workers', 
                    help='argument of num workers on DataLoader;',
                    type=int,
                    default=4)
    parser.add_argument('--num-clients', 
                    help='number of clients;',
                    type=int,
                    default=20)
    parser.add_argument('--valid-size', 
                    help='number of validation dataset;',
                    type=int,
                    default=0)
    parser.add_argument('--niid-split', 
                    help='split number when making non-iid dataset;',
                    type=int,
                    default=10)
    
    # non-iidness
    parser.add_argument('--non-iid', 
                    help='dirichlet parameter to control non-iidness of dataset;',
                    type=float,
                    default=100.0)
    
    ##################### model #####################
    parser.add_argument('--model',
                    help='name of model;',
                    type=str,
                    required=True,
                    choices=['fc', 'lenet', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'resnet8'])
    # example : --model-kwargs num_classes=10
    parser.add_argument('--model-kwargs',
                        dest='model_kwargs',
                        action=StoreDictKeyPair,
                        metavar="KEY1=VAL1,KEY2=VAL2...")
    
    ################## server_opt ###################
    parser.add_argument('--algorithm',
                    help='which algorithm to select clients;',
                    type=str,
                    default='fedavg',
                    choices=['centralized', 'fedavg', 'fedprox', 'fedavg_pdp', 'fedcsb'])
    parser.add_argument('--fedcsb-crit', 
                    help='which criterion to use on fedcsb algorithm;',
                    default='label',
                    choices=['label', 'softmax'])
    parser.add_argument('--num-rounds',
                    help='number of rounds to simulate;',
                    type=int,
                    default=100)
    parser.add_argument('--clients-per-round',
                    help='number of clients trained per round;',
                    type=int,
                    default=20)
    parser.add_argument('--global-momentum',
                    help='whether to use momentum in server optimizers;',
                    type=float,
                    default=0.0)
    
    ################## client_opt ###################
    parser.add_argument('--num-epochs',
                    help='number of rounds to local update;',
                    type=int,
                    default=5)
    # criterion
    parser.add_argument('--local-criterion',
                    help='criterion to use in local training;',
                    type=str,
                    default='ce')
    
    # optimizer
    parser.add_argument('--local-optimizer',
                    help='optimizer to use in local training;',
                    type=str,
                    default='sgd')
    parser.add_argument('--local-lr',
                    help='learning rate for local optimizers;',
                    type=float,
                    default=0.1)
    parser.add_argument('--wd',
                    help='weight decay lambda hyperparameter in local optimizers;',
                    type=float,
                    default=1e-4)
    parser.add_argument('--mu',
                    help='fedprox mu hyperparameter in local optimizers;',
                    type=float,
                    default=0.0)
    parser.add_argument('--nesterov',
                    help='nesterov switch for local momentum',
                    action='store_false')
    parser.add_argument('--local-momentum',
                    help='whether to use momentum in local optimizers;',
                    type=float,
                    default=0.9)
    parser.add_argument('--scheduler',
                    help= 'choose the scheduler type',
                    type= str,
                    default='multistep',
                    choices=['multistep', 'cosine'])
    parser.add_argument('--lr-decay',
                    help='learning rate decay',
                    type=float,
                    default=0.1)
    parser.add_argument('--milestones',
                    help='milestones for step scheduler',
                    type=str,
                    default='50,75')
    
    args = parser.parse_args()
    args = _align_args(args)
    
    return args


def _align_args(args):
    # number of clients
    assert args.num_clients >= args.clients_per_round
    
    # for multi step lr scheduler
    iterations = args.milestones.split(',')
    args.milestones = list([])
    for it in iterations:
        args.milestones.append(int(it))
        
    # align other argument according to args.dataset
    if args.dataset == 'synthetic':
        assert args.model == 'fc'
    
    # align other argument according to args.algorithm
    if args.algorithm == 'centralized':
        args.num_clients = 1
        args.clients_per_round = 1
        args.num_epochs = 1
        args.full_part = True
    
    elif args.algorithm == 'fedavg':
        args.mu = 0.0
        args.full_part = True
        
    elif args.algorithm == 'fedavg_pdp':
        args.mu = 0.0
        args.full_part = False
        
    elif args.algorithm == 'fedcsb':
        args.mu = 0.0
        args.full_part = False
        
    elif args.algorithm == 'fedprox':
        assert args.mu != 0.0
        if not args.exp_name:
            args.exp_name = 'mu%s' % args.mu
        else:
            args.exp_name += '_mu%s' % args.mu
        args.full_part = False
    else:
        pass
        
    return args