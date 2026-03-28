import torch
from .architecture import *
from .model import *
from .data import *
from .tools import *
from .functions import *
from experiment import *
from algorithms import *
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
import cProfile
from Algorithms.titan_iva_g_reg_torch import *
import argparse

# Hyperparameters

lambda_1 = 0.04
lambda_2 = 0.25
rho_bounds_1 = [0.2,0.3]
rho_bounds_2 = [0.6,0.7]
rhos = [rho_bounds_1,rho_bounds_2]
lambdas = [lambda_1,lambda_2]

dataparameters_multiparam = get_dataparameters(rhos,lambdas)
dataparameters_titles = ['Case_A','Case_B','Case_C','Case_D'] #,'Multi_case','Easy_case','Hard_case'

num_epochs = 20
batch_size = 100
num_layers = 500
N_updates_W = 10
loss = IVA_loss()
custom = False
train_size = 1000
eval_size = 200

learning_rate = 0.1
weight_decay = 0

scheduler_mode =  'StepLR' #'ReduceLROnPlateau' #
# parameters for StepLR
patience = 1
factor_lr = 0.8
min_lr = 0.01
# parameters for ReduceLROnPlateau
gamma = 0.8

OPTIMIZERS = {'SGD': (torch.optim.SGD,'normalize'),'Adam': (torch.optim.Adam,'raw')}

parser = argparse.ArgumentParser()
parser.add_argument('--K', type=int, default=20)
parser.add_argument('--N', type=int, default=30)
parser.add_argument('--V', type=int, default=10000)
parser.add_argument('--num_case',type=int, default=0)
parser.add_argument('--opt', type=str, default='SGD', choices=['SGD', 'Adam'])
parser.add_argument('--training_mode', type=str, default='local', choices=['local', 'end-to-end'])
parser.add_argument('--archi', type=str, default='untied', choices=['tied', 'untied', 'inertial-tied', 'inertial-untied'])
parser.add_argument('--num_layers', type=int, default=500)
parser.add_argument('--step_size', type=int, default=5)
args = parser.parse_args()


N = args.N
K = args.K
V = args.V
num_case = args.num_case
opt = args.opt
training_mode = args.training_mode
archi = args.archi
step_size = args.step_size
num_layers = args.num_layers


# for opt in ['SGD','Adam']:
#     for training_mode in ['local','end-to-end']:

optimizer, gradient_processing = OPTIMIZERS[opt]
# training_mode = 'local'
# for archi in ['tied', 'untied', 'inertial-tied', 'inertial-untied']:
test = UTitan(model_name='UTitan'+ str(step_size),dimensions=(N,V,K),dataparameters=[dataparameters_multiparam[num_case]],dataparameters_title=dataparameters_titles[num_case],train_size=train_size,eval_size=eval_size,batch_size=batch_size,num_epochs=num_epochs,num_layers=num_layers,optimizer=optimizer,lr=learning_rate,weight_decay=weight_decay,gradient_processing=gradient_processing,scheduler_mode=scheduler_mode,step_size=step_size,gamma=gamma,patience=patience,factor_lr=factor_lr,min_lr=min_lr,N_updates_W=N_updates_W,archi=archi,custom=custom,loss_train=loss,training_mode=training_mode,load=False)
test.train()
    # L = test.select_num_layers()
    # print(f'{test.model_name} can be shortened to {L} layers')
    # test.shorten_model()
    
    
#================ DATASETS CREATION =================    
    
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dtype=torch.float32
# for i,dataparameters in enumerate(dataparameters_multiparam):
#     dataparameters_title = dataparameters_titles[i]
#     for K in Ks:
#         for N in Ns:
#             dimensions = (N,V,K)
#             for file,size in [('train',1000),('eval',200),('test',100)]:
#                 datasets_path = f'Result_data/datasets/{dataparameters_title}/N_{N}_K_{K}'
#                 os.makedirs(datasets_path,exist_ok=True)
#                 data_path = os.path.join(datasets_path,file)
#                 dataset = IVAGDataset(data_path,dimensions,[dataparameters],size,device,dtype)
