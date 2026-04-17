import torch
from .architecture import *
from .trainer import *
from .data import *
from .tools import *
from .functions import *
from .datasets import *
from experiment import *
from algorithms import *
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
import cProfile
from Algorithms.titan_iva_g_reg_torch import *
import argparse

# Base parameters
dataparams_titles = ['Case_A','Case_B','Case_C','Case_D'] 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype=torch.float32
V = 10000
num_epochs = 20
batch_size = 100
N_updates_W = 10
N_updates_C = 1
epsilon = 1e-12
loss_train = IVA_loss()
custom = False
OPTIMIZERS = {'SGD': (torch.optim.SGD,'normalize'),'Adam': (torch.optim.Adam,'raw')}

parser = argparse.ArgumentParser()
parser.add_argument('--num_case',type=int, default=0)
parser.add_argument('--N', type=int, default=30)
parser.add_argument('--K', type=int, default=20)
parser.add_argument('--num_layers', type=int, default=500)
parser.add_argument('--archi', type=str, default='untied', choices=['tied','untied','inertial-tied','inertial-untied'])
parser.add_argument('--opt', type=str, default='SGD', choices=['SGD','Adam'])
parser.add_argument('--training_mode', type=str, default='local', choices=['local','end-to-end'])
parser.add_argument('--step_size', type=int, default=5)
args = parser.parse_args()


data_case = dataparams_titles[args.num_case]
N = args.N
K = args.K
num_layers = args.num_layers
archi = args.archi
opt = args.opt
training_mode = args.training_mode
step_size = args.step_size


### Build datasets
training_set = IVAGDataset('train',load=True,dimensions=(N,V,K),data_case=data_case,size=1000,device=device,dtype=dtype)
eval_set = IVAGDataset('eval',load=True,dimensions=(N,V,K),data_case=data_case,size=200,device=device,dtype=dtype)

### Build model, optimizer and scheduler
model = UTitanIVAGModel(N,K,num_layers,N_updates_W=N_updates_W,N_updates_C=N_updates_C,epsilon=epsilon,archi=archi,custom=False).to(device)

optimizer_cls, gradient_processing = OPTIMIZERS[opt]
# optimizer_args_SGD = {'lr':0.1,'weight_decay':0}
# scheduler_args_ReduceLROnPlateau = {'mode':min,'patience':1,'factor_lr':0.8,'min_lr':0.01,'threshold':1e-6}
optimizer_cfg = {'class':optimizer_cls,'grad_proc':gradient_processing,'args':{'lr':0.1,'weight_decay':0}}
scheduler_cfg = {'class':torch.optim.lr_scheduler.StepLR,'args':{'gamma':0.8,'step_size':step_size}}

# Build paths

data_path = f'{data_case}/N_{N}_K_{K}'
training_path = f'{training_mode}_{opt}_{step_size}_{gradient_processing}'
model_path = f'Result_data/models/{data_path}/{training_path}/UTitan_{archi}_{num_layers}'



# test = UTitan(model_name='UTitan'+ str(step_size),dimensions=(N,V,K),dataparameters=[dataparameters_multiparam[num_case]],dataparameters_title=dataparameters_titles[num_case],train_size=train_size,eval_size=eval_size,batch_size=batch_size,num_epochs=num_epochs,num_layers=num_layers,optimizer=optimizer,lr=learning_rate,weight_decay=weight_decay,gradient_processing=gradient_processing,scheduler_mode=scheduler_mode,step_size=step_size,gamma=gamma,patience=patience,factor_lr=factor_lr,min_lr=min_lr,N_updates_W=N_updates_W,archi=archi,custom=custom,loss_train=loss,training_mode=training_mode,load=False)
trainer = UTitanTrainer(model,model_path,training_set,eval_set,training_mode=training_mode,optimizer_cfg=optimizer_cfg,scheduler_cfg=scheduler_cfg,loss_train=loss_train,num_epochs=20,batch_size=100)
trainer.train()
    
    
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
