import torch
from .architecture import *
from .model import *
from .data import *
from .tools import *
from .functions import *
from class_exp import *
from class_algos import *
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
import cProfile
from algorithms.titan_iva_g_reg_torch import *

# Hyperparameters

T = 10000
K = 20
N = 30
Ks = [5,10,20]
Ns = [10,20,30]

lambda_1 = 0.04
lambda_2 = 0.25
rho_bounds_1 = [0.2,0.3]
rho_bounds_2 = [0.6,0.7]
rhos = [rho_bounds_1] #,rho_bounds_2]
lambdas = [lambda_1] #,lambda_2]

metaparameters_multiparam = get_metaparameters(rhos,lambdas)
metaparameters_title = 'Case_A' #,'Case_B','Case_C','Case_D'] #,'Multi_case','Easy_case','Hard_case'


num_epochs = 20
batch_size = 100
num_layers = 500
N_updates_W = 10
loss = IVA_loss()
custom = False
train_size = 1000
eval_size = 200

normalize_derivatives1 = True
normalize_derivatives2 = False
learning_rate = 1
weight_decay = 0

scheduler_mode =  'StepLR' #'ReduceLROnPlateau' #
# parameters for StepLR
patience = 1
factor_lr = 0.8
min_lr = 0.01
# parameters for ReduceLROnPlateau
gamma = 0.8
step_size = 5

optimizers = [torch.optim.SGD,torch.optim.Adam]
gradient_processings = ['raw','clip','normalize']
archis = ['tied','untied','inertial-tied','inertial-untied']
training_modes = ['local','end-to-end'] #'end-to-end','greedy','local'] #'group-of-layers','layer-by-layer']

archi = 'untied'
training_mode = 'end-to-end'

for training_mode in training_modes:
    for optimizer,gradient_processing in [(torch.optim.SGD,'normalize'),(torch.optim.Adam,'raw')]:
        test = UTitan(dimensions=(N,T,K),metaparameters=metaparameters_multiparam,metaparameters_title=metaparameters_title,train_size=train_size,eval_size=eval_size,batch_size=batch_size,num_epochs=num_epochs,num_layers=num_layers,optimizer=optimizer,lr=learning_rate,weight_decay=weight_decay,gradient_processing=gradient_processing,scheduler_mode=scheduler_mode,step_size=step_size,gamma=gamma,patience=patience,factor_lr=factor_lr,min_lr=min_lr,N_updates_W=N_updates_W,archi=archi,custom=custom,loss_train=loss,training_mode=training_mode,load=False)
        test.train()
    
    
#================ DATASETS CREATION =================    
    
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dtype=torch.float32
# for i,metaparameters in enumerate(metaparameters_multiparam):
#     metaparameters_title = metaparameters_titles[i]
#     for K in Ks:
#         for N in Ns:
#             dimensions = (N,T,K)
#             for file,size in [('train',1000),('eval',200),('test',20)]:
#                 datasets_path = f'Result_data/datasets/{metaparameters_title}/N_{N}_K_{K}'
#                 os.makedirs(datasets_path,exist_ok=True)
#                 data_path = os.path.join(datasets_path,file)
#                 dataset = IVAGDataset(data_path,dimensions,[metaparameters],size,device,dtype)
