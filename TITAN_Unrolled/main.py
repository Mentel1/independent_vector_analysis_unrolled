import torch
from architecture import *
from model import *
from data import *
from tools import *
from functions import *
from class_exp import *
from class_algos import *
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
import cProfile
from algorithms.titan_iva_g_reg_torch import *

# Hyperparameters

T = 10000
K = 5
N = 10
Ks = [5,10,20]
Ns = [10,20,30]

lambda_1 = 0.04
lambda_2 = 0.25
rho_bounds_1 = [0.2,0.3]
rho_bounds_2 = [0.6,0.7]
rhos = [rho_bounds_1,rho_bounds_2]
lambdas = [lambda_1,lambda_2]

metaparameters_multiparam = get_metaparameters(rhos,lambdas)
metaparameters_titles = ['Case_A','Case_B','Case_C','Case_D'] #,'Multi_case','Easy_case','Hard_case']


num_epochs = 20
batch_size = 100
num_layers = 100
N_updates_W = 10
loss = IVA_loss()
custom = False
train_size = 1000
eval_size = 200

optimizer1 = torch.optim.SGD
optimizer2 = torch.optim.Adam
normalize_derivatives1 = True
normalize_derivatives2 = False
learning_rate = 0.5
weight_decay = 0

scheduler_mode =  'StepLR' #'ReduceLROnPlateau' #
# parameters for StepLR
patience = 1
factor_lr = 0.8
min_lr = 0.01
# parameters for ReduceLROnPlateau
gamma = 0.9
step_size = 5

archis = ['tied','untied','inertial']
training_modes = ['end-to-end','local',] #'end-to-end','greedy','local'] #'group-of-layers','layer-by-layer']

archi = 'untied'
training_mode = 'end-to-end'

# for training_mode in training_modes:

# for training_mode in training_modes:
    
#     test1 = UTitan(dimensions=(N,T,K),metaparameters=metaparameters_multiparam,metaparameters_title=metaparameters_title,train_size=train_size,eval_size=eval_size,batch_size=batch_size,num_epochs=num_epochs,num_layers=num_layers,optimizer=optimizer1,lr=learning_rate,weight_decay=weight_decay,normalize_derivatives=normalize_derivatives1,scheduler_mode=scheduler_mode,step_size=step_size,gamma=gamma,patience=patience,factor_lr=factor_lr,min_lr=min_lr,N_updates_W=N_updates_W,archi=archi,custom=custom,loss_train=loss,training_mode=training_mode,load=False)
#     test1.train()

    # test2 = UTitan(dimensions=(N,T,K),metaparameters=metaparameters_multiparam,metaparameters_title=metaparameters_title,train_size=train_size,eval_size=eval_size,batch_size=batch_size,num_epochs=num_epochs,num_layers=num_layers,optimizer=optimizer1,lr=learning_rate,weight_decay=weight_decay,normalize_derivatives=normalize_derivatives2,scheduler_mode=scheduler_mode,step_size=step_size,gamma=gamma,patience=patience,factor_lr=factor_lr,min_lr=min_lr,N_updates_W=N_updates_W,archi=archi,custom=custom,loss_train=loss,training_mode=training_mode,load=False)
    # test2.train()

    # test3 = UTitan(dimensions=(N,T,K),metaparameters=metaparameters_multiparam,metaparameters_title=metaparameters_title,train_size=train_size,eval_size=eval_size,batch_size=batch_size,num_epochs=num_epochs,num_layers=num_layers,optimizer=optimizer2,lr=learning_rate,weight_decay=weight_decay,normalize_derivatives=normalize_derivatives2,scheduler_mode=scheduler_mode,step_size=step_size,gamma=gamma,patience=patience,factor_lr=factor_lr,min_lr=min_lr,N_updates_W=N_updates_W,archi=archi,custom=custom,loss_train=loss,training_mode=training_mode,load=False)
    # test3.train()
    
    
    
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


#==================================================================================

label_size = 20
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size
plt.rcParams['text.usetex'] = True

# Function to generate dataparameters for the multiparameter experiment
def get_dataparameters(rhos,lambdas):
    dataparameters_multiparam = []
    for rho_bounds in rhos:
        for lambda_ in lambdas:
            dataparameters_multiparam.append((rho_bounds,lambda_))
    return dataparameters_multiparam


def create_algos_titanIVAG(varying_param, values, color_bounds=[(0.2,1,0.2),(0.2,0.2,1)],base_params={},basename=''):
    algos = []
    nval = len(values)
    for i, value in enumerate(values):
        params = base_params.copy()
        params[varying_param] = value
        t = i / (nval - 1)
        params['color'] = tuple((1 - t) * c0 + t * c1 for c0, c1 in zip(color_bounds[0], color_bounds[1]))
        params['name'] = basename + '_' + varying_param + '=' + str(value)      
        algos.append(TitanIvaG(**params))
    return algos


lambda_1 = 0.04
lambda_2 = 0.25
rho_bounds_1 = [0.2,0.3]
rho_bounds_2 = [0.6,0.7]
rhos = [rho_bounds_1,rho_bounds_2]
lambdas = [lambda_1,lambda_2]
dataparameters_multiparam = get_dataparameters(rhos,lambdas)
dataparameters_titles_multiparam = ['Case_A','Case_B','Case_C','Case_D']
dataparameters_base = get_dataparameters([[0.4,0.6]],[0.1])
dataparameters_base_titles = ['Base_Case']
dataparameters = [{'num_samples':[10000,5000,1000,500,200,150,120,100]}]

Ks = [5,10,20]
Ns = [10,20,30] 
common_parameters = [Ks,Ns]


algo_titan = TitanIvaG([1,0,0],nu=0,max_iter_int_W=15,gamma_c=1.99)
algos = [algo_titan]
# algos = create_algos_titanIVAG(varying_param='epsilon',values=[1e-3,1e-2,1e-1],base_params={'nu':0,'max_iter_int_W':15,'gamma_c':1.99},basename='palm')
exp = ComparisonExperimentIvaG('Robustness_noise',dataparameters,dataparameters_base_titles,[[20],[20]],algos=algos,N_exp=10,legend_fontsize=20,title_fontsize=30,updates=True)
exp.compute_multi_runs()