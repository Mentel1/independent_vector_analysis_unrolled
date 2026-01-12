import torch
from architecture import *
from model import *
from data import *
from tools import *
from functions import *

# Hyperparameters

T = 10000
K = 20
N = 20

lambda_1 = 0.04
lambda_2 = 0.25
rho_bounds_1 = [0.2,0.3]
rho_bounds_2 = [0.6,0.7]
rhos = [rho_bounds_1] #,rho_bounds_2]
lambdas = [lambda_1] #,lambda_2]

metaparameters_multiparam = get_metaparameters(rhos,lambdas)
metaparameters_title = 'Case_A' #,'Case_B','Case_C','Case_D','Multi_case','Easy_case','Hard_case']

learning_rate = 0.1
min_lr = 0.02
num_epochs = 20
batch_size = 100
num_layers = 300
loss = IVA_loss()
custom=False
train_size = 1000
test_size = 200
weight_decay_begin = 1e-2
weight_decay_end = 1e-6
factor_lr=0.5
patience = 5
N_updates_W = 5


# test_end_to_end = UTitan(dimensions=(N,T,K),metaparameters=metaparameters_multiparam,metaparameters_title=metaparameters_title,train_size=train_size,test_size=test_size,batch_size=batch_size,num_epochs=num_epochs,num_layers=num_layers,lr=learning_rate,patience=patience,weight_decay_begin=weight_decay_begin,weight_decay_end=weight_decay_end,factor_lr=factor_lr,min_lr=min_lr,N_updates_W=N_updates_W,archi='untied',custom=custom,loss_train=loss,training_mode='end_to_end',load=False)
# test_end_to_end.train()
# trajectory = test_end_to_end.compute_trajectory()
# print(trajectory)

test_greedy = UTitan(dimensions=(N,T,K),metaparameters=metaparameters_multiparam,metaparameters_title=metaparameters_title,train_size=train_size,test_size=test_size,batch_size=batch_size,num_epochs=num_epochs,num_layers=num_layers,lr=learning_rate,patience=patience,weight_decay_begin=weight_decay_begin,weight_decay_end=weight_decay_end,factor_lr=factor_lr,min_lr=min_lr,N_updates_W=N_updates_W,archi='untied',custom=custom,loss_train=loss,training_mode='greedy',load=False)
test_greedy.train()
trajectory = test_greedy.compute_trajectory()
print(trajectory)

# test_group_layers = UTitan(dimensions=(N,T,K),metaparameters=metaparameters_multiparam,metaparameters_title=metaparameters_title,train_size=500,test_size=100,batch_size=batch_size,num_epochs=num_epochs,num_layers=num_layers,lr=learning_rate,N_updates_W=15,archi='untied',custom=False,loss_train=ISI_loss(),training_mode='one_by_one',load=False)
# test_group_layers.train()
# trajectory = test_group_layers.compute_trajectory()
# print(trajectory)


