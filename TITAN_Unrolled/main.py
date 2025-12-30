import torch
from architecture import *
from model import *
from data import *
from tools import *
from functions import *

# Hyperparameters

T = 10000
K = 10
N = 10

lambda_1 = 0.04
lambda_2 = 0.25
rho_bounds_1 = [0.2,0.3]
rho_bounds_2 = [0.6,0.7]
rhos = [rho_bounds_1,rho_bounds_2]
lambdas = [lambda_2,lambda_2]

metaparameters_multiparam = get_metaparameters(rhos,lambdas)
metaparameters_titles_multiparam = ['Case A','Case B','Case C','Case D']

learning_rate = 0.1
num_epochs = 30
batch_size = 64
num_layers = 50

test = UTitan(dimensions=(N,T,K),metaparameters=metaparameters_multiparam,train_size=1000,test_size=200,batch_size=batch_size,num_epochs=num_epochs,num_layers=num_layers,lr=learning_rate,N_updates_W=15)
test.train()
