import torch
from architecture import *
from model import *
from data import *
from tools import *
from functions import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gamma_c = 1
gamma_w = 0.99
eps = 1e-12
nu = 0.5
zeta = 1e-3 

# Hyperparameters

T = 10000
K = 10
N = 10

lambda_1 = 0.04
lambda_2 = 0.25
rho_bounds_1 = [0.2,0.3]
rho_bounds_2 = [0.6,0.7]
rhos = [rho_bounds_1,rho_bounds_2]
lambdas = [lambda_1,lambda_2]

metaparameters_multiparam = get_metaparameters(rhos,lambdas)
metaparameters_titles_multiparam = ['Case A','Case B','Case C','Case D']

learning_rate = 0.01
num_epochs = 60
batch_size = 20
num_layers = 70

test = UTitan(dimensions=(N,T,K),metaparameters=metaparameters_multiparam,train_size=1000,test_size=200,batch_size=100,num_epochs=num_epochs,num_layers=num_layers,lr=learning_rate)
# Afficher tous les noms et shapes des paramètres du modèle
# for name, param in test.model.named_parameters():
#     print(f"{name}: {param.shape}, requires_grad={param.requires_grad}")
test.train()

# batchA = torch.zeros(1,N,N,K)
# batchW = torch.zeros(1,N,N,K)
# for i in range(1):
#     X,A = generate_whitened_problem(T,K,N)
#     Y,W = generate_whitened_problem(T,K,N)
#     batchA[i,:,:,:] = A
#     batchW[i,:,:,:] = W

# print(joint_isi_batch(batchA,batchW)) 