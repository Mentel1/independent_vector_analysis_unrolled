import torch
import numpy as np
from .tools import *
from torch.utils.data import Dataset
from .helpers_iva import whiten_data


## Problem simumation functions 


def make_A(K,N,seed=None,device='cpu',dtype=torch.float64):
    if seed == None:
        A = torch.randn(N,N,K,dtype=dtype)
    else:
        torch.manual_seed(seed)
        A = torch.randn(N,N,K,dtype=dtype)
    A = A.to(device)
    return A


def make_A_debug(K,N,seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        
    A = torch.randn(N,N,K)
    
    # Debugging prints
    print(f"A shape: {A.shape}")
    print(f"A min value: {A.min()},A max value: {A.max()}")
    
    try:
        #A = A.to(device='cuda')
        pass
    except RuntimeError as e:
        print(f"Error when moving to CUDA: {e}")
        return None
    
    return A

def make_Sigma(K,N,rank,epsilon=1,rho_bounds=[0.4,0.6],lambda_=0.25,seed=None,normalize=False,device='cpu',dtype=torch.float64):
    
    rng = np.random.default_rng(seed)
    #if seed is not None :
    #    torch.manual_seed(seed)
    
    J = torch.ones(K,K,dtype=dtype)
    I = torch.eye(K,dtype=dtype)
    Q = torch.zeros(K,rank,N,dtype=dtype)
    mean = torch.zeros(K,dtype=dtype)
    Sigma = torch.zeros(K,K,N,dtype=dtype)
    if N == 1:
        rho = [torch.mean(rho_bounds)]
    else:
        rho = [(n/(N-1))*rho_bounds[1] + (1-(n/(N-1)))*rho_bounds[0] for n in range(N)]
    for n in range(N):
        eta = 1 - lambda_ - rho[n]
        if eta < 0 or lambda_ < 0 or rho[n] < 0:
            raise("all three coefficients must belong to [0,1]") 
        Q[:,:,n] = torch.tensor(rng.multivariate_normal(mean,I,rank).T)
        #Q[:,:,n] = torch.distributions.multivariate_normal.MultivariateNormal(mean,I).sample((rank,rank)).T
        if normalize:
            Q[:,:,n] = (Q[:,:,n].t() / torch.norm(Q[:,:,n],dim=1)).t()
            Sigma[:,:,n] = rho[n]*J + eta*I + lambda_*torch.matmul(Q[:,:,n],Q[:,:,n].t())
        else:
            Sigma[:,:,n] = rho[n]*J + eta*I + (lambda_/rank)*torch.matmul(Q[:,:,n],Q[:,:,n].t())
    for n in range(1,N):
        Sigma[:,:,n] = (1-epsilon)*Sigma[:,:,0] + epsilon*Sigma[:,:,n]
    Sigma = Sigma.to(device)
    return Sigma



""" def make_S(Sigma,V):
    _,K,N = Sigma.size()
    S = torch.zeros(N,V,K)
    mean = torch.zeros(K)
    for n in range(N):
        S[n,:,:] = torch.tensor(np.random.multivariate_normal(mean,Sigma[:,:,n],V))
        #S[n,:,:] = torch.normal(mean,torch.sqrt(Sigma[:,:,n]),(V,K))
    return S """

def make_S(Sigma,V,device='cpu',dtype=torch.float64):
    _,K,N = Sigma.size()
    S = torch.zeros(N,V,K,device=device,dtype=dtype)
    mean = torch.zeros(K,device=device,dtype=dtype)
    for n in range(N):
        cov_matrix = Sigma[:,:,n]
        mvn = torch.distributions.MultivariateNormal(mean,cov_matrix)
        S[n,:,:] = mvn.sample((V,))
    return S


def make_X(S,A):
    X = torch.einsum('MNK,NVK -> MVK',A,S)
    return X


def generate_whitened_problem(V,K,N,epsilon=1,rho_bounds=[0.4,0.6],lambda_=0.25,device='cpu',dtype=torch.float64,only_sos=True): 
    A = make_A(K,N,dtype=dtype)
    Sigma = make_Sigma(K,N,rank=K+10,epsilon=epsilon,rho_bounds=rho_bounds,lambda_=lambda_,seed=None,normalize=False,dtype=dtype)
    S = make_S(Sigma,V,dtype=dtype)
    X = make_X(S,A)
    X_,U = whiten_data(X)
    A_ = torch.einsum('nNk,Nvk->nvk',U,A)
    X_ = X_.to(device)
    A_ = A_.to(device)
    if not only_sos:
        return X_,A_
    else:
        Rx = cov_X(X_)
        return Rx,A_


def get_dataparameters(rhos,lambdas):
    dataparameters_multiparam = []
    for rho_bounds in rhos:
        for lambda_ in lambdas:
            dataparameters_multiparam.append((rho_bounds,lambda_))
    return dataparameters_multiparam


