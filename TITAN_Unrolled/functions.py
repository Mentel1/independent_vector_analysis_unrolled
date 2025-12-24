import torch
import numpy as np
from tools import *
from data import *
from initializations import _jbss_sos,_cca

def cost_iva_g_reg(W,C,Rx,alpha):
    det_C = torch.det(C.permute(2,0,1))  # Déterminant de C
    det_W = torch.det(W.permute(2,0,1))  # Déterminant de W
    tr_C = torch.trace((C - 1)**2)  # Trace de (C - 1)^2
    tr_term = torch.trace(torch.sum(torch.einsum('kKn,nNK,KJNM,nMJ -> kJn',(C,W,Rx,W)),dim=2)) / 2  # Terme de trace
    res = -torch.sum(torch.log(torch.abs(det_C))) / 2  # Premier terme
    res += 0.5 * alpha * tr_C  # Deuxième terme
    res += tr_term  # Troisième terme
    res -= torch.sum(torch.log(torch.abs(det_W)))  # Quatrième terme
    return res.item()  # Convertir le résultat en un scalaire Python

def grad_H_W(W,C,Rx):
    return torch.einsum('bKJN,bNMJ,bJKMm->bNmK',C,W,Rx)


def prox_f(W, c_w):
    W_perm = W.permute(0, 3, 1, 2)
    
    # VÉRIFICATIONS
    # print(f"W_perm - min: {W_perm.min():.6f}, max: {W_perm.max():.6f}")
    # print(f"W_perm - has NaN: {torch.isnan(W_perm).any()}, has Inf: {torch.isinf(W_perm).any()}")
    
    # Condition number (mesure si matrice mal conditionnée)
    # Plus c'est grand, plus c'est instable
    norms = torch.linalg.matrix_norm(W_perm, ord=2)
    # print(f"Matrix norms: min={norms.min():.6f}, max={norms.max():.6f}")
    
    # Vérifier si matrice trop grande
    if norms.max() > 1e6:
        print(f"WARNING: Very large matrix norm detected!")
    
    try:
        U, s, Vh = torch.linalg.svd(W_perm)
    except RuntimeError as e:
        print(f"SVD failed!")
        print(f"W_perm values: {W_perm}")
        raise e
    
    # Vérifier les valeurs singulières
    # print(f"Singular values - min: {s.min():.6f}, max: {s.max():.6f}")
    # if s.max() / (s.min() + 1e-10) > 1e10:
    #     print(f"WARNING: Very ill-conditioned matrix (condition number ~{s.max()/s.min():.2e})")
    
    # ... reste du code
    c_w = c_w.view(-1, 1, 1)
    s_new = (s + torch.sqrt(s**2 + 4 * c_w))
    s_new = s_new / 2
    diag_s = torch.diag_embed(s_new)
    W_new = torch.einsum('bknv,bkvw,bkwm -> bknm', U, diag_s, Vh)
    return W_new.permute(0, 2, 3, 1)


# def prox_f(W,c_w):
#     # Assume W has shape [B,N,N,K]
#     W_perm = W.permute(0,3,1,2)  # Permute to shape [B,K,N,N]
#     U,s,Vh = torch.linalg.svd(W_perm)
#     c_w = c_w.view(-1,1,1)  # Reshape c_w to broadcast correctly
#     s_new = (s + torch.sqrt(s**2 + 4*c_w)) / 2
#     diag_s = torch.diag_embed(s_new)
#     W_new = torch.einsum('bknv,bkvw,bkwm -> bknm',U,diag_s,Vh)
#     return W_new.permute(0,2,3,1)  # Shape back to [B,N,N,K]


def prox_g(C,c_c,epsilon):
    # Permute the tensor to get the eigenvalues in the correct shape
    C_perm = C.permute(0,3,1,2)  # (batch,last_dim,height,width)
    s,U = torch.linalg.eigh(C_perm)
    Vh = U.transpose(-1,-2)
    # Calculate the new singular values
    c_c = c_c.view(-1,1,1)  # Reshape c_c to broadcast correctly
    s_new = torch.maximum(epsilon,(s + torch.sqrt(s**2 + 2 * c_c)) / 2)
    diag_s = torch.diag_embed(s_new)
    
    # Reconstruct the matrix with the new singular values
    C_new = torch.einsum('bnNv,bnvw,bnwM -> bnNM',U,diag_s,Vh)
    C_new = C_new.permute(0,2,3,1)  # Permute back to original shape
    return sym(C_new)


def grad_H_C_reg(W,C,Rx,alpha):
    _, _, _, K = W.size()
    grad = sym(torch.einsum('bnNK,bKJNM,bnMJ->bKJn',W,Rx,W))/2
    indices = torch.arange(K).to(W.device)
    grad = grad.clone()
    grad[:, indices, indices, :] = grad[:, indices, indices, :] + alpha * (C[:, indices, indices, :] - 1)
    return grad

## a changer pour batch
def Jdiag_init(X,N,K,Rx):
    if K > 2:
        # initialize with multi-set diagonalization (orthogonal solution)
        W = _jbss_sos(X,0,'whole')
    else:
        W = _cca(X)
    W_bd = block_diag(W)
    W_bdT = np.moveaxis(W_bd,0,1)
    Sigma_tmp = np.einsum('KNn,Ni,ijn -> Kjn',W_bd,Rx,W_bdT)
    C = np.zeros((K,K,N))
    for n in range(N):
        C[:,:,n] = np.linalg.inv(Sigma_tmp[:,:,n])
    return W,C


def initialize(N,K,init_method='random',Winit=None,Cinit=None,X=None,Rx=None,seed=None,device='cpu'):
    if Winit is not None and Cinit is not None:
        W,C = Winit.clone(),Cinit.clone()
    elif init_method == 'Jdiag':
        W,C = Jdiag_init(X,N,K,Rx)
    elif init_method == 'random':
        C = make_Sigma(K,N,rank=K+10,seed=seed)
        W = make_A(K,N,seed=seed)      
    W = W.to(device)
    C = C.to(device)  
    return W,C




