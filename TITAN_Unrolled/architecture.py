"""
UTitan model classes
Classes
-------
    ISI_loss  : defines the ISI training loss 
    nn_alpha    : predicts the regularisation parameter
    W_iter     : computes the updates of W
    C_iter    : computes the updates of C
    Block      : one layer in U_TITAN
    myModel    : U_TITAN model


@author: Gaspard Blaise
@date: 11/06/2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functions import *
from tools import *
from data import *
from torch.utils.checkpoint import checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ISI_loss():
    """
    Defines the ISI training loss.
    Attributes
    ----------
        ISI : function computing the ISI Score 
    """
    def __init__(self): 
        super(ISI_loss,self).__init__()
        
    def __call__(self,input,target):
        """
        Computes the training loss.
        Parameters
        ----------
            input  (torch.FloatTensor): restored images,size n*c*h*w 
            target (torch.FloatTensor): ground-truth images,size n*c*h*w
        Returns
        -------
            (torch.FloatTensor): mean ISI Score of the batch,size 1 
        """
        return joint_isi_batch(input,target)

class IVA_loss():
    def __init__(self): 
        super(ISI_loss,self).__init__()
        
    def __call__(self,input): # A modifier pour faire le calcul sur tout le batch (W et C sont de dim B*_*_*_)
        W,C,Rx,alpha = input
        det_C = torch.det(C.permute(2,0,1))  # Déterminant de C
        det_W = torch.det(W.permute(2,0,1))  # Déterminant de W
        tr_C = torch.trace((C - 1)**2)  # Trace de (C - 1)^2
        tr_term = torch.trace(torch.sum(torch.einsum('kKn,nNK,KJNM,nMJ -> kJn',(C,W,Rx,W)),dim=2)) / 2  # Terme de trace
        res = -torch.sum(torch.log(torch.abs(det_C))) / 2  # Premier terme
        res += 0.5 * alpha * tr_C  # Deuxième terme
        res += tr_term  # Troisième terme
        res -= torch.sum(torch.log(torch.abs(det_W)))  # Quatrième terme
        return res.item()  # Convertir le résultat en un scalaire Python


class Custom_param(nn.Module):
    """
    Computes the parameters of the current layer given W, W_prev, C and C_prev.
    Attributes
    ----------
        fc1 (torch.nn.Linear): fully connected layer
        fc2 (torch.nn.Linear): fully connected layer
        soft (torch.nn.Softplus): Softplus activation function
    """
    def __init__(self,input_size,hidden_size,output_size=1):
        super(Custom_param,self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size) 
        self.fc2 = nn.Linear(hidden_size,output_size)
        self.soft = nn.Softplus()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        
    
    def forward(self, W, W_prev, C, C_prev):
        batch_size = W.shape[0]
        
        # Utilise reshape au lieu de view
        W = W.reshape(batch_size, -1)        # (b, N*M*K)
        W_prev = W_prev.reshape(batch_size, -1)
        C = C.reshape(batch_size, -1)        # (b, K*K*N)
        C_prev = C_prev.reshape(batch_size, -1)
        
        x = torch.cat((W, W_prev, C, C_prev), dim=1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.tanh(x)
        x = self.fc2(x) 
        x = (1+self.tanh(x-3))*torch.tensor([1,1,1,0.1,0.1],device=x.device)
        
        return x



class W_iter(nn.Module):

    def __init__(self,N_updates_W,inertial=False):
        super(W_iter,self).__init__()
        self.N_updates_W = N_updates_W
        self.inertial = inertial

    def inertial_step(self,W,W_prev,beta_w):
        if self.inertial:
            beta_w = beta_w.view(-1,1,1,1)
            W = W + beta_w * (W - W_prev)
        return W
    
    def gradient_step(self,Rx,c_w,W,C):
        c_w = c_w.view(-1,1,1,1)
        grad = grad_H_W(W,C,Rx)
        W = W - c_w * grad
        return W

    def prox_step(self,c_w,W):
        return prox_f(W,c_w)

    def update(self,Rx,W,W_prev,C,c_w,beta_w):
        W_inertial = self.inertial_step(W,W_prev,beta_w)
        W_gradient = self.gradient_step(Rx,c_w,W_inertial,C)
        W_prox = self.prox_step(c_w,W_gradient)
        return W_prox,W
    
    def forward(self,Rx,W,W_prev,C,c_w,beta_w):
        for j in range(self.N_updates_W):
            W,W_prev = self.update(Rx,W,W_prev,C,c_w,beta_w)
        return W,W_prev
    

class C_iter(nn.Module):
    def __init__(self,N_updates_C,epsilon,inertial=False):
        super(C_iter,self).__init__()
        self.N_updates_C = N_updates_C
        self.inertial = inertial
        self.epsilon = torch.tensor(epsilon,device=device)

    def inertial_step(self,C,C_prev,beta_c):
        if self.inertial:
            beta_c = beta_c.view(-1,1,1,1)
            C = C + beta_c * (C - C_prev)
        return C
    
    def gradient_step(self,Rx,c_c,C,W,alpha):
        c_c = c_c.view(-1,1,1,1)
        grad = grad_H_C_reg(W,C,Rx,alpha)
        C = C - c_c * grad
        return C

    def prox_step(self,c_c,C):
        return prox_g(C,c_c,self.epsilon)
    
    def update(self,Rx,C,C_prev,W,c_c,beta_c,alpha):
        C_inertial = self.inertial_step(C,C_prev,beta_c)
        C_gradient = self.gradient_step(Rx,c_c,C_inertial,W,alpha)
        C_prox = self.prox_step(c_c,C_gradient)
        return C_prox,C

    def forward(self,Rx,C,C_prev,W,c_c,beta_c,alpha):
        for j in range(self.N_updates_C):
            C,C_prev = self.update(Rx,C,C_prev,W,c_c,beta_c,alpha)
        return C,C_prev


class Block(nn.Module):

    def __init__(self,N_updates_W,N_updates_C,epsilon,inertial=False,custom=False,N=10,K=10):
    
        super().__init__()
        self.W_iter = W_iter(N_updates_W,inertial=inertial)
        self.C_iter = C_iter(N_updates_C,epsilon,inertial=inertial)
        self.epsilon = epsilon
        self.soft = nn.Softplus()
        self.tanh = nn.Tanh()
        self.custom = custom
        if self.custom:
            total_dim = 2*N*K*(N+K)
            output_size = 5
            self.get_coeff_module = Custom_param(input_size=total_dim,hidden_size=128,output_size=output_size)
        else:
            self.alpha = nn.Parameter(torch.zeros(1).to(device))
            self.beta_w = nn.Parameter(torch.zeros(1).to(device))
            self.beta_c = nn.Parameter(torch.zeros(1).to(device))      
            self.gamma_w = nn.Parameter(torch.empty(1).to(device))
            torch.nn.init.normal_(self.gamma_w,mean=-1.3,std=0.1)
            self.gamma_c = nn.Parameter(torch.empty(1).to(device))
            torch.nn.init.normal_(self.gamma_c,mean=-1.3,std=0.1)
        
          
    def get_coefficients(self,rho_Rx,C,C_prev,W,W_prev):
        if self.custom:
            return self.get_coeff_module(W,W_prev,C,C_prev).moveaxis(1,0)
        else:
            batch_size = W.shape[0]
            alpha = self.soft(self.alpha)*torch.ones(batch_size,device=W.device)
            beta_w = self.soft(self.beta_w-1)*torch.ones(batch_size,device=W.device)
            beta_c = self.soft(self.beta_c-1)*torch.ones(batch_size,device=W.device)
            gamma_w = 0.3 + 5*(self.tanh(self.gamma_w)+1)
            gamma_c = 0.3 + 5*(self.tanh(self.gamma_c)+1)   
            L_w = lipschitz(C,rho_Rx)
            c_w = gamma_w/L_w  
            c_c = gamma_c/alpha
            return torch.stack([alpha, c_w, c_c, beta_w, beta_c])   

    def forward(self,Rx,rho_Rx,W,W_prev,C,C_prev,i):
        alpha,c_w,c_c,beta_w,beta_c=self.get_coefficients(rho_Rx,C,C_prev,W,W_prev)
        # if i==0:
        #     print(alpha,c_w,c_c,beta_w,beta_c)
        if torch.isnan(alpha).any() or torch.isnan(c_w).any() or torch.isnan(c_c).any() or torch.isnan(beta_w).any() or torch.isnan(beta_c).any():
            print(f"NaN détecté au niveau des paramètres dans la couche {i} !")
        W_new,W=self.W_iter(Rx,W,W_prev,C,c_w,beta_w)
        C_new,C=self.C_iter(Rx,C,C_prev,W_new,c_c,beta_c,alpha)
        if torch.isnan(W_new).any() or torch.isnan(W).any() or torch.isnan(C_new).any() or torch.isnan(C).any():
            print(f"NaN détecté au niveau des tenseurs de sortie dans la couche {i} !")
        
        return W_new,C_new,W,C


class UTitanIVAGModel(nn.Module):

    def __init__(self,N_updates_W,N_updates_C,num_layers,epsilon,inertial=False,custom=False,N=10,K=10):
        super().__init__()
        self.Layers = nn.ModuleList([Block(N_updates_W,N_updates_C,epsilon,inertial=inertial,custom=custom,N=N,K=K) for _ in range(num_layers)])

    def forward(self,Rx,Winit,Cinit):
        _,N,_,K = Winit.shape
        rho_Rx = spectral_norm_extracted(Rx,K,N)
        W = Winit
        C = Cinit
        W_prev = W.clone()
        # C_i_1 = C.clone()
        C_prev = C.clone()
        for i in range(len(self.Layers)):
            # print(i, torch.cuda.memory_allocated() / 1024**3)
            try:
                W,C,W_prev,C_prev = checkpoint(self.Layers[i],Rx,rho_Rx,W,W_prev,C,C_prev,i,use_reentrant=False)
            except Exception as e:
                print(f"Error at layer {i}: {e}")
                print("non rien")
                raise e                
        return W,C
    
