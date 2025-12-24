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
        # batch_size = input.shape[0]
        # isi_scores = []

        # for i in range(batch_size):
        #     W = input[i] # Select the i-th element in the batch and add a batch dimension
        #     A = target[i]  # Select the i-th element in the batch and add a batch dimension
        #     score = joint_isi(W,A)
        #     isi_scores.append(score)

        # isi_scores = torch.stack(isi_scores)  # Stack scores into a tensor
        # return torch.mean(isi_scores)
        return joint_isi_batch(input,target)



class FCNN_alpha(nn.Module):
    """
    Predicts the regularization parameter alpha given W and C.
    Attributes
    ----------
        fc1 (torch.nn.Linear): fully connected layer
        fc2 (torch.nn.Linear): fully connected layer
        soft (torch.nn.Softplus): Softplus activation function
    """
    def __init__(self,input_size,hidden_size,output_size=1):
        super(FCNN_alpha,self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,output_size)
        self.soft = nn.Softplus()
        
    def forward(self,W,C):
        """
        Computes the regularization parameter alpha.
        Parameters
        ----------
            W (torch.FloatTensor): input tensor W,size [batch_size,W_size]
            C (torch.FloatTensor): input tensor C,size [batch_size,C_size]
        Returns
        -------
            torch.FloatTensor: alpha parameter,size [batch_size,1]
        """
        # Flatten W to size [N*N*K]
        W = W.view(-1)
        # Flatten C to size [K*K*N]
        C = C.view(-1)
        # Concatenate W and C along the last dimension
        x = torch.cat((W,C))
        x = self.soft(self.fc1(x))
        x = self.fc2(x)
        x = self.soft(x)
        return x



class W_iter(nn.Module):

    def __init__(self,N_updates_W):
        super(W_iter,self).__init__()
        self.N_updates_W = N_updates_W

    # def inertial_step(self,beta_w,W,W_old):
    #     beta_w = beta_w.view(-1,1,1,1)
    #     W = W + beta_w * (W - W_old)
    #     return W
    
    def gradient_step(self, Rx, c_w, W, C):
        c_w = c_w.view(-1, 1, 1, 1)
        grad = grad_H_W(W, C, Rx)
        W = W - c_w * grad
        return W

    def prox_step(self,c_w,W):
        return prox_f(W,c_w)

    def update(self,Rx,W,C,c_w):
        W_gradient = self.gradient_step(Rx,c_w,W,C)
        W_prox = self.prox_step(c_w,W_gradient)
        return W_prox
    
    def forward(self,Rx,W,C,c_w):
        for j in range(self.N_updates_W):
            # print(f"W shape before subiteration {j}: {W.shape}")
            # print(f"W min: {W.min()}, max: {W.max()}")
            # print(f"Has NaN: {torch.isnan(W).any()}")
            # print(f"Has Inf: {torch.isinf(W).any()}")
            W = self.update(Rx,W,C,c_w)
        return W
    

class C_iter(nn.Module):
    def __init__(self,N_updates_C):
        super(C_iter,self).__init__()
        self.N_updates_C = N_updates_C

    # def inertial_step(self,beta_c,C,C_old):
    #     beta_c = beta_c.view(-1,1,1,1)
    #     C = C + beta_c * (C - C_old)
    #     return C
    
    def gradient_step(self,Rx,c_c,C,W,alpha):
        c_c = c_c.view(-1,1,1,1)
        grad = grad_H_C_reg(W,C,Rx,alpha)
        C = C - c_c * grad
        return C

    def prox_step(self,c_c,C,epsilon):
        return prox_g(C,c_c,epsilon)
    
    def update(self,Rx,C,W,c_c,alpha,epsilon):
        C_gradient = self.gradient_step(Rx,c_c,C,W,alpha)
        C_prox = self.prox_step(c_c,C_gradient,epsilon)
        return C_prox

    def forward(self,Rx,C,W,c_c,alpha,epsilon):
        for j in range(self.N_updates_C):
            # print(f"C shape before subiteration {j}: {C.shape}")
            # print(f"C min: {C.min()}, max: {C.max()}")
            # print(f"Has NaN: {torch.isnan(C).any()}")
            # print(f"Has Inf: {torch.isinf(C).any()}")
            C = self.update(Rx,C,W,c_c,alpha,epsilon)
        return C


class Block(nn.Module):

    def __init__(self,N_updates_W,N_updates_C,epsilon,nu,zeta):
    
        super().__init__()
        self.W_iter = W_iter(N_updates_W)
        self.C_iter = C_iter(N_updates_C)
        self.alpha = nn.Parameter(torch.zeros(1).to(device))
        self.soft = nn.Softplus()
        self.tanh = nn.Tanh()
        self.gamma_w = nn.Parameter(torch.empty(1).to(device))
        torch.nn.init.normal_(self.gamma_w, mean=-1.5, std=0.1)
        self.gamma_c = nn.Parameter(torch.empty(1).to(device))
        torch.nn.init.normal_(self.gamma_c, mean=-1.5, std=0.1)
        self.epsilon = torch.tensor(epsilon,device=device)
        self.nu = nu
        self.zeta = zeta
    
    def get_coefficients(self,rho_Rx,C,alpha,gamma_c,gamma_w):
        L_w = lipschitz(C,rho_Rx)
        c_w = gamma_w/L_w  
        c_c = gamma_c/alpha
        return c_w,c_c


    def forward(self,Rx,rho_Rx,W,C,i):
            
        alpha = self.soft(self.alpha)
        gamma_w = 0.3 + 5*(self.tanh(self.gamma_w)+1)
        gamma_c = 0.3 + 5*(self.tanh(self.gamma_c)+1)
        # if i%10 == 0:
        #     print(f"stepsize factors at iteration {i}: gamma_w : {gamma_w}, gamma_c : {gamma_c}, alpha : {alpha}")
        c_w,c_c=self.get_coefficients(rho_Rx,C,alpha,gamma_c,gamma_w)
        W=self.W_iter(Rx,W,C,c_w)
        C=self.C_iter(Rx,C,W,c_c,alpha,self.epsilon)
        
        return W,C


class UTitanIVAGModel(nn.Module):

    def __init__(self,N_updates_W,N_updates_C,num_layers,epsilon,nu,zeta):
        super().__init__()
        self.Layers = nn.ModuleList([Block(N_updates_W,N_updates_C,epsilon,nu,zeta) for _ in range(num_layers)])

    def forward(self,X,Winit,Cinit,mode="end-to-end"):
        B,N,_,K = X.shape
        Rx = cov_X(X)
        print('test nan in Rx', torch.isnan(Rx).any())
        rho_Rx = spectral_norm_extracted(Rx,K,N)
        W = Winit
        C = Cinit
        # W_j_1 = W.clone()
        # C_i_1 = C.clone()
        # C_j_1 = C.clone()
        for i in range(len(self.Layers)):
            try:
                W,C = self.Layers[i](Rx,rho_Rx,W,C,i)
            except Exception as e:
                print(f"Error at layer {i}: {e}")
                # print("PARAMÈTRES DU MODÈLE:")
                # for name, param in self.named_parameters():
                #     if 'alpha' in name:   
                #         print(f"\n{name}: {torch.log(1+torch.exp(param))}\n")
                #     else:
                #         print(f"\n{name}: {0.3+5*(nn.Tanh()(param)+1)}\n")
                raise e
                
        return W,C
    
