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
import traceback
from contextlib import nullcontext
from .functions import *
from .tools import *
from .data import *
from torch.utils.checkpoint import checkpoint

class ISI_loss():
    """
    Defines the ISI training loss.
    Attributes
    ----------
        ISI : function computing the ISI Score 
    """
    def __init__(self): 
        super(ISI_loss,self).__init__()
        
    def __call__(self,outputs,greedy=False):
        W,store_W,A = outputs['W'],outputs['store_W'],outputs['A']
        num_layers = store_W.shape[0]
        """
        Computes the training loss.
        Parameters
        ----------
            output  (torch.FloatTensor): restored images,size n*c*h*w 
            label (torch.FloatTensor): ground-truth images,size n*c*h*w
        Returns
        -------
            (torch.FloatTensor): mean ISI Score of the batch,size 1 
        """
        if greedy:
            loss = torch.zeros(1,device=W.device)
            for i in range(num_layers):
                W = store_W[i,:,:,:,:]
                loss += joint_isi_batch(W,A)/num_layers
            return loss
        else:
            return joint_isi_batch(W,A)


class IVA_loss():
    def __init__(self): 
        super(IVA_loss,self).__init__()
        
    def __call__(self,outputs,greedy=False): # A modifier pour faire le calcul sur tout le batch (W et C sont de dim B*_*_*_)
        W,C,Rx = outputs['W'],outputs['C'],outputs['Rx']
        res = torch.zeros(1,device=W.device)
        if greedy:
            store_W,store_C = outputs['store_W'],outputs['store_C']
            for i in range(store_W.shape[0]):
                W,C = store_W[i,:,:,:,:],store_C[i,:,:,:,:]
                det_C = torch.det(C.permute(0,3,1,2))  # Déterminant de C
                det_W = torch.det(W.permute(0,3,1,2))  # Déterminant de W
                tr_C = torch.diagonal((C.permute(0,3,1,2) - 1) ** 2, dim1=-2, dim2=-1).sum()
                tr_term = (torch.einsum('bKJn,bnNK,bKJNM,bnMJ -> bnKJ',(C,W,Rx,W))).sum() / 2  # Terme de trace
                res -= torch.sum(torch.log(torch.abs(det_C))) / 2  # Premier terme
                res += 0.5 * tr_C  # Deuxième terme
                res += tr_term  # Troisième terme
                res -= torch.sum(torch.log(torch.abs(det_W)))  # Quatrième terme
        else:
            det_C = torch.det(C.permute(0,3,1,2))  # Déterminant de C
            term1 = torch.sum(torch.log(torch.abs(det_C)))
            # print(f'sum log det_C is computed and equals to {term1}')
            det_W = torch.det(W.permute(0,3,1,2))  # Déterminant de W
            term4 = torch.sum(torch.log(torch.abs(det_W)))
            # print(f'sum log |det_W| is computed and equals to {term4}')
            tr_C = torch.diagonal((C.permute(0,3,1,2) - 1) ** 2, dim1=-2, dim2=-1).sum()
            # print(f'tr_C is computed and equals to {tr_C}')
            tr_term = (torch.einsum('bKJn,bnNK,bKJNM,bnMJ -> bnKJ',(C,W,Rx,W))).sum() / 2  # Terme de trace
            # print(f'tr_term is computed and equals to {tr_term}')
            res -= term1/ 2  # Premier terme
            res += 0.5 * tr_C  # Deuxième terme
            res += tr_term  # Troisième terme
            res -= term4  # Quatrième terme
        return res


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
    
'''
Amélioration prévue du module Custom_param: au lieu de calculer les 5 paramètres d'un coup au début du bloc et avec une logique indifférenciée pour chaque paramètre, on peut sans doute faire beaucoup plus subtil avec moins de paramètres au total pour éviter l'overfitting et l'instabilité qu'on observe dans la version actuelle.

D'abord, on pourrait subdiviser les paramètres:
- c_w -> un c_wn_grad pour chaque nabla_W_n h et un c_wk_prox pour chaque prox_f (W_k)
- c_c, alpha -> même logique
- beta_w et beta_c -> même logique en subdivisant en beta_wn et beta_cn

Les paramètres seraient recalculés plusieurs fois juste avant leur utilisation réelle (on aurait donc en pratique des beta_wn^(i,j) au lieu d'un simple beta_w^(i). D'ailleurs, on pourrait utiliser un seul module pour toutes les couches, l'adaptabilité venant déjà de la prise en compte de l'état courant des variables. En contrepartie on aurait la possibilité d'avoir des modules assez complexes avec plusieurs couches cachées.)
'''

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
        W = W.double()
        c_w = c_w.double()
        return prox_f(W,c_w).float()

    def update(self,Rx,W,W_prev,C,c_w,beta_w):
        W_inertial = self.inertial_step(W,W_prev,beta_w)
        W_gradient = self.gradient_step(Rx,c_w,W_inertial,C)
        W_prox = self.prox_step(c_w,W_gradient)
        return W_prox,W
    
    def forward(self,Rx,W,W_prev,C,c_w,beta_w,i):
        for j in range(self.N_updates_W):
            try:
                W,W_prev = self.update(Rx,W,W_prev,C,c_w,beta_w)
            except Exception as e:
                print(f'error at layer {i}, sublayer {j} raising the following : {e}')
        return W,W_prev
    

class C_iter(nn.Module):
    def __init__(self,N_updates_C,epsilon,inertial=False):
        super(C_iter,self).__init__()
        self.N_updates_C = N_updates_C
        self.inertial = inertial
        self.epsilon = torch.tensor(epsilon,dtype=torch.float64)

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
        C = C.double()
        c_c = c_c.double()
        return prox_g(C,c_c,self.epsilon).float()
    
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
        self.debug_history = []
        if self.custom:
            total_dim = 2*N*K*(N+K)
            output_size = 5
            self.get_coeff_module = Custom_param(input_size=total_dim,hidden_size=128,output_size=output_size)
        else:
            self.alpha = nn.Parameter(torch.zeros(1))     
            self.gamma_w = nn.Parameter(torch.empty(1))
            torch.nn.init.normal_(self.gamma_w,mean=1,std=0.1)
            self.gamma_c = nn.Parameter(torch.empty(1))
            torch.nn.init.normal_(self.gamma_c,mean=1,std=0.1)
            self.beta_w = nn.Parameter(torch.zeros(1))
            self.beta_c = nn.Parameter(torch.zeros(1)) 
        
          
    def get_coefficients(self,rho_Rx,C,C_prev,W,W_prev):
        if self.custom:
            return self.get_coeff_module(W,W_prev,C,C_prev).moveaxis(1,0)
        else:
            batch_size = W.shape[0]
            alpha = self.soft(self.alpha)*torch.ones(batch_size,device=W.device)
            gamma_w = self.soft(self.gamma_w)
            gamma_c = self.soft(self.gamma_c)
            beta_w = self.soft(self.beta_w)*torch.ones(batch_size,device=W.device)/10
            beta_c = self.soft(self.beta_c)*torch.ones(batch_size,device=W.device)/10
            L_w = lipschitz(C,rho_Rx)
            c_w = gamma_w/L_w  
            c_c = gamma_c/alpha
            return torch.stack([alpha,c_w,c_c,beta_w,beta_c])   

    def forward(self,Rx,rho_Rx,W,W_prev,C,C_prev,i):
        alpha,c_w,c_c,beta_w,beta_c=self.get_coefficients(rho_Rx,C,C_prev,W,W_prev)
        W_new,W=self.W_iter(Rx,W,W_prev,C,c_w,beta_w,i)
        C_new,C=self.C_iter(Rx,C,C_prev,W_new,c_c,beta_c,alpha)      
        return W_new,C_new,W,C
    


class UTitanIVAGModel(nn.Module):

    def __init__(self,N_updates_W,N_updates_C,num_layers,epsilon,archi='untied',custom=False,N=10,K=10):
        super().__init__()
        self.inertial = ('inertial' in archi)
        self.tied = ('untied' not in archi)
        self.num_layers = num_layers
        if self.tied:
            self.Layer = Block(N_updates_W,N_updates_C,epsilon,inertial=self.inertial,custom=custom,N=N,K=K)
        else:
            self.Layers = nn.ModuleList([Block(N_updates_W,N_updates_C,epsilon,inertial=self.inertial,custom=custom,N=N,K=K) for _ in range(num_layers)])
        

    def forward(self,Rx,Winit,Cinit,learning_layers=(0,float('inf')),track_jisi=False,A=None,track_cost=False,greedy=False):
        first_layer,last_layer=learning_layers
        B,N,_,K = Winit.shape
        rho_Rx = spectral_norm_extracted(Rx,K,N)
        W,C = Winit,Cinit
        W_prev,C_prev = W.clone(),C.clone()
        num_iter = self.num_layers
        if first_layer == last_layer:
            num_iter = first_layer+1
        outputs = {'cost':torch.full((num_iter + 1,), float('inf')),'jisi':torch.full((num_iter + 1,), float('inf'))}
        if greedy:
            outputs['loss'] = 0
        if track_cost:     
            outputs['cost'][0] = cost_iva_g_reg(W,C,Rx,alpha=1)
        if track_jisi:
            outputs['jisi'][0] = joint_isi_batch(W,A)
        for i in range(num_iter):
            layer = self.Layer if self.tied else self.Layers[i]  
            with torch.no_grad() if (i < first_layer or i > last_layer) else nullcontext():
                W, C, W_prev, C_prev = checkpoint(layer, Rx, rho_Rx, W, W_prev, C, C_prev, i, use_reentrant=False)
            if track_cost or greedy:
                # print(f"Layer {i}: greedy={greedy}, track_cost={track_cost}, using no_grad={not greedy and i < num_iter - 1}")
                with torch.no_grad() if (not greedy and i < num_iter - 1) else nullcontext():
                    outputs['cost'][i+1] = cost_iva_g_reg(W,C,Rx,alpha=1)
                # print(f"  Cost computed, requires_grad={outputs['cost'][i+1].requires_grad if hasattr(outputs['cost'][i+1], 'requires_grad') else 'N/A'}")
            if greedy:
                outputs['loss'] += outputs['cost'][i+1]
            if track_jisi:
                with torch.no_grad():
                    outputs['jisi'][i+1] = joint_isi_batch(W,A)
        outputs['W'] = W
        outputs['C'] = C
        return outputs
