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
    
    def gradient_step(self,Rx,c_w,W,C):
        c_w = c_w.view(-1,1,1,1)
        W = W - c_w * grad_H_W(W,C,Rx)
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
        C = C - c_c * grad_H_C_reg(W,C,Rx,alpha)
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

    """
    One layer in U_TITAN.
    Attributes
    ----------
        nn_bar                            (Cnn_bar): computes the barrier parameter
        soft                    (torch.nn.Softplus): Softplus activation function
        gamma                (torch.nn.FloatTensor): stepsize,size 1 
        reg_mul,reg_constant (torch.nn.FloatTensor): parameters for estimating the regularization parameter,size 1
        delta                               (float): total variation smoothing parameter
        IPIter                             (IPIter): computes the next proximal interior point iterate
    """


    def __init__(self,N_updates_W,N_updates_C,epsilon,nu,zeta):
    
        super().__init__()
        #self.NN_alpha = FCNN_alpha(input_dim,32)
        self.W_iter = W_iter(N_updates_W)
        self.C_iter = C_iter(N_updates_C)
        self.alpha_in = nn.Parameter(torch.zeros(1).to(device))
        self.soft = nn.Softmax()
        self.tanh = nn.Tanh()
        self.gamma_w = nn.Parameter(torch.empty(1).to(device))
        torch.nn.init.normal_(self.gamma_w, mean=-1.289, std=0.1)
        self.gamma_c = nn.Parameter(torch.empty(1).to(device))
        torch.nn.init.normal_(self.gamma_c, mean=-1.289, std=0.1)
        self.epsilon = epsilon
        self.nu = nu
        self.zeta = zeta
    
    def get_coefficients(self,rho_Rx,C,alpha,gamma_c,gamma_w):
        L_w = lipschitz(C,rho_Rx)
        c_w = gamma_w/L_w  
        c_c = gamma_c/alpha
        return c_w,c_c


    def forward(self,Rx,rho_Rx,W,C,i):
            
        alpha = self.soft(self.alpha_in)
        gamma_w = 0.3 + 5*(self.tanh(self.gamma_w)+1)
        gamma_c = 0.3 + 5*(self.tanh(self.gamma_c)+1)
        # print(f"stepsize factors at iteration {i}: gamma_w : {gamma_w}, gamma_c : {gamma_c}, alpha : {alpha}")
        c_w,c_c=self.get_coefficients(rho_Rx,C,alpha,gamma_c,gamma_w)
        W=self.W_iter(Rx,W,C,c_w)
        C=self.C_iter(Rx,C,W,c_c,alpha,self.epsilon)
        
        return W,C


class UTitanIVAGModel(nn.Module):

    def __init__(self,N_updates_W,N_updates_C,num_layers,epsilon,nu,zeta):
        super().__init__()
        self.Layers = nn.ModuleList([Block(N_updates_W,N_updates_C,epsilon,nu,zeta) for _ in range(num_layers)])
        #self.alphas = nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for _ in range(num_layers)])
        #self.soft = nn.Softplus()

    def forward(self,X,Winit,Cinit,mode="end-to-end"):
        B,N,_,K = X.shape
        Rx = cov_X(X)
        rho_Rx = spectral_norm_extracted(Rx,K,N)
        W = Winit
        C = Cinit
        # W_j_1 = W.clone()
        # C_i_1 = C.clone()
        # C_j_1 = C.clone()
        for i in range(len(self.Layers)):
            W,C = self.Layers[i](Rx,rho_Rx,W,C,i)
        return W,C
    



















"""
class DeterministicBlock(nn.Module):
    def __init__(self,Rx,rho_Rx,gamma_c,gamma_w,eps,nu,zeta):
        super(DeterministicBlock,self).__init__()
        self.Rx = Rx
        self.rho_Rx = rho_Rx
        self.gamma_c = gamma_c
        self.gamma_w = gamma_w
        self.eps = eps 
        self.nu = nu
        self.zeta = zeta
        self.device = 'cuda:0'    

    def forward(self,W,C,alpha,L_w_prev):
        K = W.shape[2]
        alpha = alpha.to(self.device)
        #print(alpha.device)

        l_sup = max((self.gamma_w*alpha)/(1-self.gamma_w),self.rho_Rx*2*K*(1+torch.sqrt(2/(alpha*self.gamma_c))))
        C0 = min(self.gamma_c**2/K**2,(alpha*self.gamma_w/((1+self.zeta)*(1 - self.gamma_w)*l_sup)),(self.rho_Rx/((1+self.zeta)*l_sup)))
        l_inf = (1+self.zeta)*C0*l_sup

        c_c = self.gamma_c / alpha
        beta_c = torch.sqrt(C0*self.nu*(1-self.nu))
        L_w = max(l_inf,lipschitz(C,self.rho_Rx))
        c_w = self.gamma_w / L_w
        beta_w = (1 - self.gamma_w) * torch.sqrt(C0 * self.nu * (1 - self.nu) * L_w_prev / L_w)
        W_prev = W.clone()
        W_prev = W_prev.to(self.device)
        #print(W.device)	

        for _ in range(10):
            
            W_tilde = W + beta_w * (W - W_prev)
            grad_W = grad_H_W(W_tilde,C,self.Rx)
            W_bar = W_tilde - c_w * grad_W
            W_prev = W.clone()
            W = prox_f(W_bar,c_w)

        C_prev = C.clone()
        beta_c = torch.sqrt(C0 * self.nu * (1 - self.nu))
        C_tilde = C + beta_c * (C - C_prev)
        grad_C = grad_H_C_reg(W,C_tilde,self.Rx,alpha)
        C_bar = C_tilde - c_c * grad_C
        C_prev = C.clone()
        C = prox_g(C_bar,c_c,self.eps)

        return W,C,L_w




# Define the TitanLayer
class TitanLayer(nn.Module):
    def __init__(self,Rx,rho_Rx,gamma_c,gamma_w,eps,nu,input_dim,zeta):
        super(TitanLayer,self).__init__()
        self.alpha_net = AlphaNetwork(input_dim)
        self.deterministic_block = DeterministicBlock(Rx,rho_Rx,gamma_c,gamma_w,eps,nu,zeta)

    def forward(self,W,C,L_w_prev):
        alpha = self.alpha_net(W,C)
        W,C,L_w = self.deterministic_block(W,C,alpha,L_w_prev)
        return W,C,L_w,alpha
    




class TitanIVAGNet(nn.Module):
    def __init__(self,input_dim,num_layers=20,gamma_c=1,gamma_w=0.99,eps=1e-12,nu=0.5,zeta=0.1):
        super(TitanIVAGNet,self).__init__()
        self.num_layers = num_layers
        self.gamma_c = torch.tensor(gamma_c)
        self.gamma_w = torch.tensor(gamma_w)
        self.eps = torch.tensor(eps)
        self.nu = torch.tensor(nu)  
        self.zeta = torch.tensor(zeta)
        self.alpha_network = AlphaNetwork(input_dim)
        self.alphas = [torch.FloatTensor([1]).to('cuda') for _ in range(num_layers)]
        self.input_dim = input_dim
        self.layers = nn.ModuleList([
            TitanLayer(None,None,gamma_c,gamma_w,eps,nu,input_dim,zeta)
            for _ in range(num_layers)
        ])
    


    def initialize_L_w(self,C,rho_Rx,K):
        l_sup = max((self.gamma_w * self.alphas[0]) / (1 - self.gamma_w),rho_Rx * 2 * K * (1 + torch.sqrt(2 / (self.alphas[0] * self.gamma_c))))
        C0 = min(self.gamma_c**2 / K**2,(self.alphas[0] * self.gamma_w / ((1 + self.zeta) * (1 - self.gamma_w) * l_sup)),(rho_Rx / ((1 + self.zeta) * l_sup)))
        l_inf = (1 + self.zeta) * C0 * l_sup
        return max(l_inf,lipschitz(C,rho_Rx))
    
    
    def forward(self,X,A):
        N,_,K = X.shape
        input_dim = N * N * K + K * K * N
        Rx = cov_X(X)
        rho_Rx = spectral_norm_extracted(Rx,K,N)

        
        W,C = initialize(N,K,init_method='random',Winit=None,Cinit=None,X=X,Rx=Rx,seed=None)
        
        L_w_prev = self.initialize_L_w(C,rho_Rx,K)


        for i,layer in enumerate(self.layers):

            layer.deterministic_block = DeterministicBlock(Rx,rho_Rx,self.gamma_c,self.gamma_w,self.eps,self.nu,self.zeta)  # Ensure each layer has its own deterministic block
            W,C,L_w,alpha = layer(W,C,L_w_prev)
            L_w_prev = L_w
            self.alphas[i] = alpha
            

        
        isi_score = joint_isi(W,A)

        return W,C,isi_score

"""