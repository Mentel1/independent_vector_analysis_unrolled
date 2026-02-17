import torch
import numpy as np
from time import time
from .initializations import _jbss_sos,_cca
from .algebra_toolbox_torch import *

def cost_iva_g_reg(W,C,Rx,alpha):
    res = -torch.sum(torch.log(torch.det(C.permute(2,0,1))))/2
    res += 0.5*alpha*torch.sum(torch.trace((C-1)**2))
    WRx = torch.einsum('nNK,KJNM -> nKJM',W,Rx)
    WRxW = torch.einsum('nKJM,nMJ -> nKJ',WRx,W)
    res += torch.sum(C.permute(2,0,1)*WRxW)/2
    # res += torch.trace(torch.sum(torch.einsum('kKn,nNK,KJNM,nMJ -> kJn',C,W,Rx,W),axis=2))/2
    res -= torch.sum(torch.log(torch.abs(torch.det(W.permute(2,0,1)))))
    return res
# On ne traite pas le cas où une valeur singulière est inférieure à epsilon car ça ne peut
# pas arriver à cause du prox utilisé sauf à l'initialisation éventuellement mais on le néglige

def grad_H_W(W,C,Rx):
    # WRx = torch.einsum('NMJ,JKMm -> JKNm',W,Rx)
    # return torch.einsum('KJN,JKNm -> NmK',C,WRx)
    return torch.einsum('KJN,NMJ,JKMm->NmK',C,W,Rx)

def prox_f(W,c_w):
    U,s,Vh = torch.linalg.svd(W.permute(2,0,1))
    s_new = (s + torch.sqrt(s**2 + 4*c_w)) / 2
    W_new = torch.einsum('kNv,kvM->kNM',U * s_new.unsqueeze(1),Vh)
    return W_new.permute(1,2,0)

def prox_g(C,c_c,epsilon):
    s,U = torch.linalg.eigh(C.permute(2,0,1))
    Vh = U.transpose(1,2)
    s_new = torch.maximum(torch.tensor(epsilon),(s + torch.sqrt(s**2 + 2*c_c)) / 2)
    C_new = torch.einsum('nNv,nvM->nNM',U * s_new.unsqueeze(1),Vh)
    C_new = C_new.permute(1,2,0)
    return sym_torch(C_new)

def grad_H_C_reg(W,C,Rx,alpha):
    K = W.shape[2]
    grad = sym_torch(torch.einsum('nNK,KJNM,nMJ->KJn',W,Rx,W))/2
    idx = torch.arange(K)
    grad[idx,idx,:] += alpha*(C[idx,idx,:]-1)
    return grad

def compute_L_W(C,Rx,nu,l_inf,boost):
    K,N = C.shape[0],C.shape[2]
    if boost:
        Hess = torch.einsum('KJn,KJNM->nKNJM',C,Rx)
        Hess = Hess.reshape(N,N*K,N*K)
        L_w = torch.max(torch.linalg.matrix_norm(Hess,ord=2))
    else:
        L_w = lipschitz(C,spectral_norm_extracted_torch(Rx,K,N))
    if nu > 0:
        L_w = max(l_inf,L_w)
    return L_w

def titan_iva_g_reg_torch(Rx,alpha=1,gamma_c=1,gamma_w=0.99,max_iter=20000,max_iter_int_W=15,max_iter_int_C=1,crit_int=1e-10,crit_ext=1e-10,init_method='random',Winit=None,Cinit=None,epsilon=10**(-12),zeta=1e-3,track_times=True,track_costs=False,track_jisi=False,track_diffs=False,track_schemes=False,track_shifts=False,A=None,nu=0.5,adaptative_gamma_w=False,gamma_w_decay=0.9,boost=False,seed=None):
    alpha,gamma_c,gamma_w,N,K,rho_Rx = init_data_param(Rx,alpha,gamma_c,gamma_w)
    #Empiriquement,prend des valeurs entre 1 et 3 après whitening
    W,C = initialize(N,K,init_method=init_method,Winit=Winit,Cinit=Cinit,seed=seed)
    rho_bar = max(spectral_norm_torch(C),3*K*(1+torch.sqrt(1/(2*alpha*gamma_c))))
    l_sup = max((gamma_w*alpha)/(1-gamma_w),rho_Rx*rho_bar)
    C0 = min(4*gamma_c**2/(9*K**2),alpha*gamma_w/((1+zeta)*(1 - gamma_w)*l_sup))
    l_inf = (1+zeta)*C0*l_sup
    C_prev,W_prev=C.clone(),W.clone()
    c_c = gamma_c/alpha
    beta_c = torch.sqrt(C0*nu*(1-nu))
    #On initialise les listes utiles pour tracer les courbes. Par défaut on garde les critères pour les deux blocs,on pourra calculer le max a posteriori si besoin
    diffs_W,diffs_C,jISI,detailed_jISI,costs,detailed_costs,shifts_W,scheme,times,detailed_times = [],[],[],[],[],[],[],[],[],[]
    # shift_W = 1
    if track_costs:
        costs.append(cost_iva_g_reg(W,C,Rx,alpha))
    if track_jisi:
        if torch.any(A == None):
            raise("you must provide A to track jISI")
        else:
            jISI.append(joint_isi_torch(W,A))
    if track_times:
        t0 = time()
        times.append(0)
    diff_ext = torch.inf
    N_step = 0
    L_w = compute_L_W(C,Rx,nu,l_inf,boost)
    # dW = np.zeros_like((W))
    # gamma_w0 = gamma_w
    while diff_ext > crit_ext and N_step < max_iter:
        W_old = W.clone()
        C_old = C.clone()
        L_w_prev = L_w
        L_w = compute_L_W(C,Rx,nu,l_inf,boost) #Empiriquement le module de Lipschitz semble prendre des valeurs entre 1 et 20 dans nos exemples
        c_w = gamma_w/L_w       
        diff_int_W = torch.inf
        diff_int_C = torch.inf
        N_step_int_W = 0
        N_step_int_C = 0         
        while diff_int_W > crit_int and N_step_int_W < max_iter_int_W:
            W,W_prev = update_W(gamma_w,nu,Rx,W,C,C0,W_prev,L_w,L_w_prev,c_w)
            diff_int_W = diff_criteria_torch(W,W_prev)
            N_step_int_W += 1
            record_tracked_vars_detailed(alpha,track_times,track_costs,track_jisi,A,Rx,W,C,detailed_jISI,detailed_costs,detailed_times,t0)
            # dW_prev = dW.clone()
            # dW = W - W_prev
            # if torch.any(dW_prev):
            #     shift_W = torch.sum(dW*dW_prev)/(np.linalg.norm(dW)*np.linalg.norm(dW_prev))
            #     shifts_W.append(shift_W)
            # if adaptative_gamma_w:
                # if shift_W < 0:
                    # gamma_w = gamma_w*gamma_w_decay
                # if shift_W > 0.99:
                #     gamma_w = gamma_w/gamma_w_decay
        while diff_int_C > crit_int and N_step_int_C < max_iter_int_C:
            C,C_prev = update_C(alpha,epsilon,Rx,W,C,C_prev,c_c,beta_c)
            diff_int_C = diff_criteria_torch(C,C_prev)
            N_step_int_C += 1
            record_tracked_vars_detailed(alpha,track_times,track_costs,track_jisi,A,Rx,W,C,detailed_jISI,detailed_costs,detailed_times,t0)
        diff_ext = record_tracked_vars(alpha,track_times,track_costs,track_jisi,track_diffs,track_schemes,A,Rx,W,C,diffs_W,diffs_C,jISI,costs,scheme,times,t0,W_old,C_old,N_step_int_W,N_step_int_C)
        N_step += 1
    return report_variables(W,C,N_step,max_iter,track_times,times,detailed_times,track_costs,costs,detailed_costs,track_jisi,jISI,detailed_jISI,track_diffs,diffs_W,diffs_C,track_schemes,scheme,track_shifts,shifts_W)

def record_tracked_vars(alpha,track_times,track_costs,track_jisi,track_diffs,track_schemes,A,Rx,W,C,diffs_W,diffs_C,jISI,costs,scheme,times,t0,W_old,C_old,N_step_int_W,N_step_int_C):
    if track_schemes:
        scheme.append([N_step_int_W,N_step_int_C])
    diff_W = diff_criteria_torch(W,W_old)
    diff_C = diff_criteria_torch(C,C_old)
    diff_ext = max(diff_W,diff_C)
    if track_diffs:
        diffs_W.append(diff_W)
        diffs_C.append(diff_C)
    if track_costs:
        costs.append(cost_iva_g_reg(W,C,Rx,alpha))
    if track_jisi:
        jISI.append(joint_isi_torch(W,A))
    if track_times:
        times.append(time()-t0)
    return diff_ext

def update_C(alpha,epsilon,Rx,W,C,C_prev,c_c,beta_c):
    C_tilde = C + beta_c*(C-C_prev)
    grad_C = grad_H_C_reg(W,C_tilde,Rx,alpha)
    C_bar = C_tilde - c_c*grad_C
    C_prev = C.clone()
    C = prox_g(C_bar,c_c,epsilon)
    return C,C_prev

def update_W(gamma_w,nu,Rx,W,C,C0,W_prev,L_w,L_w_prev,c_w):
    beta_w = (1-gamma_w)*torch.sqrt(C0*nu*(1-nu)*L_w_prev/L_w)
    W_tilde = W + beta_w*(W-W_prev)
    grad_W = grad_H_W(W_tilde,C,Rx)
    W_bar = W_tilde - c_w*grad_W
    W_prev = W.clone()
    W = prox_f(W_bar,c_w)
    return W,W_prev

def record_tracked_vars_detailed(alpha,track_times,track_costs,track_jisi,A,Rx,W,C,detailed_jISI,detailed_costs,detailed_times,t0):
    if track_costs:
        detailed_costs.append(cost_iva_g_reg(W,C,Rx,alpha))
    if track_jisi:
        detailed_jISI.append(joint_isi_torch(W,A))
    if track_times:
        detailed_times.append(time()-t0)

def init_data_param(Rx,alpha,gamma_c,gamma_w):
    K,_,N,_ = Rx.shape      
    alpha,gamma_c,gamma_w = to_float64(alpha,gamma_c,gamma_w)
    # if (not adaptative_gamma_w) and gamma_w > 1:
    #     raise('gamma_w must be in (0,1) if not adaptative')
    rho_Rx = spectral_norm_extracted_torch(Rx,K,N)
    return alpha,gamma_c,gamma_w,N,K,rho_Rx

def initialize(N,K,init_method,Winit=None,Cinit=None,seed=None):
    if Winit is not None and Cinit is not None:
        W,C = Winit.clone(),Cinit.clone()
    # elif init_method == 'Jdiag':
    #     W,C = Jdiag_init(X,N,K,Rx)
    elif init_method == 'random':
        C = make_Sigma(K,N,rank=K+10,seed=seed)
        W = make_A(K,N,seed=seed)      
    return W,C

# def Jdiag_init(X,K,Rx):
#     if K > 2:
#         # initialize with multi-set diagonalization (orthogonal solution)
#         W = _jbss_sos(X,0,'whole')
#     else:
#         W = _cca(X)
#     C = np.moveaxis(np.linalg.inv(np.einsum('nNK,KJNM,nMJ -> nKJ',W,Rx,W)),0,2)
#     return W,C

def to_float64(alpha,gamma_c,gamma_w):
    alpha = torch.tensor(alpha,dtype=torch.float64)
    gamma_w = torch.tensor(gamma_w,dtype=torch.float64)
    gamma_c = torch.tensor(gamma_c,dtype=torch.float64)
    return alpha,gamma_c,gamma_w

def report_variables(W,C,N_step,max_iter,track_times,times,detailed_times,
                     track_costs,costs,detailed_costs,track_jisi,jISI,detailed_jISI,
                     track_diffs,diffs_W,diffs_C,track_schemes,scheme,
                     track_shifts,shifts_W):
    met_limit = (N_step < max_iter)
    results = {'W': W,'C': C,'met_limit': met_limit}
    
    if track_times:
        results['times'] = torch.stack(times) if isinstance(times[0],torch.Tensor) else torch.tensor(times)
        results['detailed_times'] = torch.stack(detailed_times) if isinstance(detailed_times[0],torch.Tensor) else torch.tensor(detailed_times)
    if track_costs:
        results['costs'] = torch.stack(costs) if isinstance(costs[0],torch.Tensor) else torch.tensor(costs)
        results['detailed_costs'] = torch.stack(detailed_costs) if isinstance(detailed_costs[0],torch.Tensor) else torch.tensor(detailed_costs)
    if track_jisi:
        results['jisi'] = torch.stack(jISI) if isinstance(jISI[0],torch.Tensor) else torch.tensor(jISI)
        results['detailed_jisi'] = torch.stack(detailed_jISI) if isinstance(detailed_jISI[0],torch.Tensor) else torch.tensor(detailed_jISI)
    if track_diffs:
        results['diffs_W'] = torch.stack(diffs_W) if isinstance(diffs_W[0],torch.Tensor) else torch.tensor(diffs_W)
        results['diffs_C'] = torch.stack(diffs_C) if isinstance(diffs_C[0],torch.Tensor) else torch.tensor(diffs_C)
    if track_schemes:
        results['scheme'] = torch.tensor(scheme)
    if track_shifts:
        results['shifts_W'] = torch.stack(shifts_W) if isinstance(shifts_W[0],torch.Tensor) else torch.tensor(shifts_W)
    
    return results