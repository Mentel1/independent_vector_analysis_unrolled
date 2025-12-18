import os
import numpy as np
from time import time
from algorithms.iva_g_numpy import *
from algorithms.iva_g_torch import *
from algorithms.algebra_toolbox_torch import *
from algorithms.algebra_toolbox_numpy import *
from algorithms.titan_iva_g_reg_numpy import *
from algorithms.titan_iva_g_reg_torch import *
from algorithms.titan_iva_g_reg_numpy_exact_C import *


class IvaGAlgorithms:

    def __init__(self,name,legend,color):
        self.name = name
        self.legend = legend
        self.color = color
        self.results = {}

    def solve(self,X):
        pass

    def fill_experiment(self,X,A,exp,Winit=None,Cinit=None):
        res = self.solve(X,Winit,Cinit)
        self.results['total_times'][exp] = res['times'][-1]
        W = res['W']
        A = torch.tensor(A)
        self.results['final_jisi'][exp] = joint_isi_torch(W,A)
    

    def fill_from_folder(self,output_path_individual):
        for result in ['total_times','jisi_final']:
            res_path = os.path.join(output_path_individual,self.name,result)
            self.results[result] = np.fromfile(res_path,sep=',')

class IvaG(IvaGAlgorithms):

    def __init__(self,color='b',name='IVA-G-N',legend='IVA-G-N',opt_approach='newton',max_iter=5000,crit_ext=1e-6,fast=False,down_sample=False,num_samples=10000,Jdiag_init=False):
        super().__init__(name=name,legend=legend,color=color)
        self.alternated = False
        self.opt_approach = opt_approach
        self.max_iter = max_iter
        self.crit_ext = crit_ext
        self.fast = fast
        self.down_sample = down_sample
        self.num_samples = num_samples
        self.Jdiag_init = Jdiag_init


    def _get_base_params(self):
        return {'W_diff_stop': self.crit_ext,'max_iter': self.max_iter,'down_sample':self.down_sample,'num_samples':self.num_samples,'opt_approach':self.opt_approach}
    
    def solve(self,X,Winit=None,Cinit=None,track_jisi=False,B=None,track_costs=False,track_diffs=False,track_schemes=False,track_shifts=False,track_times=True):
        self.normalize_Winit(X, Winit)
        params = self._get_base_params()
        params.update({'Winit':Winit,'A':B})
        Winit = torch.tensor(Winit)
        X = torch.from_numpy(X)
        if self.fast:
            return fast_iva_g_torch(X,**params)
        else:
            return iva_g_torch(X,**params)

    def normalize_Winit(self, X, Winit):
        _,_,K = X.shape
        for k in range(K):
            Winit[:, :, k] = np.linalg.solve(sc.linalg.sqrtm(Winit[:, :, k] @ Winit[:, :, k].T), Winit[:, :, k]) 
    

class TitanIvaG(IvaGAlgorithms):    

    def __init__(self,color,name='titan',legend='TITAN-IVA-G',nu=0.5,max_iter=20000,max_iter_int_W=15,max_iter_int_C=1,inflate=False,lambda_inflate=1e-3,down_sample=False,num_samples=10000,crit_ext=1e-10,crit_int=1e-10,epsilon=1e-12,gamma_w=0.99,gamma_c=1,alpha=1,seed=None,boost=False,exactC=False):
        super().__init__(name=name,legend=legend,color=color)
        self.crit_int = crit_int
        self.crit_ext = crit_ext
        self.max_iter = max_iter
        self.max_iter_int_W = max_iter_int_W
        self.max_iter_int_C = max_iter_int_C
        self.nu = nu
        self.alpha = alpha
        self.alternated = True
        self.epsilon = epsilon
        self.gamma_w = gamma_w
        self.gamma_c = gamma_c
        self.seed = seed
        self.boost = boost
        self.exactC = exactC
        self.inflate = inflate
        self.lambda_inflate = lambda_inflate
        self.down_sample = down_sample
        self.num_samples = num_samples

    def _get_base_params(self):
        return {'alpha': self.alpha,'gamma_w': self.gamma_w,'gamma_c': self.gamma_c,'crit_ext': self.crit_ext,'crit_int': self.crit_int,'eps': self.epsilon,'nu': self.nu,'max_iter': self.max_iter,'max_iter_int_W': self.max_iter_int_W,'max_iter_int_C': self.max_iter_int_C,'seed': self.seed,'boost':self.boost,'inflate':self.inflate,'lambda_inflate':self.lambda_inflate,'down_sample':self.down_sample,'num_samples':self.num_samples}
              
    def solve(self,X,Winit=None,Cinit=None,track_jisi=False,B=None,track_costs=False,track_diffs=False,track_schemes=False,track_shifts=False,track_times=True):
        params = self._get_base_params()
        params.update({'Winit':Winit,'Cinit':Cinit,'track_jisi':track_jisi,'B':B,'track_costs':track_costs,'track_diffs':track_diffs,'track_schemes':track_schemes,'track_shifts':track_shifts,'track_times':track_times})    
        return titan_iva_g_reg_torch(X,**params)

class UTitanIvaG(IvaGAlgorithms):
    def __init__(self,color,model_path,name='utitan',legend='U-TITAN-IVA-G',Winit=None,Cinit=None,inflate=False,lambda_inflate=1e-3,down_sample=False,num_samples=10000,seed=None,boost=False):
        super().__init__(name=name,legend=legend,color=color)
        self.alternated = True
        self.seed = seed
        self.boost = boost
        self.inflate = inflate
        self.lambda_inflate = lambda_inflate
        self.down_sample = down_sample
        self.num_samples = num_samples
        self.model_path = model_path
        self.Winit = Winit
        self.Cinit = Cinit

    def _get_base_params(self):
        return {'seed': self.seed,'boost':self.boost,'inflate':self.inflate,'lambda_inflate':self.lambda_inflate,'down_sample':self.down_sample,'num_samples':self.num_samples} 
    ## voir plus tard quels sont les paramètres à passer au réseau de neurones
              
    def solve(self,X,N,K,metaparam,metaparam_title,Winit=None,Cinit=None,track_jisi=False,B=None,track_costs=False,track_times=True):
        params = self._get_base_params()
    ## 

        
        params.update({'Winit':Winit,'Cinit':Cinit,'track_jisi':track_jisi,'B':B,'track_costs':track_costs,'track_times':track_times})    
        return titan_iva_g_reg_torch(X,**params)
    ## remplacer par l'appel du réseau de neurones