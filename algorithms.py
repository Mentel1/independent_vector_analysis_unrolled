import os
import numpy as np
from time import time
import inspect
from Algorithms.iva_g_torch import *
from Algorithms.algebra_toolbox_torch import *
from Algorithms.titan_iva_g_reg_torch import *
from TITAN_Unrolled.model import *
from TITAN_Unrolled.architecture import *

class IvaGAlgorithms:

    def __init__(self,name,legend,color):
        self.name = name
        self.legend = legend
        self.color = color
        self.results = {}
        # A ajouter : la possibilité d'avoir les features vectorielles qui permettent de tracer des courbes (cost, jisi...)
    
    def to_dict(self):
        """Convertit l'algo en dictionnaire sérialisable"""
        return {}
    
    @classmethod
    def from_dict(cls, config):
        config['color'] = tuple(config['color'])
        config = config.copy()
        class_name = config.pop('class')  # Ex: "TitanIvaG"
        current_module = sys.modules[__name__]
        actual_cls = getattr(current_module, class_name)
        return actual_cls(**config)

    def _fill_experiment(self,exp,Rx,A,Winit=None,Cinit=None,count_updates=False,track_diffs=False):
        raise NotImplementedError

    def fill_experiment(self,exp,Rx,A,Winit=None,Cinit=None,count_updates=False,track_diffs=False):
        Rx = Rx.to(self.device)
        A = A.to(self.device)
        Winit = Winit.to(self.device)
        Cinit = Cinit.to(self.device)
        # Délègue à l'implémentation spécifique de chaque sous-classe
        return self._fill_experiment(exp,Rx,A,Winit=Winit,Cinit=Cinit,count_updates=count_updates,track_diffs=track_diffs)

    def fill_from_folder(self,output_path_individual):
        for result in ['total_times','final_jisi','number_updates']:
            res_path = os.path.join(output_path_individual,self.name + '_' + result)
            if os.path.exists(res_path):
                self.results[result] = np.fromfile(res_path,sep=',')

class IvaG(IvaGAlgorithms):

    def __init__(self,color='b',name='IVA-G-N',legend='IVA-G-N',opt_approach='newton',max_iter=5000,W_diff_stop=1e-6,fast=False,jdiag_initW=False):
        super().__init__(name=name,legend=legend,color=color)
        self.opt_approach = opt_approach
        self.W_diff_stop = W_diff_stop
        self.fast = fast
        self.jdiag_initW = jdiag_initW
        self.max_iter = max_iter

    def to_dict(self):
        return {'class': self.__class__.__name__,'color': self.color,'name': self.name,'legend': self.legend,'opt_approach': self.opt_approach,'W_diff_stop': self.W_diff_stop,'fast': self.fast,'jdiag_initW': self.jdiag_initW,'max_iter': self.max_iter}
        
    def _get_base_params(self):
        return {'W_diff_stop': self.W_diff_stop,'max_iter': self.max_iter,'opt_approach':self.opt_approach,'jdiag_initW': self.jdiag_initW}
    
    def _fill_experiment(self,exp,Rx,A,**kwargs):
        # Use kwargs to compute the iva_g arguments
        params = self._get_base_params()
        params['return_W_change'] = kwargs.get('track_diffs',False)
        Winit = kwargs['Winit']
        self.normalize_Winit_(Winit)
        params['W_init'] = Winit
        # apply iva_g
        res = iva_g_torch(Rx,**params)
        # fill the dictionary "results" with the appropriate data
        self.results['total_times'][exp] = res['times'][-1]
        self.results['final_jisi'][exp] = joint_isi_torch(res['W'],A)
        if kwargs.get('count_updates',False):
            self.results['number_updates'][exp] = len(res['times'])
        if kwargs.get('track_diffs',False):
            self.results['diffs_W'] = res['W_change']

    def normalize_Winit_(self,Winit):
        _,_,K = Winit.shape
        for k in range(K):
            Winit[:, :, k] = np.linalg.solve(sc.linalg.sqrtm(Winit[:, :, k] @ Winit[:, :, k].T), Winit[:, :, k]) 
    

class TitanIvaG(IvaGAlgorithms):    

    def __init__(self,color='b',name='titan',legend='TITAN-IVA-G',nu=0.5,max_iter=20000,max_iter_int_W=15,max_iter_int_C=1,crit=1e-10,epsilon=1e-12,zeta=1e-3,gamma_w=0.99,gamma_c=1,alpha=1,init_method='random',seed=None,boost=False,exactC=False,device='cuda:0'):
        super().__init__(name=name,legend=legend,color=color)
        self.crit_int = crit # remettre les deux arguments séparés après avoir fini le manuscrit
        self.crit_ext = crit
        self.max_iter = max_iter
        self.max_iter_int_W = max_iter_int_W
        self.max_iter_int_C = max_iter_int_C
        self.nu = nu
        self.alpha = alpha
        self.epsilon = epsilon
        self.zeta = zeta
        self.gamma_w = gamma_w
        self.gamma_c = gamma_c
        self.init_method = init_method
        self.seed = seed
        self.boost = boost
        self.exactC = exactC
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

    def to_dict(self):
        return {'class': self.__class__.__name__, 'color': self.color,'name': self.name,'legend': self.legend,'crit_int': self.crit_int,'crit_ext': self.crit_ext,'max_iter_int_W': self.max_iter_int_W,'max_iter_int_C': self.max_iter_int_C,'max_iter': self.max_iter,'nu': self.nu,'alpha': self.alpha,'epsilon': self.epsilon,'zeta': self.zeta, 'gamma_w': self.gamma_w,'gamma_c': self.gamma_c,'init_method': self.init_method,'seed': self.seed,'boost': self.boost,'exactC': self.exactC}
    
    @classmethod
    def from_dict(cls, config):
        """Reconstruit un algo depuis un dictionnaire"""
        return cls(**config)
    
    def _get_base_params(self):
        return {'alpha': self.alpha,'gamma_w': self.gamma_w,'gamma_c': self.gamma_c,'crit_ext': self.crit_ext,'crit_int': self.crit_int,'epsilon': self.epsilon,'zeta' :self.zeta,'nu': self.nu,'max_iter': self.max_iter,'max_iter_int_W': self.max_iter_int_W,'max_iter_int_C': self.max_iter_int_C,'seed': self.seed,'boost': self.boost}
              
    def _fill_experiment(self,exp,Rx,A,**kwargs):
        # prepare the parameters
        params = self._get_base_params()
        params['track_schemes'] = kwargs.get('count_updates',False)
        params.update({'Winit':kwargs['Winit'],'Cinit':kwargs['Cinit'],'track_diffs':kwargs['track_diffs']})
        # apply titan_iva_g_reg
        res = titan_iva_g_reg_torch(Rx,**params)
        # fill the results
        self.results['total_times'][exp] = res['times'][-1]
        self.results['final_jisi'][exp] = joint_isi_torch(res['W'],A)
        if kwargs.get('count_updates',False):
            self.results['number_updates'][exp] = torch.sum(res['scheme'][:,0])
        if kwargs.get('track_diffs',False):
            self.results['diffs_W'] = res['diffs_W']
            self.results['diffs_C'] = res['diffs_C']
    
class UTitanIvaG(IvaGAlgorithms):    

    def __init__(self,color='b',name='Utitan',legend='U-TITAN-IVA-G',seed=None,dimensions=(30,10000,20),metaparameters_title='Case_A',archi='untied',training_mode='local',optimizer=torch.optim.SGD,gradient_processing='normalize',batch_size=100,num_layers=500,N_updates_W=10,step_size=5,device='cuda:0'):
        super().__init__(name=name,legend=legend,color=color)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.pipeline = UTitan(model_name='UTitan'+ str(step_size),archi=archi,training_mode='local',dimensions=dimensions,metaparameters_title=metaparameters_title,optimizer=optimizer,gradient_processing=gradient_processing,step_size=step_size,num_layers=num_layers,batch_size=batch_size,N_updates_W=N_updates_W,load=True)
        self.model = self.pipeline.model.to('cuda:0')
        

    def to_dict(self):
        return {'class': self.__class__.__name__, 'color': self.color,'name': self.name,'legend': self.legend,'model':self.model}
    
    @classmethod
    def from_dict(cls, config):
        """Reconstruit un algo depuis un dictionnaire"""
        return cls(**config)
    
    def _get_base_params(self):
        return {'seed':self.seed}
              
    def _fill_experiment(self,exp,Rx,A,**kwargs):
        Rx_batch = Rx.unsqueeze(0)
        Winit_batch = kwargs['Winit'].unsqueeze(0)
        Cinit_batch = kwargs['Cinit'].unsqueeze(0)
        
        with torch.no_grad():
            torch.cuda.synchronize()
            start = time()
            res = self.model(Rx_batch, Winit_batch, Cinit_batch)
            torch.cuda.synchronize()
            end = time()
        
        self.results['total_times'][exp] = end - start
        self.results['final_jisi'][exp] = joint_isi_torch(res['W'].squeeze(),A)
        if kwargs.get('count_updates',False):
            self.results['number_updates'][exp] = self.model.num_layers*10 #Hardcodé pour le moment mais le 10 doit être remplacé par N_updates_W lors de la prochaine refonte.
            