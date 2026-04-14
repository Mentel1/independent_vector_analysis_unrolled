import os
import numpy as np
from time import time
import inspect
from Algorithms.iva_g_torch import *
from Algorithms.algebra_toolbox_torch import *
from Algorithms.titan_iva_g_reg_torch import *
# from TITAN_Unrolled.model import *
from TITAN_Unrolled.architecture import *
from TITAN_Unrolled.datasets import *

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

    def __init__(self,color='b',name='titan',legend='TITAN-IVA-G',nu=0.5,max_iter=20000,max_iter_int_W=15,max_iter_int_C=1,crit=1e-10,epsilon=1e-12,zeta=1e-3,gamma_w=0.99,gamma_c=1,alpha=1,init_method='random',seed=None,boost=False,exactC=False,device='cuda'):
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

    def __init__(self,color='b',name='Utitan',legend='U-TITAN-IVA-G',dimensions=(30,10000,20),data_case='Case_A',archi='untied',num_layers=500,tol=1e-2,N_updates_W=10,N_updates_C=1,training_mode='local',opt_name ='SGD',gradient_processing='normalize',step_size=5,device='cuda'):
        super().__init__(name=name,legend=legend,color=color)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.dimensions = N,_,K = dimensions
        self.data_case = data_case
        self.N_updates_W = N_updates_W
        parameters_path=f'Result_data/models/{data_case}/N_{N}_K_{K}/UTitan{step_size}_{archi}_{training_mode}_{opt_name}_{gradient_processing}/parameters'
        self.model = UTitanIVAGModel(N,K,num_layers,N_updates_W=N_updates_W,N_updates_C=N_updates_C,archi=archi,load=True,parameters_path=parameters_path).to(self.device)
        self.tol = tol
        self.shorten_model()        

    def to_dict(self):
        return {'class': self.__class__.__name__, 'color': self.color,'name': self.name,'legend': self.legend,'model':self.model}
    
    @classmethod
    def from_dict(cls, config):
        """Reconstruit un algo depuis un dictionnaire"""
        return cls(**config)

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
            self.results['number_updates'][exp] = self.model.num_layers*self.N_updates_W 
            
    def select_num_layers(self):
        eval_set = IVAGDataset(name='eval',dimensions=self.dimensions,data_case=self.data_case,device=self.device)
        loader = DataLoader(eval_set,batch_size=eval_set.__len__())
        Rx,Winit,Cinit,A = next(iter(loader))
        outputs = self.model(Rx,Winit,Cinit,track_jisi=True,A=A,track_cost=False)
        L = self.model.num_layers - 1
        jisi_scores = outputs['jisi']
        crit = jisi_scores[L]*(1 + self.tol)
        jisi = jisi_scores[L]
        while jisi < crit and L > 0:
            L -= 1
            jisi = jisi_scores[L]
        return L
            
    def shorten_model(self,save=False):
        L = self.select_num_layers()
        if not self.model.tied:
            self.model.Layers = self.model.Layers[:L]
        self.model.num_layers = L
        if save:
            torch.save(self.model.state_dict(),self.parameters_path + '_shortened')
        print(f'succesfully shortened the model! Down to {L} layers!')