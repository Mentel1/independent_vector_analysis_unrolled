import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from time import time
from tqdm import tqdm
from TITAN_Unrolled.data import *
from TITAN_Unrolled.dataparam_dict import *
import scipy.io
import torch

from algorithms import *

class ComparisonExperimentIvaG:
#On classe les résultats et les graphes dans une arborescence de 2 niveaux : un premier niveau de data-paramètres qui dépendent du mode d'expérience (donc un sous-dossier par combinaison de MP) puis un second niveau de paramètres commun (en l'occurrence K et N), c'est la que sont les graphes de comparaison.
#Si on veut faire varier d'autres paramètres au niveau des algos, on définit plusieurs algorithmes séparés ! 

# L'idée de cette classe est de créer un objet "expérience" qui est déterminé par son nom (lié au mode de l'expérience, mais pas que, à voir au cas par cas), par la date à laquelle elle est lancée, et qui contient/fabrique les résultats sous forme de données dans les algos qu'elle implique ou dans des dossiers qui peuvent ou pas contenir des graphes. On veut pouvoir recréer un objet expérience à partir d'un dossier pour retravailler les données calculées et les présenter différemment par exemple
      
    def __init__(self,name,algos,dataparams_titles,common_params,data_corruption={},mode='multiparam',date=None,V=10000,N_exp=100,updates=False):
        self.algos = algos
        self.N_exp = N_exp
        self.mode = mode
        self.dataparams_titles = dataparams_titles
        self.common_params = common_params
        self.data_corruption = data_corruption
        self.name = name
        if date:
            self.date = date
            self.exists_setup = True
        else:
            now = datetime.now()
            self.date = now.strftime("%Y-%m-%d_%H-%M")
            self.exists_setup = False
        self.output_folder = f'Result_data/experiments/{self.date}_{self.name}'
        self.V = V
        self.updates = updates
        self.setup = {}
    
    
    def to_dict(self):
        algo_names = [algo.name for algo in self.algos]
        return {'N_exp':self.N_exp,'name':self.name,'common_params':self.common_params,'dataparams_titles':self.dataparams_titles,'mode':self.mode,'V':self.V,'date':self.date,'table_fontsize':self.table_fontsize,'median':self.median,'std':self.std,'legend':self.legend,'title_fontsize':self.title_fontsize,'legend_fontsize':self.legend_fontsize,'algo_names':algo_names}
        
        
    def save(self):
        config = self.to_dict()
        algo_folder = self.output_folder + '/algos'
        os.makedirs(algo_folder,exist_ok=True)
        filepath = self.output_folder + '/config.json' 
        with open(filepath,'w') as f:
            json.dump(config, f, indent=2)      
        for algo in self.algos:
            algo_config = algo.to_dict()
            algo_path = algo_folder + '/' + algo.name
            with open(algo_path, 'w') as f:
                json.dump(algo_config, f, indent=2)
        
    
    @classmethod
    def from_folder(cls,folderpath):
        filepath = f'Result_data/experiments/{folderpath}/config.json'
        with open(filepath, 'r') as f:
            config = json.load(f)
        algos = []
        algo_names = config.pop('algo_names')
        for algo_name in algo_names:
            algopath = f'Result_data/experiments/{folderpath}/algos/{algo_name}'
            with open(algopath, 'r') as f:
                algo_config = json.load(f)
            algo = IvaGAlgorithms.from_dict(algo_config)
            algos.append(algo)
        config['algos'] = algos
        return ComparisonExperimentIvaG(**config)
    
    
    def get_results_from_folder(self,param_path):    
        output_path_results = f'{self.output_folder}/results/{param_path}'
        for algo in self.algos:
            algo.fill_from_folder(output_path_results)
    
    def get_data_from_folder(self,param_path):
        output_path_data = f'{self.output_folder}/data/{param_path}'
        for setup_var in self.setup.keys():
            var_path = os.path.join(output_path_data,setup_var)
            self.setup[setup_var].fromfile(var_path,sep=',')
            
    def set_algos(self,new_algos):
        self.algos = new_algos

                                           
    def store_in_folder(self,param_path,subfolder='data'):
        full_path = f'{self.output_folder}/{subfolder}/{param_path}'
        os.makedirs(full_path,exist_ok=True)
        if subfolder == 'data':
            for setup_var in self.setup.keys():
                var_path = os.path.join(full_path,setup_var)
                self.setup[setup_var].tofile(var_path,sep=',')
        else:     
            for algo in self.algos:
                for res_var in algo.results.keys():
                    res_path = os.path.join(full_path,algo.name + '_' + res_var)
                    algo.results[res_var].tofile(res_path,sep=',')      
                   
    def compute_multi_runs(self):
        Ks,Ns = self.common_params      
        for data_case in self.dataparams_titles:
            dataparam_values = dataparam_dict[data_case]
            for K in Ks:
                for N in Ns:
                    # remplacer dataparam ici par un autre dictionnaire data_corruption + ajouter ce dictionnaire comme argument de classe + cascader ces changements vers create_data + en mettre des valeurs par défaut dans dataparam_dict + fusionner ce fichier avec data + revoir où va quel fichier dans le projet + terminer la séparation de cette classe avec une classe data et une classe reporting.
                    noise_levels,num_samples = self.data_corruption.get('noise_levels',[0]),self.data_corruption.get('num_samples',[self.V])
                    param_path = f'{data_case}/N_{N}_K_{K}'
                    if self.exists_setup:
                        self.get_data_from_folder(param_path)
                    else:
                        dataset_path = f'Result_data/datasets/{data_case}/N_{N}_K_{K}/test'
                        self.create_data(dataparam_values,K,N)
                    param_path_extended = param_path
                    for num_sample_idx,num_sample in enumerate(num_samples):
                        for noise_level_idx,noise_level in enumerate(noise_levels):
                            self._init_results()  
                            if len(num_samples)*len(noise_levels) > 1:
                                param_path_extended = f'{param_path}/num_sample={num_sample}_noise_level={noise_level}'
                            if not self._results_exist(param_path_extended):
                                config_sentence = f'Data configuration is {data_case}, K = {K}, N = {N}, {num_sample} samples and noise = {noise_level}'
                                self.compute_runs(num_sample_idx,noise_level_idx,param_path_extended,config_sentence)
                                
    def compute_runs(self,num_sample_idx,noise_level_idx,param_path_extended,config_sentence):
        for exp in tqdm(range(self.N_exp)):
            for algo in self.algos:
                algo.fill_experiment(exp,self.setup['Rxs'][exp, num_sample_idx, noise_level_idx],self.setup['As'][exp, num_sample_idx],Winit=self.setup['Winits'][exp],Cinit=self.setup['Cinits'][exp],count_updates=self.updates)
                print(f"{config_sentence} with algo {algo.name} : {algo.results['final_jisi'][exp]},{algo.results['total_times'][exp]}")
                if self.updates:
                    print(f"Number of updates: {algo.results['number_updates'][exp]}") 
        self.store_in_folder(param_path_extended,'res')
        
    def _init_results(self):
        for algo in self.algos:
            algo.results['total_times'] = np.zeros(self.N_exp)
            algo.results['final_jisi'] = np.zeros(self.N_exp)
            if self.updates:
                algo.results['number_updates'] = np.zeros(self.N_exp)
            
    def _results_exist(self,param_path):
        full_path = f'{self.output_folder}/res/{param_path}'
        return os.path.exists(full_path) and len(os.listdir(full_path)) > 0

    def create_data(self,dataparam,K,N,dataset_path=None):
        
        # Getting the parameters of the data generation
        epsilon,rho_bounds,lambda_,rank = dataparam.get('epsilon',1),dataparam.get('rho_bounds',[0.4,0.6]),dataparam.get('lambda',0.1),dataparam.get('rank',K+10)
        noise_levels,num_samples = self.data_corruption.get('noise_levels',[0]),self.data_corruption.get('num_samples',[self.V])
        
        # Initialization of the setup
        self.setup['Rxs'] = torch.zeros((self.N_exp,len(num_samples),len(noise_levels),K,K,N,N))
        self.setup['As'] = torch.zeros((self.N_exp,len(num_samples),N,N,K))
        self.setup['Winits'] = torch.zeros((self.N_exp,N,N,K))
        self.setup['Cinits'] = torch.zeros((self.N_exp,K,K,N))
        
        use_datasets = (dataset_path is not None and os.path.exists(dataset_path) and len(num_samples) == 1 and len(noise_levels) == 1)
        
        if use_datasets:
            data = torch.load(dataset_path,weights_only=True)
            if len(data) >= self.N_exp:
                print(f'Loading data from {dataset_path}')
                self._fill_setup_from_dataset(data)
                return
            else:
                print(f'Dataset has only {len(data)} examples, need {self.N_exp} — falling back to synthetic generation')
        
        for exp in range(self.N_exp):
            A = make_A(K,N)
            Sigma = make_Sigma(K,N,rank=rank,epsilon=epsilon,rho_bounds=rho_bounds,lambda_=lambda_,seed=None,normalize=False)
            S = make_S(Sigma,self.V)
            X = make_X(S,A)
            for num_sample_idx,num_sample in enumerate(num_samples):
                X_alt = X[:,:num_sample,:]
                X_,U = whiten_data(X_alt)
                A_ = torch.einsum('nNk,Nvk->nvk',U,A)    
                Rx_ = torch.einsum('NVK,MVJ->KJNM',X_,X_)/num_sample
                for noise_level_idx,noise_level in enumerate(noise_levels):
                    for k in range(K):
                        Rx_[k,k,:,:] += noise_level*np.eye(N)
                    self.setup['Rxs'][exp,num_sample_idx,noise_level_idx] = Rx_
                    self.setup['As'][exp,num_sample_idx] = A_
            self.setup['Winits'][exp] = make_A(K,N)
            self.setup['Cinits'][exp] = make_Sigma(K,N,rank=K+10)
            
    def _fill_setup_from_dataset(data):
        for exp in range(self.N_exp):
            Rx,Winit,Cinit,A = data[exp]
            # Pas de variation de num_samples/noise_levels depuis le dataset
            # on remplit uniquement l'index [0,0] 
            self.setup['Rxs'][exp,0,0] = Rx
            self.setup['As'][exp,0] = A
            self.setup['Winits'][exp] = Winit
            self.setup['Cinits'][exp] = Cinit
            
            
    def compute_empirical_convergence(self,a,K,N,res_vars=['jisi','costs','times'],detailed=True,exp=0):
        track_params = {}
        for var in res_vars:
            track_params['track_' + var] = var in res_vars
            if detailed and not 'detailed' in var:
                res_vars.append('detailed_' + var)
        if 'diffs' in res_vars:
            res_vars += ['diffs_W','diffs_C']
        if self.exists_setup:
            self.get_setup_from_folder(a,K,N)
        else:
            self.create_setup(self.data_params[a],K,N)
        for algo in self.algos:
            res = algo.solve(self.setup['Datasets_cov'][exp,:,:,:,:],Winit=self.setup['Winits'][exp,:,:,:],Cinit=self.setup['Cinits'][exp,:,:,:],A=self.setup['Mixings'][exp,:,:,:],**track_params)
            for res_var in res_vars:
                if res_var in res.keys():
                    algo.results[res_var] = res[res_var]
        self.store_in_folder(a,K,N)

