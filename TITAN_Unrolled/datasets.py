from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import os
from .functions import *

class IVAGDataset(Dataset):
    def __init__(self,data_path,dimensions=(10,10000,10),dataparameters=None,size=1000,device='cpu',dtype=torch.float32):
        self.N,self.V,self.K = dimensions
        self.dataparameters = dataparameters
        self.data_path = data_path
        self.size = size
        regenerate = True
        if os.path.exists(self.data_path):
            self.data = torch.load(self.data_path,weights_only=True)
            regenerate = len(self.data) != self.size             
        if regenerate:
            print('creation of a new dataset')
            self.data = [] 
            self.num_dataparameters = len(dataparameters)
            for i in tqdm(range(self.size)):
                dataparam = self.dataparameters[i%self.num_dataparameters]
                Rx,A = generate_whitened_problem(self.V,self.K,self.N,device=device,rho_bounds=dataparam[0],lambda_=dataparam[1],dtype=dtype)
                Winit = make_A(self.K,self.N,device=device,dtype=dtype)
                Cinit = make_Sigma(self.K,self.N,rank=self.K+10,device=device,dtype=dtype)
                self.data.append((Rx,Winit,Cinit,A))
            torch.save(self.data,self.data_path) 

    def __len__(self):
        return self.size  

    def __getitem__(self,idx):
        return self.data[idx]


#====================================================================================================
# DATASETS CREATION
#====================================================================================================


def get_dataparameters(rhos,lambdas):
    dataparameters_multiparam = []
    for rho_bounds in rhos:
        for lambda_ in lambdas:
            dataparameters_multiparam.append((rho_bounds,lambda_))
    return dataparameters_multiparam

lambda_1 = 0.04
lambda_2 = 0.25
rho_bounds_1 = [0.2,0.3]
rho_bounds_2 = [0.6,0.7]
rhos = [rho_bounds_1,rho_bounds_2]
lambdas = [lambda_1,lambda_2]

dataparameters_multiparam = get_dataparameters(rhos,lambdas)
dataparameters_titles = ['Case_A','Case_B','Case_C','Case_D']

size = 100
for numcase in range(4):
    for K in [5,10,20]:
        for N in [10,20,30]:
            dimensions = (N,10000,K)
            data_path = f'Result_data/datasets/{dataparameters_titles[numcase]}/N_{N}_K_{K}/test'
            dataset = IVAGDataset(data_path=data_path,dimensions=dimensions,dataparameters=[dataparameters_multiparam[numcase]],size=size)