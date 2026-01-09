import torch
import math
from tqdm import tqdm
from datetime import datetime
from architecture import *
from data import *
from tools import *
from functions import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import sys
 
class IVAGDataset(Dataset):
    def __init__(self,data_path,dimensions=(10,10000,10),metaparameters=None,size=1000,device='cpu',dtype=torch.float32):
        self.N,self.T,self.K = dimensions
        self.metaparameters = metaparameters
        self.data_path = data_path
        self.size = size
        regenerate = True
        if os.path.exists(self.data_path):
            self.data = torch.load(self.data_path,weights_only=True)
            regenerate = self.__len__() != self.size             
        if regenerate:
            print('creation of a new dataset')
            self.data = [] 
            self.num_metaparameters = len(metaparameters)
            for i in tqdm(range(self.size)):
                metaparam = self.metaparameters[i%self.num_metaparameters]
                Rx,A = generate_whitened_problem(self.T,self.K,self.N,device=device,rho_bounds=metaparam[0],lambda_=metaparam[1],dtype=dtype)
                Winit = make_A(self.K,self.N,device=device,dtype=dtype)
                Cinit = make_Sigma(self.K,self.N,rank=self.K+10,device=device,dtype=dtype)
                self.data.append((Rx,Winit,Cinit,A))
            torch.save(self.data,self.data_path) 

    def __len__(self):
        return self.size  

    def __getitem__(self,idx):
        return self.data[idx]


class UTitan:
    def __init__(self,model_name='UTitan',train_file='training_data',test_file='testing_data',parameters_file='parameters',archi='untied',training_mode='end-to-end',dimensions=(10,10000,10),metaparameters=None,metaparameters_title='Multi_case',train_size=1000,test_size=200,lr=0.1,N_updates_W=15,N_updates_C=1,num_epochs=20,loss_train=ISI_loss(),loss_test=ISI_loss(),batch_size=64,num_layers=100,epsilon=1e-12,custom=False,load=True):
        # Path information
        self.model_name = model_name
        now = datetime.now()
        self.date = now.strftime("%Y-%m-%d_%H-%M")
        self.dimensions = dimensions
        self.N,self.T,self.K = dimensions
        self.metaparameters_title=metaparameters_title
        # Dataset information
        self.metaparameters = metaparameters
        self.train_size = train_size
        self.test_size = test_size
        self.train_path = f'Result_data/{self.metaparameters_title}/N_{self.N}_K_{self.K}/{train_file}'
        self.test_path = f'Result_data/{self.metaparameters_title}/N_{self.N}_K_{self.K}/{test_file}'
        # Model information
        self.dtype = torch.cuda.FloatTensor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_layers = num_layers
        self.model = UTitanIVAGModel(N_updates_W,N_updates_C,num_layers=num_layers,epsilon=epsilon,archi=archi,custom=custom,N=self.N,K=self.K).to(self.device)
        self.model_path = f'Result_data/{self.metaparameters_title}/N_{self.N}_K_{self.K}/{self.model_name}_{archi}_{training_mode}'
        os.makedirs(self.model_path,exist_ok=True)
        self.parameters_path = os.path.join(self.model_path,parameters_file)
        if os.path.exists(self.parameters_path) & load:
           self.model.load_state_dict(torch.load(self.parameters_path,weights_only=True))
        self.train_loss_path = os.path.join(self.model_path,'train_loss')
        self.test_loss_path = os.path.join(self.model_path,'test_loss')
        
        # training information
        self.training_mode = training_mode #'end-to-end' or 'greedy' or 'group_of_layers'
        self.greedy = self.training_mode in ['greedy','group_of_layers']
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        param_groups = []
        if archi == 'tied':
            param_groups.append({'params' : self.model.Layer.parameters(),'lr': self.lr})
        else:
            for i, layer in enumerate(self.model.Layers):
                layer_lr = self.lr #* (i+1)       
                param_groups.append({'params': layer.parameters(),'lr': layer_lr})
        self.optimizer = torch.optim.Adam(param_groups, weight_decay=0)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,mode='min',factor=0.5,patience=0,min_lr=1e-2,threshold=1e-6) 
        self.loss_train = loss_train
        self.loss_test = loss_test
        
    def train(self):
        # load or create datasets and data loaders
        self.training_set = IVAGDataset(data_path=self.train_path,dimensions=self.dimensions,metaparameters=self.metaparameters,size=self.train_size,device=self.device)
        self.testing_set = IVAGDataset(data_path=self.test_path,dimensions=self.dimensions,metaparameters=self.metaparameters,size=self.test_size,device=self.device)
        self.train_loader = DataLoader(self.training_set,batch_size=self.batch_size,shuffle=True)
        self.test_loader = DataLoader(self.testing_set,batch_size=self.batch_size,shuffle=True)
        self.num_batches = math.ceil(self.training_set.size/self.batch_size)
        self.model.train()
        jisi_train = torch.zeros((self.num_epochs,self.num_batches))
        jisi_eval = torch.zeros((self.num_epochs,self.num_batches))
        min_eval = float('inf')
        # trains the whole network
        print(f'=================== begin {self.training_mode} training ===================')
        # initialize tracking variables
        for epoch in range(self.num_epochs):
            for batch,(Rxs,Winits,Cinits,As) in enumerate(self.train_loader):
                if self.training_mode == 'group_of_layers':
                    active_layers = epoch*(self.num_layers//self.num_epochs),(epoch+1)*(self.num_layers//self.num_epochs)
                elif self.training_mode == 'one_by_one':
                    active_layers = (epoch,epoch)
                else:
                    active_layers = (0,self.num_layers)
                Ws,Cs,store_W,store_C = self.model(Rxs,Winits,Cinits,active_layers=active_layers)
                outputs = {'W':Ws,'C':Cs,'Rx':Rxs,'A':As,'store_W':store_W,'store_C':store_C}
                loss_train = self.loss_train(outputs,greedy=self.greedy) #,emphasis=()
                jisi_train[epoch,batch] = loss_train.item()/self.train_size
                sys.stdout.write(f'\r Epoch {epoch+1}/{self.num_epochs}, batch {batch+1}/{self.num_batches}, loss: {loss_train.item():.4f} \n')
                # sets the gradients to zero, performs a backward pass, and updates the weights.
                self.optimizer.zero_grad()
                loss_train.backward()
                trajectory = self.compute_trajectory(loss=ISI_loss(),save=False)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                for (Rxs,Winits,Cinits,As) in self.test_loader:
                    with torch.no_grad():
                        Ws,Cs,store_W,store_C = self.model(Rxs,Winits,Cinits,active_layers=active_layers)
                        outputs = {'W':Ws,'C':Cs,'Rx':Rxs,'A':As,'store_W':store_W,'store_C':store_C}
                        jisi_eval[epoch,batch] += self.loss_test(outputs).item()/self.test_size
                self.scheduler.step(jisi_eval[epoch,batch])
                if jisi_eval[epoch,batch].item() < min_eval:
                    min_eval = jisi_eval[epoch,batch].item()
                    torch.save(self.model.state_dict,self.parameters_path)
                #     best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                # else:
                #     with torch.no_grad():
                #         for k, v in self.model.state_dict().items():
                #             v.copy_(best_state[k])
                print(f'validation loss after epoch {epoch+1} and batch {batch+1} is {jisi_eval[epoch,batch].item()}')     
        torch.save(jisi_train,self.train_loss_path)
        torch.save(jisi_eval,self.test_loss_path)
        
    def compute_trajectory(self,save=True,loss=ISI_loss(),verbose=0):        
        trajectory = torch.zeros(self.num_layers,device=self.device)
        self.testing_set = IVAGDataset(data_path=self.test_path,dimensions=self.dimensions,metaparameters=self.metaparameters,size=self.test_size,device=self.device)
        self.test_loader = DataLoader(self.testing_set,batch_size=self.batch_size,shuffle=True)  
        for batch,(Rxs,Winits,Cinits,As) in enumerate(self.test_loader):
            with torch.no_grad():
                Ws,Cs,store_W,store_C = self.model(Rxs,Winits,Cinits)
                for i in range(self.num_layers):
                    W = store_W[i,:,:,:,:]
                    C = store_C[i,:,:,:,:]
                    outputs = {'W':W,'C':C,'Rx':Rxs,'A':As,'store_W':store_W,'store_C':store_C}
                    trajectory[i] = trajectory[i] + loss(outputs)/self.test_size
                    if batch == len(self.test_loader) - 1 and verbose >= 1:
                        print(f"\n--- Layer {i} ---")
                        print(f"Output score: {trajectory[i]}")
                        for name, param in [
                            ("alpha", self.model.Layers[i].alpha),
                            # ("beta_w", self.model.Layers[i].beta_w),
                            # ("beta_c", self.model.Layers[i].beta_c),
                            ("gamma_w", self.model.Layers[i].gamma_w),
                            ("gamma_c", self.model.Layers[i].gamma_c)]:
                            if name == 'alpha':
                                value = self.model.Layers[i].soft(param.data)
                            else:
                                value = 0.3 + 5*(self.model.Layers[i].tanh(param.data)+1)
                            if param.grad == None:
                                gradient = 'not computed'
                            else:
                                gradient = param.grad.item()
                            print(f"{name}:  Value = {value.item()}, Gradient = {gradient}")
        if save:
            self.trajectory_path = os.path.join(self.model_path,'trajectory')
            torch.save(trajectory,self.trajectory_path)
        return trajectory

  