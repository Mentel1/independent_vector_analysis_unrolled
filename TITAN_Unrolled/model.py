import torch
import math
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
    def __init__(self,data_path,dimensions=(10,10000,10),metaparameters=None,size=1000,device='cpu'):
        self.N,self.T,self.K = dimensions
        self.metaparameters = metaparameters
        self.data_path = data_path
        self.size = size
        regenerate = True
        if os.path.exists(self.data_path):
            self.data = torch.load(self.data_path)
            regenerate = self.__len__() != self.size             
        if regenerate:
            self.data = [] 
            self.num_metaparameters = len(metaparameters)
            for i in range(self.size):
                metaparam = self.metaparameters[i%self.num_metaparameters]
                Rx,A = generate_whitened_problem(self.T,self.K,self.N,device=device,rho_bounds=metaparam[0],lambda_=metaparam[1])
                Winit = make_A(self.K,self.N,device=device)
                Cinit = make_Sigma(self.K,self.N,rank=self.K+10,device=device)
                self.data.append((Rx,Winit,Cinit,A))
            torch.save(self.data,self.data_path) 

    def __len__(self):
        return self.size  

    def __getitem__(self,idx):
        return self.data[idx]


class UTitan:
    def __init__(self,model_name='UTitan',train_file='training_data',test_file='testing_data',parameters_file='parameters',mode='end-to-end',dimensions=(10,10000,10),metaparameters=None,metaparameters_title='Multi_case',train_size=1000,test_size=200,lr=0.1,N_updates_W=15,N_updates_C=1,num_epochs=20,loss=ISI_loss(),batch_size=64,num_layers=100,epsilon=1e-12):
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
        self.model = UTitanIVAGModel(N_updates_W,N_updates_C,num_layers=num_layers,epsilon=epsilon).to(self.device)
        self.model_path = f'Result_data/{self.metaparameters_title}/N_{self.N}_K_{self.K}/{self.model_name}_{self.date}'
        os.makedirs(self.model_path,exist_ok=True)
        self.parameters_path = os.path.join(self.model_path,parameters_file)
        if os.path.exists(self.parameters_path):
           self.model.load_state_dict(torch.load(self.parameters_path,weights_only=True))
        self.train_loss_path = os.path.join(self.model_path,'train_loss')
        self.test_loss_path = os.path.join(self.model_path,'test_loss')   
        # training information
        self.mode = mode #'first_layer' or 'greedy' or 'last_layers_lpp' or 'test'
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size 
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.lr,weight_decay=1e-3)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,mode='min',factor=0.5,patience=2,threshold=1e-6) 
        self.loss = loss
        
    def train(self):
        # load or create datasets and data loaders
        self.training_set = IVAGDataset(data_path=self.train_path,dimensions=self.dimensions,metaparameters=self.metaparameters,size=self.train_size,device=self.device)
        self.testing_set = IVAGDataset(data_path=self.test_path,dimensions=self.dimensions,metaparameters=self.metaparameters,size=self.test_size,device=self.device)
        self.train_loader = DataLoader(self.training_set,batch_size=self.batch_size,shuffle=True)
        self.test_loader = DataLoader(self.testing_set,batch_size=self.batch_size,shuffle=True)
        self.num_batches = math.ceil(self.training_set.size/self.batch_size)
        self.model.train()
        if self.mode == 'end-to-end':
            # trains the whole network
            print('=================== End-to-end training ===================')
            # initialize tracking variables
            jisi_train = torch.zeros(self.num_epochs)
            jisi_eval = torch.zeros(self.num_epochs)
            min_eval = float('inf')
            for epoch in range(self.num_epochs):
                for batch,(Rxs,Winits,Cinits,As) in enumerate(self.train_loader):
                    Ws,_ = self.model(Rxs,Winits,Cinits)
                    loss = self.loss(Ws,As)
                    jisi_train[epoch] += loss.item()/self.train_size
                    sys.stdout.write(f'\r Epoch {epoch+1}/{self.num_epochs}, batch {batch+1}/{self.num_batches}, loss: {loss:.4f} \n')
                    # sets the gradients to zero, performs a backward pass, and updates the weights.
                    self.optimizer.zero_grad()
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                for (Rxs,Winits,Cinits,As) in self.test_loader:
                    with torch.no_grad():
                        Ws,_ = self.model(Rxs,Winits,Cinits)
                        jisi_eval[epoch] += self.loss(Ws,As).item()/self.test_size
                if jisi_eval[epoch].item() < min_eval:
                    min_eval = jisi_eval[epoch].item()
                    torch.save(self.model.state_dict,self.parameters_path)
                print(f'validation loss at iteration {epoch} is {jisi_eval[epoch].item()}')
                self.scheduler.step(jisi_eval[epoch])        
        torch.save(jisi_train,self.train_loss_path)
        torch.save(jisi_eval,self.test_loss_path)

  