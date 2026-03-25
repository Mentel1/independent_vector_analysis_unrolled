import torch
import math
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from .architecture import *
from .data import *
from .tools import *
from .functions import *
from .datasets import *
from torch.utils.tensorboard import SummaryWriter
import os
import sys


class UTitan:
    def __init__(self,model_name='UTitan',archi='untied',training_mode='end-to-end',dimensions=(10,10000,10),dataparameters=None,dataparameters_title='Multi_case',train_size=1000,eval_size=200,optimizer=torch.optim.SGD,lr=1,weight_decay=0,gradient_processing='normalize',scheduler_mode='StepLR',step_size=3,gamma=0.9,patience=3,factor_lr=0.5,min_lr=0.01,N_updates_W=15,N_updates_C=1,num_epochs=20,loss_train=IVA_loss(),loss_eval=ISI_loss(),batch_size=64,num_layers=100,epsilon=1e-12,custom=False,load=True):
        
        # Dataset information
        self.date = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.dimensions = dimensions
        self.N,self.V,self.K = dimensions
        self.dataparameters_title = dataparameters_title
        self.dataparameters = dataparameters
        self.train_size = train_size
        self.eval_size = eval_size
        self.dataset_path = f'Result_data/datasets/{self.dataparameters_title}/N_{self.N}_K_{self.K}'
        os.makedirs(self.dataset_path,exist_ok=True)
        self.train_set_path = f'{self.dataset_path}/train'
        self.eval_set_path = f'{self.dataset_path}/eval'
        
        # Model architecture information
        self.model_name = model_name
        self.dtype = torch.cuda.FloatTensor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_layers = num_layers
        self.archi = archi
        self.N_updates_W = N_updates_W
        self.N_updates_C = N_updates_C
        self.epsilon = epsilon
        self.model = UTitanIVAGModel(N_updates_W,N_updates_C,num_layers=num_layers,epsilon=epsilon,archi=archi,custom=custom,N=self.N,K=self.K).to(self.device)
        layer = self.model.Layer if self.model.tied else self.model.Layers[0] 
        self.param_names = [name for name,_ in layer.named_parameters()]
        self.num_param = len(self.param_names)
        
        # training information
        self.training_mode = training_mode # 'end-to-end' or 'greedy' or 'group_of_layers' or 'local' or 'one_by_one'
        if optimizer == torch.optim.Adam:
            opt_name = 'Adam'
        elif optimizer == torch.optim.SGD:
            opt_name = 'SGD'
        opt_name = opt_name + '_' + gradient_processing
        self.scheduler_mode = scheduler_mode
        self.gradient_processing = gradient_processing
        self.is_greedy = self.training_mode in ['greedy','group_of_layers']
        self.lr = lr
        self.factor_lr = factor_lr
        self.min_lr = min_lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        if self.training_mode == 'local' and not self.model.tied:
            self.optimizers = []
            self.schedulers = []
            for i, layer in enumerate(self.model.Layers):
                self.optimizers.append(optimizer(self.model.Layers[i].parameters(),lr=self.lr,weight_decay=weight_decay))
                if scheduler_mode == 'ReduceLROnPlateau':
                    self.schedulers.append(torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizers[i],mode='min',factor=factor_lr,patience=patience,min_lr=min_lr,threshold=1e-6))
                elif scheduler_mode == 'StepLR':
                    self.schedulers.append(torch.optim.lr_scheduler.StepLR(self.optimizers[i],step_size=step_size,gamma=gamma))
        else:
            self.optimizer = optimizer(self.model.parameters(),lr=self.lr,weight_decay=weight_decay)
            if scheduler_mode == 'ReduceLROnPlateau':
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,mode='min',factor=factor_lr,patience=patience,min_lr=min_lr,threshold=1e-6)
            elif scheduler_mode == 'StepLR':
                self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=step_size,gamma=gamma)           
        self.loss_train = loss_train
        self.loss_eval = loss_eval
        self.writer = SummaryWriter(f'runs/{self.model_name}_{archi}_{training_mode}_{opt_name}_{self.dataparameters_title}_N_{self.N}_K_{self.K}')
        
        # Model path information
        self.model_path = f'Result_data/models/{self.dataparameters_title}/N_{self.N}_K_{self.K}/{self.model_name}_{archi}_{training_mode}_{opt_name}'
        os.makedirs(self.model_path,exist_ok=True)
        self.parameters_path = os.path.join(self.model_path,'parameters')
        if os.path.exists(self.parameters_path) & load:
           self.model.load_state_dict(torch.load(self.parameters_path,weights_only=False))
           print('Model succesfully loaded!')
        self.train_loss_path = os.path.join(self.model_path,'train_loss')
        self.eval_trajectories_path = os.path.join(self.model_path,'eval_trajectories')
        self.param_values_path = os.path.join(self.model_path,'param_values')
        self.grad_values_path = os.path.join(self.model_path,'grad_values')
        self.lr_values_path = os.path.join(self.model_path,'lr_values')

        # load or create datasets and data loaders 
        self.training_set = IVAGDataset(data_path=self.train_set_path,dimensions=self.dimensions,dataparameters=self.dataparameters,size=self.train_size,device=self.device)
        self.eval_set = IVAGDataset(data_path=self.eval_set_path,dimensions=self.dimensions,dataparameters=self.dataparameters,size=self.eval_size,device=self.device)
        self.training_loader = DataLoader(self.training_set,batch_size=self.batch_size,shuffle=True)
        self.eval_loader = DataLoader(self.eval_set,batch_size=self.batch_size,shuffle=True)
        self.num_batches = math.ceil(self.training_set.size/self.batch_size)

        # records
        self.train_loss_record = torch.zeros((self.num_epochs,self.num_batches))
        self.eval_trajectories_record_jisi = torch.zeros((self.num_epochs,self.num_batches,self.num_layers+1))
        self.eval_trajectories_record_jiva = torch.zeros((self.num_epochs,self.num_batches,self.num_layers+1))
        self.param_values_records = torch.zeros((self.num_epochs,self.num_batches,self.num_layers,self.num_param))
        self.grad_values_records = torch.zeros((self.num_epochs,self.num_batches,self.num_layers,self.num_param))
        self.lr_values_records = torch.zeros(self.num_epochs,self.num_batches,self.num_layers)
        self.min_eval = float('inf')
        
    def train(self):
        print(f'=================== begin {self.training_mode} training ===================')
        # initialize tracking variables
        self.model.train()
        self.nan_detected = False
        for epoch in range(self.num_epochs):
            for batch,(Rx,Winit,Cinit,A) in enumerate(self.training_loader):
                global_step = epoch * len(self.training_loader) + batch
                if not self.model.tied:
                    self.log_layer_parameters(epoch,batch,global_step)
                outputs = {'W':Winit,'C':Cinit,'Rx':Rx,'A':A}
                B = Winit.shape[0]
                if self.training_mode == 'local':
                    self.local_training(Winit,Cinit,Rx,A,epoch,batch)
                else:
                    if self.training_mode == 'group_of_layers':
                        learning_layers = epoch*(self.num_layers//self.num_epochs),(epoch+1)*(self.num_layers//self.num_epochs)
                    elif self.training_mode == 'one_by_one':
                        learning_layers = (epoch,epoch)
                    else: 
                        #training_mode is 'greedy' or 'end-to-end'
                        learning_layers = (0,self.num_layers)
                    outputs = self.model(Rx,Winit,Cinit,learning_layers=learning_layers,track_cost=True,greedy=self.is_greedy)
                    if self.is_greedy:
                        loss_train_value = torch.mean(outputs['cost'])
                    else:
                        loss_train_value = outputs['cost'][-1]
                    self.optimizer.zero_grad()
                    loss_train_value.backward()
                    if self.gradient_processing == 'normalize':
                        for p in self.model.parameters():
                            if p.grad is not None:
                                p.grad /= (torch.abs(p.grad) + 1e-12)
                    elif self.gradient_processing == 'clip':
                        torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
                    self.optimizer.step()
                    self.train_loss_record[epoch,batch] = loss_train_value/B
                self.writer.add_scalar('Loss/train',self.train_loss_record[epoch,batch], global_step)
                sys.stdout.write(f'\r Epoch {epoch+1}/{self.num_epochs}, batch {batch+1}/{self.num_batches}, loss: {self.train_loss_record[epoch,batch]:.4f} \n') 
                self.compute_trajectory(self.eval_loader,epoch=epoch,batch=batch)
                if torch.isnan(self.eval_trajectories_record_jiva).any() or torch.isnan(self.eval_trajectories_record_jisi).any():
                    self.nan_detected = True
                    break     
                if self.eval_trajectories_record_jiva[epoch,batch,-1].item() < self.min_eval:
                    self.min_eval = self.eval_trajectories_record_jiva[epoch,batch,-1].item()
                    torch.save(self.model.state_dict(),self.parameters_path)
                print(f'jisi loss after epoch {epoch+1} and batch {batch+1} is {self.eval_trajectories_record_jisi[epoch,batch,-1].item()}')
                if self.training_mode != 'local':
                    self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], global_step)
                    if self.scheduler_mode=='ReduceLROnPlateau':
                        self.scheduler.step(self.eval_trajectories_record_jiva[epoch,batch,-1].item())
                    if self.scheduler_mode=='StepLR':
                        self.scheduler.step()
                self.writer.add_scalar('Loss/jisi-eval',self.eval_trajectories_record_jisi[epoch,batch,-1], global_step)
                self.writer.add_scalar('Loss/jiva-eval',self.eval_trajectories_record_jiva[epoch,batch,-1], global_step)
            min_epochs = 1 if (self.model.tied and self.training == 'local') else 3
            if epoch >= min_epochs and torch.mean(self.eval_trajectories_record_jiva[epoch,:,-1]).item() > torch.mean(self.eval_trajectories_record_jiva[epoch-1,:,-1]).item() or self.nan_detected:
                torch.save((epoch,batch),self.model_path+'/ending_step')
                break
        self.writer.close()  
        torch.save(self.train_loss_record,self.train_loss_path)
        torch.save(self.eval_trajectories_record_jiva,self.eval_trajectories_path+'_jiva')
        torch.save(self.eval_trajectories_record_jisi,self.eval_trajectories_path+'_jisi')
        torch.save(self.param_values_records,self.param_values_path)
        torch.save(self.grad_values_records,self.grad_values_path)
        torch.save(self.lr_values_records,self.lr_values_path)
        torch.save(self.nan_detected,self.model_path+'/finished_with_nan')
   
    def local_training(self,Winit,Cinit,Rx,A,epoch,batch):
        B,N,_,K = Winit.shape
        rho_Rx = spectral_norm_extracted(Rx,K,N)  
        W,C,W_prev,C_prev = Winit.clone(),Cinit.clone(),Winit.clone(),Cinit.clone()
        outputs = {'W':W,'C':C,'Rx':Rx,'A':A}
        for i in range(self.num_layers):
            layer = self.model.Layer if self.model.tied else self.model.Layers[i]
            optimizer = self.optimizer if self.model.tied else self.optimizers[i]
            scheduler = self.scheduler if self.model.tied else self.schedulers[i]
            W,C,W_prev,C_prev = layer(Rx,rho_Rx,W,W_prev,C,C_prev,i)
            outputs = {'W':W,'C':C,'Rx':Rx,'A':A}
            loss_train_value = self.loss_train(outputs,greedy=False)
            optimizer.zero_grad()
            loss_train_value.backward()
            if self.gradient_processing == 'normalize':
                for p in self.model.parameters():
                    if p.grad is not None:
                        p.grad /= (torch.abs(p.grad) + 1e-12)
            elif self.gradient_processing == 'clip':
                torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
            optimizer.step()
            scheduler.step()
            W = W.detach() #.requires_grad_(True)
            C = C.detach() #.requires_grad_(True) 
            W_prev = W_prev.detach() #.requires_grad_(True)
            C_prev = C_prev.detach() #.requires_grad_(True)
        self.train_loss_record[epoch,batch] = loss_train_value.item()/B
        
    def compute_trajectory(self,loader=None,write=True,epoch=None,batch=None,record_layer_improvements=False):
        global_step = epoch * len(self.training_loader) + batch + 1
        if loader == None:
            eval_set = IVAGDataset(data_path=self.eval_set_path,dimensions=self.dimensions,dataparameters=self.dataparameters,size=self.eval_size,device=self.device)
            loader = DataLoader(eval_set,batch_size=self.batch_size,shuffle=True)
        for _,(Rx,Winit,Cinit,A) in enumerate(loader):
            with torch.no_grad():                        
                outputs = self.model(Rx,Winit,Cinit,track_jisi=True,A=A,track_cost=True)
                self.eval_trajectories_record_jiva[epoch,batch,:] += outputs['cost']/self.eval_size
                self.eval_trajectories_record_jisi[epoch,batch,:] += outputs['jisi']/self.eval_size
        if write:
            self.plot_trajectory(self.eval_trajectories_record_jiva[epoch,batch,:],'eval_jiva','IVA cost',epoch,batch,'Trajectories',global_step,color='b')
            self.plot_trajectory(self.eval_trajectories_record_jisi[epoch,batch,:],'eval_jisi','jISI score',epoch,batch,'Trajectories',global_step,color='g')
        if record_layer_improvements:
            layer_improvements = self.eval_trajectories_record_jiva[epoch,batch,:-1] - self.eval_trajectories_record_jiva[epoch,batch,1:]       
            return layer_improvements
             
    def log_layer_parameters(self,epoch,batch,global_step):
        for i,layer in enumerate(self.model.Layers):
            for j,param in enumerate(layer.parameters()):
                if 'beta' not in self.param_names[j]:
                    self.param_values_records[epoch,batch,i,j] = self.model.Layers[i].soft(param).item()
                else:
                    self.param_values_records[epoch,batch,i,j] *= 0.1
                if param.grad != None:
                    self.grad_values_records[epoch,batch,i,j] = param.grad.item()
                else:
                    break
            optimizer = self.optimizers[i] if (self.training_mode == 'local' and not self.model.tied) else self.optimizer
            self.lr_values_records[epoch,batch,i] = optimizer.param_groups[0]['lr']

        # Créer les graphes
        for idx in range(self.num_param):
            name = self.param_names[idx]
            self.plot_trajectory(self.param_values_records[epoch,batch,:,idx],name,f'Values of {name}',epoch,batch,'Parameters',global_step,color=plt.cm.tab10(idx),marker='s')
            self.plot_trajectory(self.grad_values_records[epoch,batch,:,idx],'grad '+ name,f'Gradients of {name}',epoch,batch,'Parameters_gradients',global_step,color=plt.cm.tab10(idx),marker='h')
        self.plot_trajectory(self.lr_values_records[epoch,batch,:],'Learning_rates','learning rates',epoch,batch,'Training_parameters',global_step,color='k',marker='o')
        
    def select_num_layers(self,loader=None,tol=1e-2):
        if loader == None:
            eval_set = IVAGDataset(data_path=self.eval_set_path,dimensions=self.dimensions,dataparameters=self.dataparameters,size=self.eval_size,device=self.device)
            loader = DataLoader(eval_set,batch_size=self.batch_size,shuffle=True)
        Rx,Winit,Cinit,A = next(iter(loader))
        outputs = self.model(Rx,Winit,Cinit,track_jisi=True,A=A,track_cost=False)
        L = self.num_layers-1
        jisi_scores = outputs['jisi']
        crit = jisi_scores[L]*(1 + tol)
        jisi = jisi_scores[L]
        while jisi < crit and L > 0:
            L -= 1
            jisi = jisi = jisi_scores[L]
        return L
            
    def shorten_model(self,loader=None,tol=1e-2,save=True):
        L = self.select_num_layers(loader,tol)
        if not self.model.tied:
            newmodel = self.model.Layers[:L]
        else:
            newmodel = self.model
        newmodel.num_layers = L
        if save:
            torch.save(newmodel.state_dict(),self.parameters_path + '_shortened')
            print('succesfully shortened the model!')

    def plot_trajectory(self,data,name,ylabel,epoch,batch,folder,global_step,color='g',marker=''):
        fig,ax = plt.subplots(figsize=(12, 6))
        data_array = data.cpu().numpy()
        ax.plot(range(len(data_array)),data_array,color=color,marker=marker,linestyle='-')
        ax.set_xlabel('Layer number')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel} across layers - Epoch {epoch+1}/Batch {batch+1}')
        ax.grid(True)
        self.writer.add_figure(f'{folder}/{name}',fig,global_step)
  