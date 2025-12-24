import torch
import math
from architecture import *
from data import *
from tools import *
from functions import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
class IVAGDataset(Dataset):
    def __init__(self,data_path='',dimensions=None,metaparameters=None,size=1200):
        if os.path.exists(data_path):
            self.data = torch.load(data_path)
            self.size = len(self.data)
            self.N,self.T,self.K = self.data[0][0].shape()
        else:
            self.data = []
            self.N,self.T,self.K = dimensions
            self.metaparameters = metaparameters
            self.num_metaparameters = len(metaparameters)
            self.size = size
            for i in range(self.size):
                metaparam = self.metaparameters[i%self.num_metaparameters]
                X,A = generate_whitened_problem(self.T,self.K,self.N,device=device,rho_bounds=metaparam[0],lambda_=metaparam[1])
                if torch.isnan(X).any() or torch.isnan(A).any():
                    print(f"NaN in sample {i} from generate_whitened_problem")
                    print(f"  X has NaN: {torch.isnan(X).any()}, A has NaN: {torch.isnan(A).any()}")
                    print(f"  metaparam: {metaparam}")
                Winit = make_A(self.K,self.N,device=device)
                if torch.isnan(Winit).any():
                    print(f"NaN in sample {i} from make_A")
                Cinit = make_Sigma(self.K,self.N,rank=self.K+10,device=device)
                if torch.isnan(Cinit).any():
                    print(f"NaN in sample {i} from make_Sigma")
                self.data.append((X,Winit,Cinit,A))       

    def __len__(self):
        return self.size  

    def __getitem__(self,idx):
        return self.data[idx]


class UTitan:
    def __init__(self,path_train='training_data',path_test='testing_data',path_parameters='parameters',mode='end-to-end',dimensions=None,metaparameters=None,train_size=None,test_size=None,lr=0.1,N_updates_W=15,N_updates_C=1,num_epochs=2,early_stopping=True,batch_size=100,num_layers=100,epsilon=1e-12,nu=0.5,zeta=1e-3):
        # Model information
        self.dimensions = dimensions
        self.num_layers = num_layers
        self.model = UTitanIVAGModel(N_updates_W, N_updates_C, num_layers=num_layers, epsilon=epsilon, nu=nu, zeta=zeta).to(device)
        if os.path.exists(path_parameters):
           self.model.load_state_dict(torch.load(path_parameters, weights_only=True))
        # training information
        self.metaparameters = metaparameters
        self.train_size = train_size
        self.test_size = test_size
        self.path_test, self.path_train, self.path_save = path_test,path_train,path_parameters
        self.mode = mode #'first_layer' or 'greedy' or 'last_layers_lpp' or 'test'
        self.lr = lr
        self.num_epochs = num_epochs
        self.early_stopping = early_stopping
        self.batch_size = batch_size 
        self.dtype = torch.cuda.FloatTensor 
        print(self.model.parameters())
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.lr,weight_decay=1e-3)
            # Initialize learning rate scheduler
            #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10) 
        self.loss = ISI_loss()
        
    def train(self):
        """
        Trains U_TITAN.
        Parameters
        ----------
            layer (int): number of the layer to be trained, numbering starts at 0 (default is 0)
        """
        # load or create datasets and data loaders
        if os.path.exists(self.path_train):
            self.training_set = IVAGDataset(data_path=self.path_train)
        else:
            self.training_set = IVAGDataset(dimensions=self.dimensions,metaparameters=self.metaparameters,size=self.train_size)
        for i in range(len(self.training_set)):
            X,Winit,Cinit,A = self.training_set[i]
            if torch.isnan(X).any() or torch.isnan(Winit).any() or torch.isnan(Cinit).any() or torch.isnan(A).any():
                print(f"NaN detected in training set at index {i}")
                break
        self.train_loader = DataLoader(self.training_set,batch_size=self.batch_size,shuffle=True)
        if os.path.exists(self.path_test):
            self.testing_set = IVAGDataset(data_path=self.path_test)
        else:
            self.testing_set = IVAGDataset(dimensions=self.dimensions,metaparameters=self.metaparameters,size=self.test_size)
        self.test_loader = DataLoader(self.testing_set,batch_size=self.batch_size,shuffle=True)
          
        self.num_batches = math.ceil(self.training_set.size/self.batch_size)
        self.model.train()
        if self.mode == 'end-to-end':
            # trains the whole network
            print('=================== End-to-end training ===================')
            # initialize tracking variables
            jisi_train = torch.zeros(self.num_epochs)
            jisi_eval = torch.zeros(self.num_epochs)
            if self.early_stopping:
                loss_min_eval = float('Inf')
            for epoch in range(0,self.num_epochs):
                for batch,(Xs,Winits,Cinits,As) in enumerate(self.train_loader):
                    if torch.isnan(Winits).any():
                        print(f"NaN in W at batch {batch}")
                    if torch.isnan(Cinits).any():
                        print(f"NaN in C at batch {batch}")
                    Ws,_ = self.model(Xs,Winits,Cinits,self.mode)
                    loss = self.loss(Ws,As)
                    jisi_train[epoch] += loss.item()/self.train_size
                    sys.stdout.write(f'\r Epoch {epoch+1}/{self.num_epochs}, batch {batch+1}/{self.num_batches}, loss: {loss:.4f} \n')
                    # sets the gradients to zero, performs a backward pass, and updates the weights.
                    self.optimizer.zero_grad()
                    loss.backward()
                    print(loss.device)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                for (Xs,Winits,Cinits,As) in self.test_loader:
                    with torch.no_grad():
                        Ws,_ = self.model(Xs,Winits,Cinits,self.mode)
                        jisi_eval[epoch] += self.loss(Ws,As).item()/self.test_size
                print(jisi_eval[epoch])
                self.scheduler.step(jisi_eval[epoch])
                if self.early_stopping:
                    if jisi_eval[epoch] < loss_min_eval:
                        loss_min_eval = jisi_eval[epoch]
                    else:
                        break

                    
        # elif self.mode == 'greedy':
        #     # trains the next layer
        #     print('=================== Layer number %d ==================='%(layer))
        #     # to store results
        #     loss_epochs       =  np.zeros(self.num_epochs)
        #     isi_train   =  np.zeros(self.num_epochs)
        #     isi_val     =  np.zeros(self.num_epochs)
        #     loss_min_val      =  float('Inf')

        #     # puts first blocks in evaluation mode: gradient is not computed
        #     #self.model.GradFalse(layer,self.mode) 
        #     # defines the optimizer
        #     lr = self.lr
        #     optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,self.parameters()),lr=lr)
        #     #==========================================================================================================
        #     # trains for several epochs
        #     for epoch in range(0,self.num_epochs):
                
        #         self.model.Layers[layer].train() # training mode
        #         # goes through all minibatches
        #         for i,minibatch in enumerate(self.train_loader):
        #             [X, A] = minibatch  # gets the minibatch
        #             #print("minibatch size: ", minibatch[0].size())
        #             X = X[0]
        #             A = A[0]
        #             print("X size: ", X.size())
        #             Rx = cov_X(X)
        #             print("Rx size: ", Rx.size())
        #             W,C = initialize(N,K,X=X,Rx=Rx)

        #             W_predicted,C_predicted = self.model(Rx,W,C,self.mode,layer)

        #             # Computes and prints loss
        #             loss = self.loss_fun(W_predicted, A)
        #             loss_epochs[epoch] += torch.Tensor.item(loss)
        #             sys.stdout.write('\r Epoch %d/%d, minibatch %d/%d, loss: %.4f \n' % (epoch+1,self.num_epochs,i+1,self.size_train//self.batch_size,loss))
                    
        #             # sets the gradients to zero, performs a backward pass, and updates the weights.
        #             optimizer.zero_grad()
        #             loss.backward()
        #             optimizer.step()

        #             # Check gradients
        #             """ for name, param in self.model.named_parameters():
        #                 if param.grad is None:
        #                     print(f"Parameter '{name}' has no gradient")
        #                 else:
        #                     print(f"Parameter '{name}' gradient mean: {param.grad.mean().item()}") """


        #     # tests on validation set
                
        #     # training is finished
        #     print('-----------------------------------------------------------------')
        #     print('Training of Layer ' + str(layer) + ' is done.')
        #     print('-----------------------------------------------------------------')
            
            # # calls the same function to start training of next block 
            # if layer < self.num_layers-1:
            #     self.train(layer=layer+1)
            # else:
            #     print('Training of all layers is done.')
        

