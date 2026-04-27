import torch
from .functions import *

class ISI_loss():
    """
    Defines the ISI training loss.
    Attributes
    ----------
        ISI : function computing the ISI Score 
    """
    def __init__(self): 
        super(ISI_loss,self).__init__()
        
    def __call__(self,outputs,greedy=False):
        W,store_W,A = outputs['W'],outputs['store_W'],outputs['A']
        num_layers = store_W.shape[0]
        """
        Computes the training loss.
        Parameters
        ----------
            output  (torch.FloatTensor): restored images,size n*c*h*w 
            label (torch.FloatTensor): ground-truth images,size n*c*h*w
        Returns
        -------
            (torch.FloatTensor): mean ISI Score of the batch,size 1 
        """
        if greedy:
            loss = torch.zeros(1,device=W.device)
            for i in range(num_layers):
                W = store_W[i,:,:,:,:]
                loss += joint_isi_batch(W,A)/num_layers
            return loss
        else:
            return joint_isi_batch(W,A)


class IVA_loss():
    def __init__(self): 
        super(IVA_loss,self).__init__()
        
    def __call__(self,outputs,greedy=False): # A modifier pour faire le calcul sur tout le batch (W et C sont de dim B*_*_*_)
        W,C,Rx = outputs['W'],outputs['C'],outputs['Rx']
        res = torch.zeros(1,device=W.device)
        if greedy:
            store_W,store_C = outputs['store_W'],outputs['store_C']
            for i in range(store_W.shape[0]):
                W,C = store_W[i,:,:,:,:],store_C[i,:,:,:,:]
                det_C = torch.det(C.permute(0,3,1,2))  # Déterminant de C
                det_W = torch.det(W.permute(0,3,1,2))  # Déterminant de W
                tr_C = torch.diagonal((C.permute(0,3,1,2) - 1) ** 2, dim1=-2, dim2=-1).sum()
                tr_term = (torch.einsum('bKJn,bnNK,bKJNM,bnMJ -> bnKJ',(C,W,Rx,W))).sum() / 2  # Terme de trace
                res -= torch.sum(torch.log(torch.abs(det_C))) / 2  # Premier terme
                res += 0.5 * tr_C  # Deuxième terme
                res += tr_term  # Troisième terme
                res -= torch.sum(torch.log(torch.abs(det_W)))  # Quatrième terme
        else:
            det_C = torch.det(C.permute(0,3,1,2))  # Déterminant de C
            term1 = torch.sum(torch.log(torch.abs(det_C)))
            # print(f'sum log det_C is computed and equals to {term1}')
            det_W = torch.det(W.permute(0,3,1,2))  # Déterminant de W
            term4 = torch.sum(torch.log(torch.abs(det_W)))
            # print(f'sum log |det_W| is computed and equals to {term4}')
            tr_C = torch.diagonal((C.permute(0,3,1,2) - 1) ** 2, dim1=-2, dim2=-1).sum()
            # print(f'tr_C is computed and equals to {tr_C}')
            tr_term = (torch.einsum('bKJn,bnNK,bKJNM,bnMJ -> bnKJ',(C,W,Rx,W))).sum() / 2  # Terme de trace
            # print(f'tr_term is computed and equals to {tr_term}')
            res -= term1/ 2  # Premier terme
            res += 0.5 * tr_C  # Deuxième terme
            res += tr_term  # Troisième terme
            res -= term4  # Quatrième terme
        return res
