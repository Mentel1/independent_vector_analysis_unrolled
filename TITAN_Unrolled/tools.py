import torch


def sym(A):
    if A.ndim == 2:
        return (A.transpose(0,1) + A) / 2
    elif A.ndim == 3:
        return (A + torch.moveaxis(A,1,2)) / 2
    else:
        return (A + torch.moveaxis(A,1,2)) / 2
                


def cov_X(X):
    _,T,_ = X.size()
    Rx = torch.einsum('NTK,MTJ->KJNM',X,X) / T
    return Rx


def covid(X):
    """
    Computes the covariance matrix of the input tensor X.
    Parameters
    ----------
        X (torch.FloatTensor): input tensor,size B x N x T x K
    Returns
    -------
        Rx (torch.FloatTensor): covariance matrix,size B x K x K x N x N
    """
    B,N,T,K = X.size()
    
    # Debugging prints
    print(f"Input X shape: {X.shape}")
    
    try:
        Rx = torch.einsum('bntk,bmtj->bkjmn',X,X) / T
        print("Successfully computed covariance matrix")
    except RuntimeError as e:
        print(f"Error in einsum operation: {e}")
        return None
    return Rx

def spectral_norm(M):
    if M.dim() == 2:
        return torch.linalg.norm(M,ord=2)
    else:
        return torch.max(torch.linalg.norm(M,ord=2,dim=(1,2)))
    

def spectral_norm_extracted(Rx,K,N):
    # Rx is expected to have shape (B,K,K,N,N) for batched input
    B = Rx.shape[0]
    Rx_moved = Rx.permute(0, 2, 1, 3, 4)
    # Reshape Rx to (B,K,K*N,N)
    Rx_reshaped = torch.reshape(Rx_moved,(B,K,K*N,N))
    # Compute the 2-norm over dimensions (2,3)
    # norms = torch.norm(Rx_reshaped,p=2,dim=(2,3))
    norms = torch.linalg.matrix_norm(Rx_reshaped, ord=2)
    # Return the maximum norm for each batch
    return torch.max(norms,dim=1).values


def smallest_singular_value(C):
    _,s,_ = torch.svd(C.permute(2,0,1))
    return torch.min(s)

def block_diag(W):
    N,N,K = W.size()
    W_bd = torch.zeros(K,K*N,N,device='cuda')
    for k in range(K):
        W_bd[k,k*N:(k+1)*N,:] = W[:,:,k].t()
    return W_bd


def lipschitz(C,rho_Rx):
    return spectral_norm(C)*rho_Rx


def joint_isi_batch(W,A):
    _,N,_,_ = W.shape
    G_bar = torch.sum(torch.abs(torch.einsum('bnNk,bNvk->bnvk',W,A)),dim=3)
    term1 = torch.sum(torch.sum(G_bar / torch.max(G_bar,dim=2,keepdim=True)[0],dim=2) - 1,dim=1)
    G_bar = G_bar.moveaxis(1,2)
    term2 = torch.sum(torch.sum(G_bar / torch.max(G_bar,dim=2,keepdim=True)[0],dim=2) - 1,dim=1)
    score =  term1 + term2
    return torch.sum(score) / (2 * N * (N - 1))


def joint_isi(W,A):
    N,_,_ = W.shape
    G_bar = torch.sum(torch.abs(torch.einsum('nNk,Nvk->nvk',W,A)),dim=2)
    score = torch.sum(torch.sum(G_bar / torch.max(G_bar,dim=0)[0],dim=0) - 1) + torch.sum(torch.sum(G_bar.t() / torch.max(G_bar.t(),dim=0)[0],dim=0) - 1)
    return score / (2 * N * (N - 1))


def decrease(cost,verbose=0):
    accr = torch.tensor(cost[:-1]) - torch.tensor(cost[1:])
    if torch.all(accr >= 0):
        return True
    else:
        if verbose >= 1:
            for i in range(len(accr)):
                if accr[i] < 0:
                    print("increase at index :",i)
                    print("an increase of :",-accr[i])
                    break
        return False
    
def diff_criteria(U,V):
    B,d1,d2,d3 = U.shape
    D = U - V
    max_norms = torch.max(torch.sum(D**2,dim=2),dim=(1,2))/(2 * d2)
    return torch.sum(max_norms)/B


