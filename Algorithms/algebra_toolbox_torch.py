import torch
from concurrent.futures import ThreadPoolExecutor

def sym_torch(A):
    if A.ndim == 2:
        return (A.transpose(0, 1) + A) / 2
    else:
        return (A + torch.moveaxis(A, 0, 1)) / 2
    
""" def cov_X(X):
    N, V, K = X.size()
    vec_X = X.permute(2, 0, 1).reshape(K * N, V)
    Lambda = torch.matmul(vec_X, vec_X.t()) / V
    return Lambda """


def spectral_norm_torch(M):
    if M.dim() == 2:
        return torch.linalg.norm(M, ord=2)
    else:
        return torch.max(torch.linalg.norm(M, ord=2,dim=(0,1)))
    

""" def spectral_norm_extracted(Lambda, K, N):
    norms = []
    device = Lambda.device  # Get the device of the input tensor
    for j in range(K):
        norms.append(torch.linalg.norm(Lambda[:, j * N:(j + 1) * N].to(device), ord=2))  # Move slice of Lambda to GPU
    return torch.max(torch.tensor(norms).to(device))  # Move the list of norms to GPU and compute the maximum
 """
""" def spectral_norm_extracted(Rx, K, N):
    reshaped_Rx = Rx.view(K, K*N, N)  # Remodelage du tenseur Rx
    norms = torch.norm(reshaped_Rx, p=2, dim=(1, 2))  # Calcul des normes L2
    return torch.max(norms)  # Renvoyer la norme spectrale maximale """

def spectral_norm_extracted_torch(Rx,K,N):
    Rx_moved = Rx.permute(1, 0, 2, 3)
    Rx_reshaped = torch.reshape(Rx_moved,(K,K*N,N))
    norms = torch.linalg.matrix_norm(Rx_reshaped, ord=2)
    # Return the maximum norm for each batch
    return torch.max(norms,dim=0).values

def smallest_singular_value(C):
    return torch.min(torch.svd(C.permute(2, 0, 1))[1])


# def quick_block_diag(W):
#     N, N, K = W.shape
#     W_bd = np.zeros((K, K * N, N))

#     def fill_block(k):
#         nonlocal W_bd
#         W_bd[k, k * N:(k + 1) * N, :] = W[:, :, k].T

#     # Utilisation d'un ThreadPoolExecutor pour paralléliser les boucles
#     with ThreadPoolExecutor() as executor:
#         executor.map(fill_block, range(K))

#     return W_bd

def lipschitz(C,lam):
    return spectral_norm_torch(C)*lam

def joint_isi_torch(W,A):
    N,_,_ = W.size()
    G_bar = torch.sum(torch.abs(torch.einsum('nNk,Nvk->nvk', W, A)), dim=2)
    score = (torch.sum(torch.sum(G_bar / torch.max(G_bar, dim=0)[0], dim=0) - 1) +
             torch.sum(torch.sum(G_bar.t() / torch.max(G_bar.t(), dim=0)[0], dim=0) - 1))
    return score / (2 * N * (N - 1))

def decrease(cost, verbose=0):
    accr = torch.tensor(cost[:-1]) - torch.tensor(cost[1:])
    if torch.all(accr >= 0):
        return True
    else:
        if verbose >= 1:
            for i in range(len(accr)):
                if accr[i] < 0:
                    print("increase at index :", i)
                    print("an increase of :", -accr[i])
                    break
        return False
    
def diff_criteria_torch(A,B):
    if A.shape != B.shape:
        raise ValueError("A and B must be of the same dimension")
    elif A.ndim < 2 or A.ndim > 3:
        raise ValueError("Only tensors of order 2 or 3 are accepted")
    D = A - B
    max_norm = torch.max(torch.sum(D ** 2, dim=1))
    res =  max_norm / (2 * A.shape[1])
    return res

  