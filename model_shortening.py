import torch
import os
import matplotlib.pyplot as plt
import read_data_files
from TITAN_Unrolled.datasets import *
from TITAN_Unrolled.architecture import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_model_shortening(model_path,tol=1e-2,folder='plot_val_loss',file='shortening',colors=['green','black'],labels=['decrease','stagnation'],fontsize=16):
    N,V,K = 30,10000,20
    num_layers = 500
    print(os.path.exists(f'{model_path}/parameters'))
    model = UTitanIVAGModel(N,K,num_layers,N_updates_W=10,N_updates_C=1,archi='untied',load=True,parameters_path=f'{model_path}/parameters').to(device)
    xlabel = 'Layer'
    ylabel = 'Validation loss'
    title = rf'{ylabel} across layers'
    output_path = f'Result_data/{folder}'
    eval_set = IVAGDataset(name='eval',dimensions=(N,V,K),data_case='Case_A',device=device)
    loader = DataLoader(eval_set,batch_size=eval_set.__len__())
    Rx,Winit,Cinit,A = next(iter(loader))
    outputs = model(Rx,Winit,Cinit,track_jisi=True,A=A,track_cost=False)
    L = model.num_layers - 1
    jisi_scores = outputs['jisi']/eval_set.__len__()
    crit = jisi_scores[L]*(1 + tol)
    jisi = jisi_scores[L]
    while jisi < crit and L > 0:
        L -= 1
        jisi = jisi_scores[L]
    os.makedirs(output_path,exist_ok=True)
    _,ax = plt.subplots(figsize=(12,6))
    ydata_1 = jisi_scores[:L].numpy()
    xdata_1 = range(L)
    ax.plot(xdata_1,ydata_1,color=colors[0],linestyle='-',linewidth=2,label=labels[0])
    ydata_2 = jisi_scores[L:].numpy()
    xdata_2 = range(L,model.num_layers+1)
    ax.plot(xdata_2,ydata_2,color=colors[1],linestyle='-',linewidth=2,label=labels[1])
    ax.set_xlabel(rf'{xlabel}',fontsize=fontsize)
    ax.set_ylabel(rf'{ylabel}',fontsize=fontsize)
    ax.set_title(rf'{title}',fontsize=fontsize)
    ax.legend(loc='upper right',fontsize=fontsize)
    ax.grid(True)
    ax.axvline(x=L,color='gray',linestyle='--',linewidth=1,alpha=0.8)
    ax.text(L,ax.get_ylim()[1],'$L^*$',fontsize=fontsize,color='black',ha='center',va='bottom')
    plt.savefig(f'{output_path}/{file}')
    plt.close()
  
model_path = os.path.join(
    "Result_data",
    "models",
    "Case_A",
    "N_30_K_20",
    "local_SGD_5_normalize",
    "UTitan_untied_500"
)
plot_model_shortening(model_path)