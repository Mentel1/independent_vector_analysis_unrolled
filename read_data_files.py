import torch
import os
import matplotlib.pyplot as plt

def plot_trajectory(model_path,data,name,ylabel,epoch,batch,folder,final_step,color='g',marker='',xlim=None,ylim=None,fontsize=14):
    fig,ax = plt.subplots(figsize=(12,6))
    data_array = data.cpu().numpy()
    ax.plot(range(len(data_array)),data_array,color=color,marker=marker,linestyle='-')
    ax.set_xlabel('Layer number',fontsize=fontsize)
    ax.set_ylabel(rf'{ylabel}',fontsize=fontsize)
    ax.set_title(rf'{ylabel} across layers - Epoch {epoch+1}/Batch {batch+1}',fontsize=fontsize)
    if not xlim == None:
        ax.set_xlim(xlim)
    if not ylim == None:
        ax.set_ylim(ylim)
    ax.grid(True)
    os.makedirs(os.path.join(model_path ,folder),exist_ok=True)
    file_path = model_path + f'/{folder}/{name}' + f'_{final_step}'
    plt.savefig(file_path)
    plt.close()

def plot_loss(model_path,data,name,ylabel,num_batch,folder='Losses',color='r',marker='',xlim=None,ylim=None,fontsize=14):
    fig,ax = plt.subplots(figsize=(12,6))
    data_array = data.detach().cpu().numpy()
    ax.plot(range(1,len(data_array)+1),data_array,color=color,marker=marker,linestyle='-')
    ax.set_xlabel('Global step',fontsize=fontsize)
    ax.set_ylabel(rf'{ylabel}',fontsize=fontsize)
    ax.set_title(rf'{ylabel} across steps',fontsize=fontsize)
    if not xlim == None:
        ax.set_xlim(xlim)
    if not ylim == None:
        ax.set_ylim(ylim)
    ax.grid(True)
    for epoch in range(1,len(data_array) // num_batch + 1):
        ax.axvline(x=epoch * num_batch,color='gray',linestyle='--',linewidth=0.8,alpha=0.5)
        # ax.text(epoch * num_batch,ax.get_ylim()[1],f'E{epoch}',fontsize=8,color='gray',ha='center',va='bottom')
    os.makedirs(os.path.join(model_path ,folder),exist_ok=True)
    file_path = model_path + f'/{folder}/{name}'
    plt.savefig(file_path)
    plt.close()


def report_model(param_names,param_labels,model_path):
    model_values = load_model_data(model_path)
    print(f"finished with nan : {model_values['finished_with_nan']}")
    print(f"finished after {model_values['ending_step']} steps")
    end_epoch,end_batch = model_values['ending_step']
    num_epoch,num_batch,num_layers,num_param = model_values['param_values'].shape
    final_step = end_epoch * num_batch + end_batch
    max_step = num_epoch*num_batch
    param_values = model_values['param_values'].reshape(max_step,num_layers,num_param)
    grad_values = model_values['grad_values'].reshape(max_step,num_layers,num_param)
    eval_trajectory_jisi = model_values['eval_trajectories_jisi'].reshape(max_step,num_layers+1)
    eval_trajectory_jiva = model_values['eval_trajectories_jiva'].reshape(max_step,num_layers+1)
    jiva_min,jiva_max = eval_trajectory_jiva[:final_step,:].min().item() ,eval_trajectory_jiva[:final_step,:].max().item()
    # for step in range(final_step):
    #     epoch,batch = step//num_batch, step%num_batch
    #     plot_trajectory(model_path,eval_trajectory_jisi[step,:],'eval_trajectories_jisi','jISI score',epoch,batch,'Trajectories',step,'g',ylim=(0,1))    
    #     plot_trajectory(model_path,eval_trajectory_jiva[step,:],'eval_trajectories_jiva','IVA cost',epoch,batch,'Trajectories',step,'b',ylim=(jiva_min-0.5,jiva_max+0.5))
    #     for param_idx in range(num_param):
    #         plot_trajectory(model_path,param_values[step,:,param_idx],f'values of {param_names[param_idx]}',rf'values of ${param_labels[param_idx]}$',epoch,batch,'Weights_' + param_names[param_idx],step,color=plt.cm.tab10(param_idx))
    #         plot_trajectory(model_path,grad_values[step,:,param_idx],f'gradients of {param_names[param_idx]}',rf'gradients of ${param_labels[param_idx]}$',epoch,batch,'Weights_grad_' + param_names[param_idx],step,color=plt.cm.tab10(param_idx))
    #         # Plot the validation losses
    # eval_loss_jisi = eval_trajectory_jisi[:final_step,-1]
    # plot_loss(model_path,eval_loss_jisi,'eval_loss_jisi','jISI score',num_batch)
    # eval_loss_jiva = eval_trajectory_jiva[:final_step,-1]
    # plot_loss(model_path,eval_loss_jiva,'eval_loss_jiva','IVA cost',num_batch)
    # train_loss = model_values['train_loss'].reshape(-1)
    # plot_loss(model_path,train_loss[:final_step],'train_loss','IVA cost',num_batch)                 
    # lr_values = model_values['lr_values'][:,:,-1].reshape(-1)
    # plot_loss(model_path,lr_values[:final_step],'lr_values','learning rate',num_batch,folder='LR',color='k')

def load_model_data(model_path):
    model_values = {}
    for feature in ['ending_step','eval_trajectories_jisi','eval_trajectories_jiva','finished_with_nan','grad_values','lr_values','param_values','parameters','train_loss']:
        if feature in os.listdir(model_path):
            feature_path = os.path.join(model_path,feature)
            model_values[feature] = torch.load(feature_path,weights_only=False,map_location=torch.device('cpu'))
    return model_values

def main():
    param_names = ['alpha','gamma_W','gamma_C','beta_W','beta_C']
    param_labels = [r'\alpha',r'\gamma_\mathcal{W}',r'\gamma_\mathcal{C}',r'\beta_\mathcal{W}',r'\beta_\mathcal{C}']
    cases = os.listdir('Result_data/models')
    for case in ['Case_D']: #cases:
        case_path = os.path.join('Result_data/models',case)
        dims = os.listdir(case_path)
        for dim in dims:
            dim_path = os.path.join(case_path,dim)
            models = os.listdir(dim_path)
            for model in models:
                model_path = os.path.join(dim_path,model)
                # if not "Losses" in os.listdir(model_path):
                print(model_path)
                report_model(param_names,param_labels,model_path)
                    
    
if __name__ == "__main__":
    main()