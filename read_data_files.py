import torch
import os
import matplotlib.pyplot as plt


def plot_curve(
    xdata,ydata,xlabel,ylabel,title,save_path,color='b',marker='',xlim=None,ylim=None,milestones=[],fontsize=14):
    _,ax = plt.subplots(figsize=(12,6))
    ydata = ydata.cpu().numpy()
    ax.plot(xdata,ydata,color=color,marker=marker,linestyle='-')
    ax.set_xlabel(rf'{xlabel}',fontsize=fontsize)
    ax.set_ylabel(rf'{ylabel}',fontsize=fontsize)
    ax.set_title(rf'{title}',fontsize=fontsize)
    if not xlim == None:
        ax.set_xlim(xlim)
    if not ylim == None:
        ax.set_ylim(ylim)
    for x in milestones:
        ax.axvline(x=x,color='gray',linestyle='--',linewidth=0.8,alpha=0.5)
        # ax.text(x,ax.get_ylim()[1],f'E{epoch}',fontsize=8,color='gray',ha='center',va='bottom')
    ax.grid(True)
    plt.savefig(save_path)
    plt.close()
    
    
def plot_trajectory(model_path,data,name,ylabel,epoch,batch,folder,step,color='g',marker='',xlim=None,ylim=None,fontsize=14):
    xlabel = 'Layer number'
    title = rf'{ylabel} across layers - Epoch {epoch+1}/Batch {batch+1}'
    os.makedirs(f'{model_path}/{folder}',exist_ok=True)
    save_path = f'{model_path}/{folder}/{name}_{step}'
    plot_curve(range(len(data)),data,xlabel,ylabel,title,save_path,color,marker,xlim,ylim,fontsize=fontsize)
    
   
def plot_loss(model_path,data,name,ylabel,num_batch,folder='Losses',color='r',marker='',xlim=None,ylim=None,fontsize=14):
    xlabel = 'Step'
    title = rf'{ylabel} across steps'
    os.makedirs(f'{model_path}/{folder}',exist_ok=True)
    save_path = f'{model_path}/{folder}/{name}'
    ydata = data
    xdata = range(1,len(data)+1)
    milestones = []
    for epoch in range(1,len(data)//num_batch+1):
        milestones.append(epoch*num_batch)
    plot_curve(xdata,ydata,xlabel,ylabel,title,save_path,color,marker,xlim,ylim,milestones,fontsize)


def report_model(param_names,param_labels,model_path):
    model_values = load_model_data(model_path)
    print(f"finished with nan: {model_values['finished_with_nan']}")
    print(f"finished after {model_values['ending_step']} steps")
    end_epoch,end_batch = model_values['ending_step']
    num_epoch,num_batch,num_layers,num_param = model_values['param_values'].shape
    final_step = end_epoch * num_batch + end_batch
    max_step = num_epoch*num_batch
    report_params(model_values,max_step,final_step,num_layers,num_batch,num_param,param_names,param_labels,model_path)
    report_trajectories(model_values,num_layers,max_step,final_step,num_batch,model_path)
    report_losses(model_path,model_values,num_batch,final_step,max_step,num_layers)

def load_model_data(model_path):
    model_values = {}
    for feature in ['ending_step','eval_trajectories_jisi','eval_trajectories_jiva','finished_with_nan','grad_values','lr_values','param_values','parameters','train_loss']:
        if feature in os.listdir(model_path):
            feature_path = os.path.join(model_path,feature)
            model_values[feature] = torch.load(feature_path,weights_only=False,map_location=torch.device('cpu'))
    return model_values

def report_losses(model_path,model_values,num_batch,final_step,max_step,num_layers):
    eval_trajectory_jisi = model_values['eval_trajectories_jisi'].reshape(max_step,num_layers+1)
    eval_trajectory_jiva = model_values['eval_trajectories_jiva'].reshape(max_step,num_layers+1)
    eval_loss_jisi = eval_trajectory_jisi[:final_step,-1]
    plot_loss(model_path,eval_loss_jisi,'eval_loss_jisi','jISI score',num_batch)
    eval_loss_jiva = eval_trajectory_jiva[:final_step,-1]
    plot_loss(model_path,eval_loss_jiva,'eval_loss_jiva','IVA cost',num_batch)
    train_loss = model_values['train_loss'].reshape(-1).detach()
    plot_loss(model_path,train_loss[:final_step],'train_loss','IVA cost',num_batch)                 
    lr_values = model_values['lr_values'][:,:,-1].reshape(-1)
    plot_loss(model_path,lr_values[:final_step],'lr_values','learning rate',num_batch,folder='LR',color='k')

def report_trajectories(model_values,num_layers,max_step,final_step,num_batch,model_path):
    eval_trajectory_jisi = model_values['eval_trajectories_jisi'].reshape(max_step,num_layers+1)
    eval_trajectory_jiva = model_values['eval_trajectories_jiva'].reshape(max_step,num_layers+1)
    jiva_min,jiva_max = eval_trajectory_jiva[:final_step,:].min().item() ,eval_trajectory_jiva[:final_step,:].max().item()
    for step in range(final_step):
        epoch,batch = step//num_batch, step%num_batch
        plot_trajectory(model_path,eval_trajectory_jisi[step,:],'eval_trajectories_jisi','jISI score',epoch,batch,'Trajectories',step,'g',ylim=(0,1))    
        plot_trajectory(model_path,eval_trajectory_jiva[step,:],'eval_trajectories_jiva','IVA cost',epoch,batch,'Trajectories',step,'b',ylim=(jiva_min-0.5,jiva_max+0.5))
        
def report_params(model_values,max_step,final_step,num_layers,num_batch,num_param,param_names,param_labels,model_path):
    param_values = model_values['param_values'].reshape(max_step,num_layers,num_param)
    grad_values = model_values['grad_values'].reshape(max_step,num_layers,num_param)
    for param_idx in range(num_param):
        param_min,param_max = param_values[:final_step,:,param_idx].min().item(),param_values[:final_step,:,param_idx].max().item()
        grad_min,grad_max = grad_values[:final_step,:,param_idx].min().item(),grad_values[:final_step,:,param_idx].max().item()
        for step in range(final_step):
            epoch,batch = step//num_batch, step%num_batch
            plot_trajectory(model_path,param_values[step,:,param_idx],f'values of {param_names[param_idx]}',rf'values of ${param_labels[param_idx]}$',epoch,batch,'Weights_' + param_names[param_idx],step,color=plt.cm.tab10(param_idx),ylim=(param_min,param_max))
            plot_trajectory(model_path,grad_values[step,:,param_idx],f'gradients of {param_names[param_idx]}',rf'gradients of ${param_labels[param_idx]}$',epoch,batch,'Weights_grad_' + param_names[param_idx],step,color=plt.cm.tab10(param_idx),ylim=(grad_min,grad_max))

   
def main():
    param_names = ['alpha','gamma_W','gamma_C','beta_W','beta_C']
    param_labels = [r'\alpha',r'\gamma_\mathcal{W}',r'\gamma_\mathcal{C}',r'\beta_\mathcal{W}',r'\beta_\mathcal{C}']
    cases = os.listdir('Result_data/models')
    for case in cases: #['Case_D']: #
        case_path = os.path.join('Result_data/models',case)
        dims = os.listdir(case_path)
        for dim in dims:
            dim_path = os.path.join(case_path,dim)
            trainings = os.listdir(dim_path)
            for training in trainings:
                training_path = os.path.join(dim_path,training)
                models = os.listdir(training_path)
                for model in models:
                    model_path = os.path.join(training_path,model)
                    # if not "Losses" in os.listdir(model_path):
                    print(model_path)
                    report_model(param_names,param_labels,model_path)
                    
    
if __name__ == "__main__":
    main()