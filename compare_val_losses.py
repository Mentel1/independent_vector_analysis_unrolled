import torch
import os
import matplotlib.pyplot as plt

def compare_val_loss(model_path_list,folder,file,names,colors,fontsize=16):
    eval_loss_jisi_list = []
    max_step_all = 0
    for model_path in model_path_list:
        file_path = f'{model_path}/eval_trajectories_jisi'
        eval_traj_jisi = torch.load(file_path,weights_only=False,map_location=torch.device('cpu'))
        num_epoch,num_batch,num_layers_p1 = eval_traj_jisi.shape
        end_epoch,end_batch = torch.load(f'{model_path}/ending_step',weights_only=False,map_location=torch.device('cpu'))
        print(f'{model_path} : {end_epoch} epochs')
        final_step = end_epoch * num_batch + end_batch
        max_step = num_epoch*num_batch
        eval_traj_jisi = eval_traj_jisi.reshape(max_step,num_layers_p1)
        eval_loss_jisi = eval_traj_jisi[:final_step,-1]
        eval_loss_jisi_list.append(eval_loss_jisi)
        max_step_all = max(max_step_all,max_step)
    xlabel = 'Step'
    ylabel = 'Validation loss'
    title = rf'{ylabel} across steps'
    output_path = f'Result_data/{folder}'
    os.makedirs(output_path,exist_ok=True)
    _,ax = plt.subplots(figsize=(12,6))
    for i,eval_loss_jisi in enumerate(eval_loss_jisi_list):
        ydata = eval_loss_jisi.cpu().numpy()
        xdata = range(1,len(ydata)+1)
        ax.plot(xdata,ydata,color=colors[i],linestyle='-',linewidth=2,label=names[i])
        ax.set_xlabel(rf'{xlabel}',fontsize=fontsize)
        ax.set_ylabel(rf'{ylabel}',fontsize=fontsize)
        ax.set_title(rf'{title}',fontsize=fontsize)
    ax.legend(fontsize=fontsize)
    ax.grid(True)
    plt.savefig(f'{output_path}/{file}')
    plt.close()
        
    
# colors = ["navy","royalblue","mediumpurple","darkviolet"]
# names = ["end-to-end + Adam","end-to-end + SGD","local + Adam","local + SGD"]
# base_path = 'Result_data/models/Case_A/N_30_K_20/'
# model_name = '/UTitan_untied_500'
# model_path_list = []
# for mode in ['end-to-end_Adam_5_raw','end-to-end_SGD_5_normalize','local_Adam_5_raw','local_SGD_5_normalize']:
#     model_path_list.append(base_path + mode + model_name)
    
# compare_val_loss(model_path_list,'comparison',names,colors)

colors = ["royalblue","slateblue","blueviolet","darkviolet","mediumorchid","mediumvioletred","crimson"]
names = [r'$T_{\eta} = 2$',r'$T_{\eta} = 3$',r'$T_{\eta} = 4$',r'$T_{\eta} = 5$',r'$T_{\eta} = 6$',r'$T_{\eta} = 7$',r'$T_{\eta} = 8$']
base_path = 'Result_data/models/Case_A/N_30_K_20/local_SGD_'
model_name = '_normalize/UTitan_untied_500'
model_path_list = []
for mode in ['2','3','4','5','6','7','8']:
    model_path_list.append(base_path + mode + model_name)
    
compare_val_loss(model_path_list,'comparison','periods',names,colors)

