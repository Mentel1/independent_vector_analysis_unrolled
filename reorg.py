import os
import shutil
import torch

base_path = 'Result_data/models'
for data_case in os.listdir(base_path):
    for dim_path in os.listdir(f'{base_path}/{data_case}'):
        data_path = f'{base_path}/{data_case}/{dim_path}'
        for training_path in os.listdir(data_path):
            print(f'{data_path}: {training_path}')
            list_features = training_path.split(sep='_')
            # if len(list_features) == 5:
            #     model_path_param_values = f'{data_path}/{model_name}/param_values'
            #     param_values = torch.load(model_path_param_values)
            #     _,_,num_layers,_ = param_values.shape
            #     step_size = list_features[0][-1]
            #     name = list_features[0][:-1]
            #     if not name == 'UTitan':
            #         continue
            #     archi = list_features[1]
            #     training_mode = list_features[2]
            #     opt = list_features[3]
            #     grad_proc = list_features[4]
            #     training_path = f'{training_mode}_{opt}_{step_size}_{grad_proc}'
            #     new_model_name = f'UTitan_{archi}_{num_layers}'
            #     old_folder_path = f'{data_path}/{model_name}'
            #     new_folder_path = f'{data_path}/{training_path}/{new_model_name}'
            #     os.makedirs(f'{data_path}/{training_path}',exist_ok=True)
            #     os.rename(old_folder_path,new_folder_path)
            if len(list_features) == 4:
                training_mode = list_features[0]
                opt = list_features[1]
                step_size = list_features[2]
                grad_proc = list_features[3]
                list_models = os.listdir(f'{data_path}/{training_path}')
                for model_name in list_models:
                    model_features = model_name.split(sep='_')
                    archi = model_features[1]
                    num_layers = model_features[2]
                    if num_layers != 800:
                        old_model_path = f'{data_path}/UTitan{step_size}_{archi}_{training_mode}_{opt}_{grad_proc}'
                        new_model_path = f'{data_path}/{training_path}/{model_name}'
                        if os.path.exists(old_model_path):
                            for subfolder in os.listdir(old_model_path):
                                shutil.move(f'{old_model_path}/{subfolder}',f'{new_model_path}/{subfolder}')
                            if len(os.listdir(old_model_path)) == 0:
                                os.remove(old_model_path)
