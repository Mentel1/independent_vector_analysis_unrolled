import os
import torch

base_path = 'Result_data/models'
for data_case in os.listdir(base_path):
    for dim_path in os.listdir(f'{base_path}/{data_case}'):
        data_path = f'{base_path}/{data_case}/{dim_path}'
        for model_name in os.listdir(data_path):
            print(f'{data_path}: {model_name}')
            list_features = model_name.split(sep='_')
            if len(list_features) == 5:
                model_path_param_values = f'{data_path}/{model_name}/param_values'
                param_values = torch.load(model_path_param_values)
                _,_,num_layers,_ = param_values.shape
                step_size = list_features[0][-1]
                name = list_features[0][:-1]
                if not name == 'UTitan':
                    continue
                archi = list_features[1]
                training_mode = list_features[2]
                opt = list_features[3]
                grad_proc = list_features[4]
                training_path = f'{training_mode}_{opt}_{step_size}_{grad_proc}'
                new_model_name = f'UTitan_{archi}_{num_layers}'
                old_folder_path = f'{data_path}/{model_name}'
                new_folder_path = f'{data_path}/{training_path}/{new_model_name}'
                os.makedirs(f'{data_path}/{training_path}',exist_ok=True)
                os.rename(old_folder_path,new_folder_path)

