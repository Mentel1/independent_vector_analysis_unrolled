import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
import cProfile
from experiment import *
from algorithms import *
from Algorithms.titan_iva_g_reg_torch import *
from TITAN_Unrolled.data import *

def create_algos_titanIVAG(varying_param, values, color_bounds=[(0.2,1,0.2),(0.2,0.2,1)],base_params={},basename=''):
    algos = []
    nval = len(values)
    for i, value in enumerate(values):
        params = base_params.copy()
        params[varying_param] = value
        t = i / (nval - 1)
        params['color'] = tuple((1 - t) * c0 + t * c1 for c0, c1 in zip(color_bounds[0], color_bounds[1]))
        params['name'] = basename + '_' + varying_param + '=' + str(value)      
        algos.append(TitanIvaG(**params))
    return algos


#=============================================================================================
# MAIN EXPERIMENT (MULTIPARAMETER)
#=============================================================================================

Ks = [5]
Ns = [10] 
common_params = [Ks,Ns]
data_case = 'Case_D'

algos = []
algos.append(TitanIvaG(nu=0,gamma_c=1.99))
for archi in ['tied','untied','inertial-tied','inertial-untied']:
    algo = UTitanIvaG(name='UTitan'+'_'+archi,archi=archi,dimensions=(Ns[0],10000,Ks[0]),data_case=data_case,num_layers=200,tol=1e-2)
    algos.append(algo)

exp = ComparisonExperimentIvaG(name='Unrolling_comparison_D_small',dataparams_titles=[data_case],common_params=common_params,algos=algos,N_exp=100)
exp.compute_multi_runs()

# exp_path = 'Result_data/experiments/2026-03-18_11-13_Testing_unrolling/res/Case_A/N_30_K_20'
# algo_names = ['titan','UTitan_inertial-tied','UTitan_inertial-untied','UTitan_tied','UTitan_untied']
# features = ['final_jisi','total_times']

# for name in algo_names:
#     for feature in features:
#         res_path = f'{exp_path}/{name}_{feature}'
#         vec = np.fromfile(res_path,sep=',')
#         print(f'Average {feature} for {name} is {np.mean(vec)} with a standard deviation of {np.std(vec)}')
        

#===========================================================================================
# ANALYSIS OF THE SLOWEST SUBPROCESS
#===========================================================================================

# if __name__ == '__main__':
#     import cProfile, pstats
#     profiler = cProfile.Profile()
#     profiler.enable()
#     exp.compute_empirical_convergence(0,20,20,['costs','jisi','times'],detailed=False)
#     profiler.disable()
#     stats = pstats.Stats(profiler).sort_stats('cumtime')
#     stats.print_stats()
