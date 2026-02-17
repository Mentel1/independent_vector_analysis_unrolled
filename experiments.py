import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
import cProfile
from class_exp import *
from class_algos import *
from algorithms.titan_iva_g_reg_torch import *

label_size = 20
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size
plt.rcParams['text.usetex'] = True

# Function to generate dataparameters for the multiparameter experiment
def get_dataparameters(rhos,lambdas):
    dataparameters_multiparam = []
    for rho_bounds in rhos:
        for lambda_ in lambdas:
            dataparameters_multiparam.append((rho_bounds,lambda_))
    return dataparameters_multiparam


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



#================================================================================================
# MAIN EXPERIMENT (MULTIPARAMETER)
#================================================================================================

lambda_1 = 0.04
lambda_2 = 0.25
rho_bounds_1 = [0.2,0.3]
rho_bounds_2 = [0.6,0.7]
rhos = [rho_bounds_1,rho_bounds_2]
lambdas = [lambda_1,lambda_2]
dataparameters_multiparam = get_dataparameters(rhos,lambdas)
dataparameters_titles_multiparam = ['Case_A','Case_B','Case_C','Case_D']
dataparameters_base = get_dataparameters([[0.4,0.6]],[0.1])
dataparameters_base_titles = ['Base_Case']
# dataparameters_identifiability = [1e-2,1e-1,1]
# dataparameters_titles_identifiability = ['low identifiability','medium identifiability','high identifiability']
# dataparameters = [{'noise_levels':[0,1e-3,1e-2,1e-1,1,10]}]
dataparameters = [{'num_samples':[10000,5000,1000,500,200,150,120,100]}]

Ks = [5,10,20]
Ns = [10,20,30] 
common_parameters = [Ks,Ns]


# exp = ComparisonExperimentIvaG.from_folder('2025-12-29_16-46_ExternalRace_multiparam')
# exp.make_table(),

algo_titan = TitanIvaG([1,0,0],nu=0,max_iter_int_W=15,gamma_c=1.99)
algos = [algo_titan]
# algos = create_algos_titanIVAG(varying_param='epsilon',values=[1e-3,1e-2,1e-1],base_params={'nu':0,'max_iter_int_W':15,'gamma_c':1.99},basename='palm')
exp = ComparisonExperimentIvaG('Robustness_noise',dataparameters,dataparameters_base_titles,[[20],[10]],algos=algos,N_exp=10,legend_fontsize=20,title_fontsize=30,updates=True)
exp.compute_multi_runs()
# exp.compute_empirical_convergence(0,20,20,['costs','jisi','times'],detailed=False)
# exp.draw_empirical_convergence(0,20,5,'jisi',mode='iter')

   
# ================================================================================================
# ANALYSIS OF THE SLOWEST SUBPROCESS
# ================================================================================================

# if __name__ == '__main__':
#     import cProfile, pstats
#     profiler = cProfile.Profile()
#     profiler.enable()
#     exp.compute_empirical_convergence(0,20,20,['costs','jisi','times'],detailed=False)
#     profiler.disable()
#     stats = pstats.Stats(profiler).sort_stats('cumtime')
#     stats.print_stats()
