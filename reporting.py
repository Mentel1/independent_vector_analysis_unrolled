import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import os


label_size = 20
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size
plt.rcParams['text.usetex'] = True

class ExperimentReporter:
    
    def __init__(self,experiment,N_exp=100,table_fontsize=8,median=False,std=False,updates=False,legend=True,legend_fontsize=12,title_fontsize=14):
        self.experiment = experiment
        self.output_folder = self.experiment.output_folder
        self.algos = self.experiment.algos
        self.dataparameters = self.experiment.dataparameters
        self.dataparameters_titles = self.experiment.data_parameters_titles
        self.common_parameters = self.experiment.common_parameters
        self.table_fontsize = table_fontsize
        self.median = median
        self.std = std
        self.legend = legend
        self.title_fontsize = title_fontsize
        self.legend_fontsize = legend_fontsize
        
        
    def compute_features(self,algo):
        Ks,Ns = self.common_parameters
        algo.results['full_results_jisi'] = np.zeros((len(self.data_parameters),len(Ks),len(Ns),self.N_exp))
        algo.results['full_results_times'] = np.zeros((len(self.data_parameters),len(Ks),len(Ns),self.N_exp))
        if self.updates:
            algo.results['full_results_updates'] = np.zeros((len(self.data_parameters),len(Ks),len(Ns),self.N_exp))
        for a,dataparam in enumerate(self.dataparameters_titles):
            for jn,N in enumerate(Ns):
                for ik,K in enumerate(Ks):
                    path = f'{self.output_folder}/{dataparam}/N_{N}_K_{K}'
                    algo.fill_from_folder(path)
                    algo.results['full_results_jisi'][a,ik,jn,:] = algo.results['final_jisi']
                    algo.results['full_results_times'][a,ik,jn,:] = algo.results['total_times']
                    if self.updates:
                        algo.results['full_results_updates'][a,ik,jn,:] = algo.results['number_updates']
        algo.results['mean_jisi'] = np.mean(algo.results['full_results_jisi'],axis=-1)
        algo.results['mean_times'] = np.mean(algo.results['full_results_times'],axis=-1)
        if self.updates:
            algo.results['mean_updates'] = np.mean(algo.results['full_results_updates'],axis=-1)
        print(algo.results['mean_jisi'])
        if self.std:
            algo.results['std_jisi'] = np.std(algo.results['full_results_jisi'],axis=-1)
            algo.results['std_times'] = np.std(algo.results['full_results_times'],axis=-1)
            if self.updates:
                algo.results['std_updates'] = np.std(algo.results['full_results_updates'],axis=-1)
        if self.median:
            algo.results['median_jisi'] = np.median(algo.results['full_results_jisi'],axis=-1)
            algo.results['median_dev_jisi'] = np.median(abs(algo.results['full_results_jisi'] - algo.results['median_jisi']),axis=-1)
            algo.results['median_times'] = np.median(algo.results['full_results_times'],axis=-1)
            algo.results['median_dev_times'] = np.median(abs(algo.results['full_results_times'] - algo.results['median_times']),axis=-1)
            if self.updates:
                algo.results['median_updates'] = np.median(algo.results['full_results_updates'],axis=-1)
                algo.results['median_updates'] = np.median(abs(algo.results['full_results_updates'] - algo.results['median_updates']),axis=-1)
    
    def list_features(self):
        res = ['mean_jisi','mean_times']
        if self.updates:
            res += ['mean_updates']
            if self.std:
                res += ['std_updates']
            if self.median:
                res += ['median_updates','median_dev_updates']
        if self.std:
            res += ['std_jisi','std_times']
        if self.median:
            res += ['median_jisi','median_dev_jisi','median_times','median_dev_times']
        return res
        
    def best_perf(self,feature):
        all_perfs = np.array([algo.results[feature] for algo in self.algos])
        return np.min(all_perfs, axis=0)
   
    base_feature_names = {'mean_jisi':'$\\mu_{\\rm jISI}$','mean_times':'$\mu_\\texttt{T}$','mean_updates':'$\mu_\\texttt{N}$','median_jisi':'$\\widehat{\\mu}_{\\rm jISI}$','std_jisi':'$\\sigma_{\\rm jISI}$','median_dev_jisi':'$\\widehat{\\sigma}_{\\rm jISI}$',}
    base_tols = {'mean_jisi':1e-4,'mean_times':1e-2}
    
    def make_table(self,tols=base_tols,feature_names=base_feature_names,filename='table_results.txt'):
        Ks,Ns = self.common_parameters
        for algo in self.algos:
            self.compute_features(algo,Ks,Ns)  
        features = self.list_features()
        n_cols = len(Ks)*len(Ns)
        bold_numbers = {}
        for feature in ['mean_jisi','mean_times']:
            best_feature = self.best_perf(feature)
            for algo in self.algos:
                bold_numbers[(feature,algo)] = algo.results[feature] <= best_feature + tols[feature]
        # We consider that results_algo come from the same experiment 
        output_path = os.path.join(self.output_folder, filename)
        if os.path.exists(output_path):
            os.remove(output_path)
        with open(output_path, 'a') as file:
            file.write('\\begin{table}[h!]\n\\caption{'+'blablabla'+'}\n\\vspace{0.4cm}\n')
            file.write(f'\\fontsize{{{self.table_fontsize}pt}}{{{self.table_fontsize}pt}}\selectfont\n')
            file.write('\\begin{tabular}{cm{1cm}m{0.5cm}'+n_cols*'c'+'}\n')
            file.write('& &')
            for K in Ks:
                file.write(f' & \\multicolumn{{{len(Ns)}}}{{c}}{{$K$ = {K}}}')
            file.write('\\\\\n')
            for ik,K in enumerate(Ks):
                file.write(f' \\cmidrule(lr){{{4+ik*len(Ns)}-{3+(ik+1)*len(Ns)}}}')
            file.write('\n')
            file.write('& &')
            for K in Ks:
                for N in Ns:
                    file.write(f' & $N$ = {N}')
            file.write('\\\\\n')
            for algo in self.algos:
                self.write_algo_in_table(file,algo,Ks,Ns,n_cols,features,feature_names,bold_numbers)
            file.write('\\end{tabular}\n\\end{table}')

    def write_algo_in_table(self,file,algo,Ks,Ns,n_cols,features,feature_names,bold_numbers):
        file.write('\\midrule\n')
        file.write(f'\\multirow{{{3*len(self.data_parameters)}}}{{*}}{{\\raisebox{{-2\\height}}{{\\rotatebox[origin=c]{{90}}{{\\makebox[0pt][c]{{\\Large{{\\textbf{{{algo.legend}}}}}}}}}}}}}')
        for a,dataparam_title in enumerate(self.data_parameters_titles):
            file.write(f'& \\multirow{{{len(features)}}}{{*}}{{\\begin{{tabular}}{{c}} {dataparam_title} \\end{{tabular}}}}')
            for idx,feature in enumerate(features):
                if idx > 0:
                    file.write('& ')
                file.write('& ' + feature_names[feature])
                for ik,_ in enumerate(Ks):
                    for jn,_ in enumerate(Ns):
                        value = algo.results[feature][a,ik,jn]
                        bold = False
                        exp_notation = True
                        if feature in ['mean_jisi','mean_times']:
                            bold = bold_numbers[(feature,algo)][a,ik,jn]
                        if feature in ['mean_updates','mean_times']:
                            exp_notation = False
                        self.write_in_table(file,value,bold,exp_notation)
                file.write('\\\\\n')
            if a == len(self.data_parameters)-1:
                file.write('\\bottomrule\n')
                file.write('\\\\\n')
            else:
                file.write(f'\\cmidrule(lr){{2-{3+n_cols}}}')

    def write_in_table(self,file,value,bold=False,exp_notation=True):
        fmt = '.2E' if exp_notation else '.1f'
        if bold:
            file.write(f' & \\textbf{{{value:{fmt}}}}')
        else:
            file.write(f' & {value:{fmt}}')

    def draw_empirical_convergence(self,a,K,N,res_type='costs',mode='time',algos=None):
        if algos == None:
            algos = self.algos
        # xlabels = {'time':'Time (s)','iter':'Nb of iterations'}
        # ylabels = {'jisi':'jISI score','costs':'cost function','diffs':'criteria'}
        output_path_individual = self.output_folder+ f'/{self.data_parameters_titles[a]}/N_{N}_K_{K}'
        fig,ax = plt.subplots()
        ax.set_yscale('log')
        for algo in algos:
            algo.fill_from_folder(output_path_individual)
            values = algo.results[res_type]
            if mode == 'time':
                if 'detailed' in res_type:
                    times = algo.results['detailed_times']
                else:
                    times = algo.results['times']              
                ax.plot(times,values,color=algo.color,label=algo.legend,linewidth=1)
            else:
                ax.plot(values,color=algo.color,label=algo.legend,linewidth=1)
            ax.legend(loc=1,fontsize=self.legend_fontsize)
        # ax.set_xlabel(xlabels[mode],fontsize=20)
        # ax.set_ylabel(ylabels[res_type],fontsize=20)
        ax.set_title('Empirical convergence',fontsize=self.title_fontsize)
        for extension in ['eps','png']:
            fig_path = os.path.join(output_path_individual, res_type + '_' + mode)
            os.makedirs(os.path.dirname(fig_path), exist_ok=True)
            fig.savefig(fig_path,dpi=200,format=extension)
            plt.show()
            
            
#====== FONCTIONS POUR REPORT LES EXPERIENCES D'UNROLLING ======     
ALGO_NAME_MAP = {
    "titan": "PALM-IVA-G",
    "UTitan_tied": "U-PALM-IVA-G-tied",
    "UTitan_untied": "U-PALM-IVA-G-untied",
    "UTitan_inertial-tied": "U-PALM-IVA-G-tied-inertial",
    "UTitan_inertial-untied": "U-PALM-IVA-G-untied-inertial",
}

ALGO_ORDER = [
    "titan",
    "UTitan_tied",
    "UTitan_untied",
    "UTitan_inertial-tied",
    "UTitan_inertial-untied",
]

def collect_results(experiment_paths):
    results = {}
    for exp_path in experiment_paths:
        res_path = os.path.join(exp_path, "res")
        cases = os.listdir(res_path)
        for case in cases:
            case_path = os.path.join(res_path,case)
            dims = os.listdir(case_path)
            for dim in dims:
                dim_path = os.path.join(case_path,dim)
                if case not in results:
                    results[case] = {}
                if dim not in results[case]:
                    results[case][dim] = {}
                for algo in ALGO_ORDER:
                    jisi_path = os.path.join(dim_path, f"{algo}_final_jisi")
                    time_path = os.path.join(dim_path, f"{algo}_total_times")
                    # print(f'metrics for {dim_path} and algo {algo}: ')
                    jisi_mean, jisi_std = read_metric(jisi_path)
                    time_mean, _ = read_metric(time_path)
                    results[case][dim][algo] = {"jisi_mean": jisi_mean,"jisi_std": jisi_std,"time_mean": time_mean,}
    return results

def read_metric(path):
    result = np.fromfile(path, sep=',')
    mean = np.nanmean(result)
    std = np.nanstd(result)
    return mean, std

def select_best(results,tol_time=1e-2,tol_jisi=1e-4):
    for case in results.keys():
        for dim in results[case].keys():
            min_jisi = np.inf
            min_time = np.inf
            for algo in results[case][dim].keys():
                min_jisi = min(min_jisi,results[case][dim][algo]['jisi_mean'])
                min_time = min(min_time,results[case][dim][algo]['time_mean'])
            for algo in results[case][dim].keys():
                results[case][dim][algo]['is_best_jisi'] = results[case][dim][algo]['jisi_mean'] <= min_jisi + tol_jisi
                results[case][dim][algo]['is_best_time'] = results[case][dim][algo]['time_mean'] <= min_time + tol_time
                

def format_scientific(x, precision=2):
    # print(f'{x:.{precision}e}')
    mantissa, exponent = f"{x:.{precision}e}".split("e")
    exponent = int(exponent)
    return rf"{mantissa} \times 10^{{{exponent}}}"

def format_jisi(mean,std,bold=False):
    mean_str = format_scientific(mean)
    std_str = format_scientific(std)
    formatted = rf"{mean_str} \pm {std_str}"
    if bold:
        result = rf"$\boldsymbol{{{formatted}}}$"
    else:
        result = rf"${formatted}$"
    return result

def format_time(mean, bold=False, precision=2):
    result = rf"{mean:.{precision}f}"
    if bold:
        result = rf"$\boldsymbol{{{result}}}$"
    else:
        result = rf"${{{result}}}$"
    return result

def report_table(results,output_path):
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\renewcommand{\arraystretch}{1.7}")
    lines.append(r"\resizebox{0.8\linewidth}{!}{")
    lines.append(r"\begin{tabular}{clcccc}")
    lines.append(r"\toprule")
    # Header dynamique
    dims = sorted(list(next(iter(results.values())).keys()),key=lambda dim: int(dim.split("_")[3]))
    header_1 = r"\multirow{2}{*}{\textbf{Dataset}} & \multirow{2}{*}{\textbf{Algorithm}}"
    for dim in dims:
        K = dim.split("_")[3]
        N = dim.split("_")[1]
        header_1 += rf" & \multicolumn{{2}}{{c}}{{\textbf{{$ (K,N)=({K},{N}) $}}}}"
    header_1 += r" \\"
    lines.append(header_1)
    # cmidrule
    cmid = r""
    col = 3
    for _ in dims:
        cmid += rf"\cmidrule(lr){{{col}-{col+1}}} "
        col += 2
    lines.append(cmid)
    header_2 = r"& "
    header_2 += r"& " + " & ".join([r"$\mu_{\rm jISI} \pm \sigma_{\rm jISI}$ & $\mu_\texttt{T}(s)$"] * len(dims))
    header_2 += r" \\"
    lines.append(header_2)
    lines.append(r"\midrule")
    # Corps du tableau
    for case in results.keys():
        algos = list(results[case][dims[0]].keys())
        lines.append(rf"\multirow{{{len(algos)}}}{{*}}{{ $\D^{{\rm {case}}}$ }}")
        for i,algo in enumerate(ALGO_ORDER):
            algo_name = ALGO_NAME_MAP[algo]
            line = "& " + rf"\textbf{{{algo_name}}}"
            for dim in dims:
                res = results[case][dim][algo]
                jisi = format_jisi(res["jisi_mean"],res["jisi_std"],res["is_best_jisi"],)
                time = format_time(res["time_mean"],res["is_best_time"],)
                line += f"& {jisi} & {time} "
            line += r"\\"
            lines.append(line)
        lines.append(r"\midrule")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\caption{Results}")
    lines.append(r"\label{tab:results}")
    lines.append(r"\end{table}")
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
        

experiment_paths = [
'Result_data/experiments/2026-03-18_11-13_Testing_unrolling',
'Result_data/experiments/2026-04-17_16-08_Unrolling_comparison_small',
'Result_data/experiments/2026-04-19_18-45_Unrolling_comparison_D_small',
'Result_data/experiments/2026-04-19_19-17_Unrolling_comparison_D_big'
]

results = collect_results(experiment_paths)
select_best(results)
report_table(results, "table_results.tex")