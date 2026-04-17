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