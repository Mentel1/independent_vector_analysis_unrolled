
# Function to generate dataparameters for the multiparameter experiment
def get_dataparameters(rhos,lambdas):
    dataparameters_multiparam = []
    for rho_bounds in rhos:
        for lambda_ in lambdas:
            dataparameters_multiparam.append((rho_bounds,lambda_))
    return dataparameters_multiparam

lambda_1 = 0.04
lambda_2 = 0.25
rho_bounds_1 = [0.2,0.3]
rho_bounds_2 = [0.6,0.7]
rhos = [rho_bounds_1] #,rho_bounds_2]
lambdas = [lambda_1] #,lambda_2]
dataparameters_multiparam = get_dataparameters(rhos,lambdas)
dataparameters_titles_multiparam = ['Case_A'] #,'Case_B','Case_C','Case_D']


dataparam_dict = {'Case_A':{'rho_bounds':rho_bounds_1,'lambda':lambda_1},'Case_B':{'rho_bounds':rho_bounds_1,'lambda':lambda_2},'Case_C':{'rho_bounds':rho_bounds_2,'lambda':lambda_1},'Case_D':{'rho_bounds':rho_bounds_2,'lambda':lambda_2}}

data_corruption = {}