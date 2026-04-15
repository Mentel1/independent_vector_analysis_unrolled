# Independent Vector Analysis Unrolled
   
This package contains the Pytorch implementations of the architecture, datasets, and training/reporting pipelines for U-TITAN-IVA-G. [article to come]
<!-- This package contains the Python versions of IVA-G [1] and IVA-L-SOS [2], converted from the [MLSP-Lab MATLAB Codes](http://mlsp.umbc.edu/resources.html).

- **Website:** http://mlsp.umbc.edu/jointBSS_introduction.html
- **Source-code:** https://github.com/SSTGroup/independent_vector_analysis -->


## Installing independent_vector_analysis_unrolled

<!-- The only pre-requisite is to have **Python 3** (>= version 3.6) installed.
The iva package can be installed with

    pip install independent_vector_analysis

Required third party packages will automatically be installed. -->


## Package description

This package is organized as follows: 

The subfolder "Result_data" contains all the data that needs to be made persistant. In particular, this is where we store datasets, model parameters and the reporting of their training, and the results from the experiments.

The subfolder "Algorithms" contains the code to run the iterative versions of TITAN-IVA-G and IVA-G, in a pytorch implementation, as well as all the useful helpers.

The subfolder "runs" contains the figures and data used for Tensorboard monitoring.

The subfolder "TITAN_Unrolled" contains the classes of datasets, models, and training pipeline for processing UTitan architectures.

### Naming path conventions

The names "Case_A,...,Case_D" refer to the parameters used to generate the data, we can have datasets that do not purely belong to one of these cases but then we will come up with new names for them. The data is also characterized by the dimensions (N,K), so we can name "{data_case}/N_{N}_K_{K}" the data_path.

The training is characterized by the optimizer

A model subfolder (containing its parameters and all the reporting data of its training) should be named after its own characteristics (architecture/number_of_layers)
But the total path to reach a model should also identify:
the characteristics of the data it can process/it has been trained on
and the characteristics of the training it followed.
So we store each model in a folder with path composed as follows: {models_folder}/{data_path}/{training_path}/{architecture_path}.
Likewise, a dataset is characterized with the type of data it contains and its size/function. It is stored at an address composed as {datasets_folder}/{data_path}/{function}.

We cannot at the same time have reasonably short paths and paths that contain exhaustive information about the models/datasets. If we are sure that the information displayed in the path identifies at most one model, the path may be not fully informative, but at least we will not have conflicts of addresses, and we can store a config file with the model containing all the information we need to know. 
However, if conflicts are possible, and we still do not want to extend the length of the paths, there is the possibility of using hash-codes to embedd all the config information in a short code (as it is done in the git projects for instance). The problem with this approach is that it makes it harder for the user to explore the file system by hand...

I need to sort this out with competent architects, for now, I will just define informative path such that there will be no conflicts for the usage that covers my dissertation.

## Contact

If you have any questions, you can contact me at clement.cosserat@gmail.com

## Citing

If you use this software, please cite... 

