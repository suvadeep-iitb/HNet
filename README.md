# A Hierarchical Transformer Network for SCA

This repository contains the implementation of HierNet, a hierarchical transformer network for Side Channel Analysis.

The implementation is composed of the following files:
* **fast_attention.py:** It contains the code of the proposed GaussiP attention layer.
* **self_attention.py:** It contains the code of the self-attanetion used in the second-level transformer layer.
* **normalization.py:** It contains the code of layer-centering layer.
* **transformer.py:** It contains the code of the HierNet model.
* **train_trans.py** It contains the code for training and evaluating the EstraNet model.
* **data_utils.py:** It contains the code for reading data from the ASCADf or ASCADr dataset.
* **data_utils_ches20.py:** It contains the code for reading data from the CHES20 dataset.
* **evaluation_utils.py:** It contains the code for computing the guessing entropy for the ASCAD datasets.
* **evaluation_utils_ches20.py:** It contains the code for computing the guessing entropy for the CHES20 dataset.
* **run_trans_\<dataset\>.sh:** It is the bash script with proper hyper-parameter setting to perform experiments 
on dataset \<dataset\> where \<dataset\> is one of ASCADf, ASCADr and CHES20.
