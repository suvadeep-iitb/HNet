# A Hierarchical Deep Learning Model for Side-Channel Analysis

This repository contains the implementation of a hierarchical deep learning model for Side-Channel Analysis ([paper](https://eprint.iacr.org/2024/1437)).

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
on dataset \<dataset\> where \<dataset\> is one of ASCADf ([ASCAD fixed key](https://github.com/ANSSI-FR/ASCAD/tree/master/ATMEGA\_AES\_v1/ATM\_AES\_v1\_fixed\_key)), ASCADr ([ASCAD random key](https://github.com/ANSSI-FR/ASCAD/tree/master/ATMEGA\_AES\_v1/ATM\_AES\_v1\_variable\_key)) and CHES20 ([CHES CTF 2020](https://ctf.spook.dev/)).


## Data Pre-processing:
* The traces of the CHES CTF 2020 dataset have been multiplied by the constant 0.004 to keep the range of the feature values within [-120, 120].

## Citation:
```
@misc{cryptoeprint:2024/1437,
      author = {Suvadeep Hajra and Debdeep Mukhopadhyay},
      title = {{HierNet}: A Hierarchical Deep Learning Model for {SCA} on Long Traces},
      howpublished = {Cryptology {ePrint} Archive, Paper 2024/1437},
      year = {2024},
      url = {https://eprint.iacr.org/2024/1437}
}
```
