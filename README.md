# HierNet: A Hierarchical Deep Learning Model Built on the UNIDLE Framework for Side-channel Analysis

This repository contains the implementation of **HierNet**, proposed in [this paper](https://eprint.iacr.org/2024/1437).

---

## Repository Structure

The implementation is composed of the following files:
- **`fast_attention.py`** – Implements the proposed GaussiP attention layer.
- **`self_attention.py`** – Implements the self-attention module used in the second-level transformer layer.
- **`normalization.py`** – Implements the layer-centering normalization layer.
- **`transformer.py`** – Defines the HierNet model architecture.
- **`train_trans.py`** – Training and evaluation script for the HierNet model.
- **`data_utils.py`** – Utilities for loading ASCADf and ASCADr datasets.
- **`data_utils_ches20.py`** – Utilities for loading the CHES20 dataset.
- **`evaluation_utils.py`** – Computes guessing entropy for ASCAD datasets.
- **`evaluation_utils_ches20.py`** – Computes guessing entropy for the CHES20 dataset.
- **`run_trans_\<dataset\>.sh`** – Bash scripts with predefined hyperparameters for running experiments on specific datasets, where `<dataset>` is one of:
  - **ASCADf** ([fixed key](https://github.com/ANSSI-FR/ASCAD/tree/master/ATMEGA_AES_v1/ATM_AES_v1_fixed_key))
  - **ASCADr** ([random key](https://github.com/ANSSI-FR/ASCAD/tree/master/ATMEGA_AES_v1/ATM_AES_v1_variable_key))
  - **CHES20** ([CHES CTF 2020](https://ctf.spook.dev/))

---

## Data Pre-processing:

- For the **CHES CTF 2020** dataset, the traces are multiplied by a constant `0.004` to normalize the feature values within the range **[-120, 120]**.

---

## Tested on
- Python 3.8.10  
- absl-py == 2.3.1 
- numpy == 1.24.3
- scipy == 1.10.1
- h5py == 3.11.0
- tensorflow == 2.13.0

---

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/suvadeep-iitb/HNet.git
   cd HNet
   ```
2. **Install dependencies (Python >= 3.8 recommended):**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set dataset path in the bash script:**
   ```
   Open run_trans_\<dataset\>.sh and set the dataset path variable properly.
   ```
4. **Train HierNet:**
   ```bash
   bash run_trans_\<dataset\>.sh train
   ```
5. **Perform Evaluation:**
   ```bash
   bash run_trans_\<dataset\>.sh test
   ```

---

## Citation:
```
@misc{cryptoeprint:2024/1437,
      author = {Suvadeep Hajra and Debdeep Mukhopadhyay and Soumi Chatterjee},
      title = {{UNIDLE}: A Unified Framework for Deep Learning-based Side-channel Analysis},
      howpublished = {Cryptology {ePrint} Archive, Paper 2024/1437},
      year = {2024},
      url = {https://eprint.iacr.org/2024/1437}
}
```
