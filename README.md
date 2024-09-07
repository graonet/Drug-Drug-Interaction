# Drug-Drug Interaction Prediction with SWIN Transformer

This repository contains two Jupyter notebooks for predicting drug-drug interactions (DDIs). The project leverages data preprocessing techniques and the SWIN Transformer model to train and evaluate DDI predictions.

## Project Overview

Drug-drug interactions (DDIs) can cause adverse effects, making it important to predict potential interactions between drugs. This project aims to preprocess large datasets of drug-related information and use a SWIN Transformer for training and evaluation of drug-drug interaction predictions.

The project consists of two main phases:
1. **Data Preprocessing** using `Prepare_Big_Datasets.ipynb`.
2. **Model Training and Evaluation** using `SWIN-TR-86.ipynb` with a SWIN Transformer architecture.

## Notebooks

### 1. Prepare_Big_Datasets.ipynb
This notebook handles the data preprocessing for large-scale datasets used in drug-drug interaction studies. It includes:
- Loading raw drug interaction datasets.
- Cleaning and transforming the data for input to the model.
- Feature extraction for drug characteristics.

### 2. SWIN-TR-86.ipynb
This notebook trains and evaluates a SWIN Transformer model for predicting DDIs. It includes:
- Loading preprocessed datasets.
- Defining the SWIN Transformer architecture.
- Training the model on the prepared dataset.
- Evaluating the model’s performance using metrics such as accuracy, precision, and recall.
- Visualizing the performance of the model.

## Data

The dataset used for training and evaluation is large and specific to the drug-drug interaction domain. Due to its size and sensitivity, the dataset can be provided upon request. Please reach out if you are interested in accessing the data for replication or research purposes.

## Requirements

To run the notebooks, ensure you have the following Python libraries installed:

- `numpy`
- `pandas`
- `scikit-learn`
- `tensorflow` or `pytorch` (depending on the implementation in the notebooks)
- `matplotlib`
- `seaborn`

Install the required libraries using pip:

```bash
pip install numpy pandas scikit-learn tensorflow matplotlib seaborn
