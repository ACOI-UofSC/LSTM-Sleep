# Sleep Stage Prediction using LSTM

This repository contains Python scripts for predicting sleep stages using an LSTM model. The workflow involves preprocessing raw data, generating features, and training the model to classify sleep stages.

## Project Structure
```
./
|-- data/
|   |-- raw_data/              # Directory for raw input files
|   |-- processed_data/        # Directory for processed data
|   |-- ...                    # Other data files
|
|-- preprocessing/
|   |-- preprocessing_raw.py   # Script for processing raw input files
|   |-- preprocessing_features.py # Script for generating features from processed data
|   |-- ...                    # Other preprocessing scripts
|
|-- source/
|   |-- analysis_runner_weighted_split_torch.py # Script for model training and prediction
|   |-- ...                    # Other source files
|
|-- README.md                  # Project documentation
```

## How to Use
Follow the steps below to prepare the data, preprocess it, and train the LSTM model.

### Step 1: Prepare Raw Input Files
1. Place the raw input files containing **motion data** and **PSG labels** into the `./data/raw_data/` directory.
2. Ensure that the input files are in the correct format, including all necessary device motion data and corresponding PSG labels.

### Step 2: Preprocess Raw Data
Run the following script to process the raw input files:
```bash
python preprocessing/preprocessing_raw.py
```
This will clean and process the data, generating intermediate outputs ready for feature extraction.

### Step 3: Generate Features
Run the following script to extract features from the processed data:
```bash
python preprocessing/preprocessing_features.py
```
This step converts the processed data into features that can be fed into the LSTM model.

### Step 4: Train and Predict Using the LSTM Model
Run the following script to train the LSTM model and make predictions:
```bash
python source/analysis_runner_weighted_split_torch.py
```
The script will train the LSTM model using the generated features and produce predictions for sleep stages.


## Output
The output includes the predictions for the input data.

## Notes
- Make sure the input files in `data/raw_data/` are correctly formatted before running the preprocessing scripts.



