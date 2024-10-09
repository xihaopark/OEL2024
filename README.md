# OEL: Osaka University Electricity Load Forecasting

Welcome to the **OEL (Osaka University Electricity Load Forecasting)** project! This repository contains the code and resources for forecasting electricity loads using a unified Transformer-based model. Below you will find an overview of the project structure, data processing pipeline, model architecture, training and evaluation procedures, and visualization of results.

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
- [Visualization](#visualization)
- [Algorithm Design](#algorithm-design)
- [Usage](#usage)
- [Dependencies](#dependencies)

## Introduction

The **OEL** project aims to accurately forecast electricity load using data from multiple channels and various temporal aggregations (daily, weekly, monthly, yearly). 
The model learns comprehensive representations that leverage the shared feature space across these tasks by unifying different input sub-tasks into a single framework.

## Project Structure

```
OEL/
├── data/
│   └── OEL_all.csv
├── prediction_results/
│   ├── prediction_results_test.csv
│   ├── prediction_results_A_test.csv
│   ├── prediction_results_B_test.csv
│   ├── prediction_results_C_test.csv
│   ├── prediction_results_D_test.csv
│   ├── actual_labels_sample.csv
│   ├── predicted_values_sample.csv
│   └── plots/
│       ├── plot_0.png
│       ├── plot_1.png
│       ├── plot_2.png
│       ├── plot_3.png
│       └── plot_4.png
├── main.ipynb
├── README.md
```

- **`main.ipynb`**: The primary notebook containing the complete workflow, including data processing, model training, evaluation, and visualization.
- **`data/`**: Directory to store the input data file.
  - **`OEL_all.csv`**: The main dataset used for training and testing.
- **`prediction_results/`**: Directory where all prediction outputs and visualizations are saved.
  - **`prediction_results_test.csv`**: Comprehensive CSV file containing predicted and actual values for all channels on the test set.
  - **`prediction_results_A_test.csv`**, **`prediction_results_B_test.csv`**, **`prediction_results_C_test.csv`**, **`prediction_results_D_test.csv`**: CSV files containing predicted and actual values segmented by different levels (A, B, C, D).
  - **`actual_labels_sample.csv`**: Sample of actual labels for further inspection.
  - **`predicted_values_sample.csv`**: Sample of predicted values for further inspection.
  - **`plots/`**: Sub-directory containing visualization plots for different channels.
    - **`plot_0.png`**, **`plot_1.png`**, ... : PNG files visualizing actual vs. predicted values for specific channels.

## Data Preparation

1. **Create the Data Directory**:
   
   Before running the project, ensure that a `data` directory exists in the root of the repository. If not, create it:

   ```bash
   mkdir data
   ```

2. **Add the Dataset**:
   
   Place the dataset file `OEL_all.csv` into the `data` directory. 

## Model Architecture

The project utilizes a **Transformer-based** architecture to handle multiple sub-tasks with unified representations. The architecture comprises:

- **Transformer Encoder**: Encodes the input data into a latent representation.
- **Transformer Decoder for Reconstruction (`DecoderRecon`)**: Reconstructs a portion of the input data, aiding in learning a unified representation through reconstruction loss.
- **Transformer Decoder for Prediction (`DecoderPred`)**: Predicts the future electricity load based on the latent representation.

### Positional Encoding

To retain the order information of the input sequence, positional encoding is added to the input embeddings, enabling the Transformer to capture temporal dependencies effectively.

## Training and Evaluation

1. **Data Processing**:
   
   - Load and preprocess the data from `OEL_all.csv`.
   - Generate aggregated datasets (daily, weekly, monthly, yearly maximums) using resampling techniques.
   - Normalize the data using `MinMaxScaler` to scale features between 0 and 1.

2. **Generating Dataloader**:
   
   - Prepare input sequences and corresponding targets for each level (A, B, C, D).
   - Split the dataset into training (80%) and testing (20%) sets, ensuring each level's samples are proportionally represented.
   - Create PyTorch `DataLoader` instances for efficient batching and shuffling during training.

3. **Model Construction**:
   
   - Instantiate the Transformer-based Encoder and Decoders.
   - Define loss functions (`MSELoss`) for training.

4. **Training Process**:
   
   - **Encoder Training**: Train the Transformer Encoder and `DecoderRecon` using reconstruction loss to learn a unified latent representation.
   - **Decoder Training**: Freeze the Encoder's parameters and train `DecoderPred` to perform the prediction task based on the learned latent representation.

5. **Evaluation**:
   
   - Test the trained model on the test set.
   - Compute the average loss on the test set.
   - Inverse transform the normalized predictions and actual values back to the original scale for interpretation.

## Results

### Prediction Outputs

- **Comprehensive Predictions**:
  
  The file `prediction_results_test.csv` contains the predicted and actual values for all channels on the test set. Each channel's predictions are labeled sequentially (e.g., `Predicted_0`, `Actual_0`, ..., `Predicted_N`, `Actual_N`).

- **Level-specific Predictions**:
  
  Separate CSV files (`prediction_results_A_test.csv`, `prediction_results_B_test.csv`, etc.) store the predictions and actual values segmented by different levels (A, B, C, D).

### Visualization

- **Channel-wise Prediction Plots**:
  
  The `prediction_results/plots/` directory contains PNG files (`plot_0.png`, `plot_1.png`, etc.) that visualize the actual vs. predicted values for the first five channels. Each plot displays the comparison over the first 100 samples of the test set, providing a clear visual assessment of the model's performance.

## Visualization

The project includes visualizations to assess the performance of the model across different channels:

1. **Channel-wise Plots**:
   
   - Located in the `prediction_results/plots/` directory.
   - Each plot (`plot_0.png`, `plot_1.png`, etc.) shows the actual and predicted electricity load values for a specific channel over the first 100 samples of the test set.

2. **Level-specific Results**:
   
   - The level-specific CSV files (`prediction_results_A_test.csv`, etc.) provide detailed predictions and actual values for each hierarchical level, allowing for granular analysis.

## Algorithm Design

The architecture and workflow of the OEL project are meticulously designed to address the complexities of electricity load forecasting through the following considerations:

1. **Unified Handling of Multiple Sub-tasks**:
   
   - **Objective**: Manage different temporal aggregations (daily, weekly, monthly, yearly) within a single framework.
   - **Implementation**: Use a unified Transformer-based Encoder that learns comprehensive representations from varied input sequences corresponding to different levels.

2. **Comprehensive Representation Learning**:
   
   - **Objective**: Enable the model to capture intricate patterns and dependencies across various temporal scales.
   - **Implementation**: The Encoder, trained with a reconstruction loss, ensures that the latent representation encapsulates essential features from all sub-tasks, facilitating better generalization.

3. **Shared Feature Space Across Sub-tasks**:
   
   - **Objective**: Leverage the inherent similarities in the feature space across different temporal aggregations.
   - **Implementation**: Although each sub-task (level) has different input sequence lengths, they originate from the same dataset, ensuring their features lie within a common feature space. This allows the reconstruction loss to guide the learning of a unified representation effectively.

4. **Flexible and Scalable Decoder**:
   
   - **Objective**: Facilitate downstream tasks such as load prediction without redesigning the entire model.
   - **Implementation**: A common Transformer-based Decoder (`DecoderPred`) utilizes the shared latent representation to perform predictions, ensuring consistency and scalability.

## Usage

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/yourusername/OEL.git
   cd OEL
   ```

2. **Prepare the Data**:

   - Create the `data` directory if it doesn't exist:

     ```bash
     mkdir data
     ```

   - Place the `OEL_all.csv` dataset into the `data` directory.

3. **Run the Main Notebook**:

   Open and run `main.ipynb` using Jupyter Notebook or JupyterLab:

   ```bash
   jupyter notebook main.ipynb
   ```

   The notebook will execute the entire workflow, including data preprocessing, model training, evaluation, and visualization.

## Dependencies

The project relies on the following Python libraries:

- `numpy`
- `pandas`
- `torch`
- `scikit-learn`
- `matplotlib`

