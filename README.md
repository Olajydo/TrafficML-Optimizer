# Traffic Congestion Prediction Using Neural Networks

## Overview
This project focuses on predicting **traffic congestion** using machine learning, particularly **deep neural networks (DNNs) in TensorFlow**. The model learns to identify patterns in traffic flow to improve road safety and efficiency. It involves **feature engineering, model training, and hyperparameter tuning** to optimize performance.

## Dataset
The dataset used is `smart_traffic_system_data.csv`, which contains various traffic-related features and a target variable, **Congestion** (1 for congested traffic, 0 for no congestion).

## Project Workflow
1. **Data Preprocessing:**
   - Load dataset using Pandas.
   - Split data into training and testing sets.
   - Standardize features using `StandardScaler`.

2. **Model Development:**
   - Define a **deep neural network (DNN)** using `TensorFlow Keras`.
   - Implement multiple layers with **ReLU activations** and a **sigmoid output layer** for binary classification.
   - Train the model using the **Adam optimizer** and **binary cross-entropy loss**.

3. **Cross-Validation & Early Stopping:**
   - Use **Stratified K-Fold Cross-Validation** to ensure model robustness.
   - Implement **early stopping** to prevent overfitting.

4. **Hyperparameter Tuning:**
   - Experiment with different **batch sizes, number of epochs, optimizers (Adam, RMSprop), and activation functions (ReLU, Tanh)**.
   - Select the best combination based on **validation accuracy**.

5. **Model Evaluation:**
   - **Confusion Matrix:** Visualizes correct and incorrect predictions.
   - **ROC Curve:** Measures model's ability to distinguish between congestion and non-congestion.
   - **Loss & Accuracy Graphs:** Show training progress over epochs.
   - **Actual vs. Predicted Plot:** Compares predictions with actual values.

## Key Findings
- **Optimized Hyperparameters:**
  - Best **batch size**: `{best_batch_size}`
  - Best **epochs**: `{best_epochs}`
  - Best **optimizer**: `{best_optimizer}`
  - Best **activation function**: `{best_activation}`

- **Performance Summary:**
  - **Validation Accuracy:** `{best_validation_accuracy:.4f}`
  - **Final Model Accuracy:** `{final_model_accuracy:.4f}`
  - **False Negatives:** 4 cases where congestion was missed
  - **False Positives:** 0 cases where congestion was wrongly predicted

## Visualizations
- **Confusion Matrix:** Shows model performance on test data.
- **ROC Curve:** Demonstrates classification effectiveness.
- **Training Loss & Accuracy Graphs:** Insights into model learning.
- **Actual vs Predicted Congestion Plot:** Evaluates real-world applicability.

## How to Run
1. Install dependencies:  
   ```bash
   pip install tensorflow pandas scikit-learn matplotlib seaborn
   ```
2. Check the output results and visualizations.

---
**Author:** Yusuf Olajide  
**Date:** March 2025

