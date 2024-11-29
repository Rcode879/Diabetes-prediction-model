# Diabetes Prediction Model

This project implements a machine learning pipeline to predict diabetes using the **Pima Indians Diabetes Dataset**. The dataset contains information about various health indicators for Pima Indian women, along with whether or not they have been diagnosed with diabetes. A neural network model is built and trained to predict diabetes using this data.

---

## About Diabetes

Diabetes is a chronic health condition that affects how the body processes blood sugar (glucose). If untreated, it can lead to serious complications like heart disease, kidney damage, and nerve damage. Early detection is crucial, and machine learning can play a significant role in identifying individuals at risk of developing diabetes.

---

## Dataset: Pima Indians Diabetes Dataset

The Pima Indians Diabetes Dataset contains 768 samples of data with the following features:

- **Pregnancies**: Number of times the patient has been pregnant.
- **Glucose**: Plasma glucose concentration.
- **BloodPressure**: Diastolic blood pressure (mm Hg).
- **SkinThickness**: Triceps skin fold thickness (mm).
- **Insulin**: 2-hour serum insulin (mu U/ml).
- **BMI**: Body Mass Index (weight in kg/(height in m²)).
- **DiabetesPedigreeFunction**: A function that scores the likelihood of diabetes based on family history.
- **Age**: Patient's age (years).
- **Outcome**: Target variable (1 if diabetic, 0 otherwise).

The dataset is widely used in machine learning and contains both features for prediction and a binary target variable (`Outcome`).

---

## Features of the Script

1. **Dataset Loading**  
   - The dataset is loaded from the `diabetes.csv` file.  
   - This script assumes the file is present in the same directory as the script.

2. **Visualization**  
   - Histograms of all numerical columns in the dataset are generated.  
   - These provide insights into the distribution of features.
   ![Dataframe plotted on a histogram](histogram.png)

3. **Data Preprocessing**  
   - Missing or zero values in key columns (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`) are replaced with their column means.  
   - All features are standardized using `sklearn.preprocessing.scale` for better model performance.

4. **Data Splitting**  
   - The dataset is split into:
     - **Training set**: 64% of the data.
     - **Validation set**: 16% of the data (split from the training set).
     - **Test set**: 20% of the data.

5. **Neural Network Model**  
   - The neural network is built using Keras' `Sequential` API.  
   - **Architecture**:
     - **Input Layer**: 8 input features.
     - **Hidden Layers**:
       - First layer: 32 nodes, ReLU activation.
       - Second layer: 16 nodes, ReLU activation.
     - **Output Layer**: 1 node, sigmoid activation (binary output for classification).
   - The model is compiled with:
     - **Optimizer**: Adam.
     - **Loss Function**: Binary cross-entropy.
     - **Metric**: Accuracy.

6. **Training**  
   - The model is trained on the training set for 200 epochs.

7. **Evaluation**  
   - The model is evaluated on the test set.  
   - The script outputs the testing accuracy after evaluation.

---

## How to Run the Script

1. **Prepare the Dataset**  
   - Download the Pima Indians Diabetes Dataset (`diabetes.csv`) from [Kaggle](https://www.kaggle.com/uciml/pima-indians-diabetes-database) or any other reliable source.  
   - Place the dataset in the same directory as the script.

2. **Run the Script**  
   - Execute the script in your terminal or IDE:  
     ```bash
     python main.py
     ```

3. **View Results**  
   - The script will:
     - Display histograms of the dataset for visualization.
     - Train the neural network.
     - Output the testing accuracy, for example:
       ```
       testing accuracy: 88.50%
       ```

---

## Customization Options

- **Neural Network Architecture**:  
  - Modify the number of layers or nodes in the `model` definition.  
  - Experiment with different activation functions.

- **Hyperparameters**:  
  - Adjust the optimizer, loss function, or number of epochs.

- **Preprocessing**:  
  - Experiment with different strategies for handling missing values (e.g., median imputation).

---

## Example Output

- **Histograms**:  
  - Visualize dataset feature distributions.  
- **Final Testing Accuracy**:  
  - Example:
    ```
    testing accuracy: 88.50%
    ```

---

## Limitations and Assumptions

1. **Data Quality**:  
   - Missing values in certain columns (e.g., `Glucose`, `BloodPressure`) are assumed to be zeros and are replaced with the column mean.

2. **Feature Engineering**:  
   - No additional feature engineering is performed beyond scaling and imputation.

3. **Model Generalization**:  
   - Performance is limited to the quality and size of the dataset. Results may vary with different data.

---

## Results Interpretation

The trained model predicts whether an individual is diabetic based on their health metrics. The accuracy of the model on the test dataset indicates its predictive performance. Higher accuracy reflects better model generalization.

---

## References

- **Dataset Source**: [UCI Machine Learning Repository - Pima Indians Diabetes Database](https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes)
- **Documentation**:
  - [Scikit-learn Documentation](https://scikit-learn.org/)
  - [Keras Documentation](https://keras.io/)

---

## License

This project is licensed under the MIT License. You are free to use, modify, and distribute the code with proper attribution.
