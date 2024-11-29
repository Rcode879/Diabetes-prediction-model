# Diabetes-prediction-model
5. **Neural Network Model**  
   - A neural network is created using Keras' Sequential API. The architecture is as follows:  
     - **Input Layer**: Accepts 8 features from the dataset.  
     - **Hidden Layers**:  
       - First layer: 32 nodes with ReLU activation.  
       - Second layer: 16 nodes with ReLU activation.  
     - **Output Layer**: 1 node with sigmoid activation for binary classification.  
   - Model compilation is done with:  
     - Optimizer: `Adam`  
     - Loss function: `binary_crossentropy`  
     - Metric: `accuracy`  

6. **Training**
   - The model is trained on the training dataset for 200 epochs.

7. **Evaluation**
   - The model is evaluated on the test dataset, and the testing accuracy is displayed.

---

## How to Run the Script

1. **Prepare the Dataset**  
   - Download the Pima Indians Diabetes Dataset (`diabetes.csv`) from [Kaggle](https://www.kaggle.com/uciml/pima-indians-diabetes-database) or any reliable source.  
   - Place the dataset in the same directory as the script.  

2. **Run the Script**  
   - Execute the script in your terminal or IDE:  
     ```bash
     python script_name.py
     ```

3. **View Results**  
   - The script will:  
     - Display histograms of the data for visualization.  
     - Train the neural network.  
     - Output the testing accuracy of the model, e.g.:  
       ```
       testing accuracy: <value>
       ```

---

## Customization Options

- **Neural Network Architecture**:  
  Modify the structure by adding/removing layers in the `model` definition section. You can also adjust the number of nodes or activation functions.  

- **Hyperparameters**:  
  Adjust parameters such as the optimizer, loss function, or number of epochs for training.  

- **Preprocessing**:  
  Experiment with different strategies for handling missing or zero values, such as median imputation instead of mean.  

---

## Example Output

- Histograms of the dataset features will be displayed.  
- A final message indicating the testing accuracy of the trained model, for example:
testing accuracy: 88.50%


---

## Limitations and Assumptions

1. **Data Quality**:  
 - The dataset assumes that missing values are represented by zeros in certain columns (e.g., `Glucose`, `BloodPressure`).  

2. **Feature Engineering**:  
 - No feature engineering is performed beyond scaling and imputation. This is a baseline implementation.  

3. **Model Generalization**:  
 - The model's performance is dependent on the quality and size of the dataset. Additional data might improve its accuracy.  

---

## Results Interpretation

The trained model provides a prediction of whether an individual has diabetes based on their health indicators. The model's accuracy on the test dataset is a measure of its effectiveness. Higher accuracy indicates better predictive performance.

---

## References

- **Dataset Source**: [UCI Machine Learning Repository - Pima Indians Diabetes Database](https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes)  
- **Machine Learning Frameworks**:  
- [Scikit-learn Documentation](https://scikit-learn.org/)  
- [Keras Documentation](https://keras.io/)  

---

## License

This project is licensed under the MIT License. You are free to use, modify, and distribute the code with proper attribution.


