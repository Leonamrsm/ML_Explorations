# Machine Learning Explorations

## Business Problem

### Description

Data Money company believes that the expertise in training and fine-tuning algorithms, carried out by the company's Data Scientists, is the main reason for the excellent results that the consultancies have been delivering to their clients.

### Objective

The objective of this project will be to carry out tests with Classification, Regression and Clustering algorithms, to study the change in performance behavior, as the values ​​of the main overfitting and underfitting control parameters change.

## Solution planning

### Final Product

The final product will be 7 tables showing the performance of the algorithms, evaluated using multiple metrics, for 3 different data sets: Training, validation and testing.

### Tested algorithms

#### Classification:

- **Algorithms:** KNN, Decision Tree, Random Forest and Logistic Regression
- **Performance metrics:** Accuracy, Precision, Recall and F1-Score

#### Regression:
- **Algorithms:** Linear Regression, Decision Tree Regressor, Random Forest Regressor, Polinomial
Regression, Linear Regression Lasso, Linear Regression Ridge, Linear Regression Elastic Net,
Polinomial Regression Lasso, Polinomial Regression Ridge e Polinomial Regression Elastic Net
- **Performance metrics:** R2, MSE, RMSE, MAE e MAPE

#### Clustering:
- **Algorithms:**  K-Means e Affinity Propagation
- **Performance metrics:** Silhouette Score

### Tools Used

Python 3.11 and Scikit-learn


## Development

### Solution strategy

1.  Split the data into training, testing, and validation.
2. Perform a grid search to find the best hyperparameters that control the algorithm's overfitting. For each combination of hyperparameters, train the model on the training set and evaluate its performance on the validation set.
3. Identify the combination of hyperparameters that results in the best model performance on the validation set.
4. Merge the training and validation data.
5. Retrain the algorithm by combining the training and validation data, using the best values ​​for the algorithm's control parameters.
6. Measure the performance of the algorithms trained with the best parameters, using the test data set.
7. Evaluate the tests and note the 3 main insights that stood out.

## Top 3 Insights

1. Tree-based algorithms perform better in all metrics when applied to test data in the Classification test.

2. The performance of the classification algorithms on the validation data was very close to the performance on the test data.

3. All regression algorithms did not present good performance metrics, which shows a need for attribute selection and better preparation of the independent variables of the data set.

## Results

### Classification Test:


### Regression Test:


### Clustering Test:
