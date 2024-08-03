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

#### Training Set

| Algorithm                   | Accuracy | Precision | Recall | F1-Score |
|-----------------------------|----------|-----------|--------|----------|
| K Neighbors Classifier      | 0.941    | 0.967     | 0.894  | 0.929    |
| Decision Tree Classifier    | 0.971    | 0.979     | 0.953  | 0.966    |
| Random Forest Classifier    | 0.999    | 1.000     | 0.998  | 0.999    |
| Logistic Regression         | 0.875    | 0.870     | 0.836  | 0.853    |

#### Validation Set

| Algorithm                   | Accuracy | Precision | Recall | F1-Score |
|-----------------------------|----------|-----------|--------|----------|
| K Neighbors Classifier      | 0.939    | 0.966     | 0.891  | 0.927    |
| Decision Tree Classifier    | 0.971    | 0.979     | 0.953  | 0.966    |
| Random Forest Classifier    | 0.999    | 1.000     | 0.998  | 0.999    |
| Logistic Regression         | 0.874    | 0.869     | 0.835  | 0.851    |

#### Test Set

| Algoritmo                   | Accuracy | Precision | Recall | F1-Score |
|-----------------------------|----------|-----------|--------|----------|
| K Neighbors Classifier      | 0.929    | 0.957     | 0.879  | 0.916    |
| Decision Tree Classifier    | 0.956    | 0.961     | 0.937  | 0.949    |
| Random Forest Classifier    | 0.963    | 0.972     | 0.942  | 0.957    |
| Logistic Regression         | 0.871    | 0.869     | 0.832  | 0.850    |


### Regression Test:

#### Training Set

| Algorithm                         | R2    | MSE    | RMSE   | MAE    | MAPE  |
|-----------------------------------|-------|--------|--------|--------|-------|
| Linear Regression                 | 0.046 | 455.996| 21.354 | 16.998 | 8.653 |
| Decision Tree Regressor           | 0.104 | 428.127| 20.691 | 16.460 | 8.184 |
| Random Forest Regressor           | 0.905 | 45.552 | 6.749  | 4.730  | 2.519 |
| Polynomial Regression             | 0.092 | 434.135| 20.836 | 16.505 | 8.386 |
| Lasso                             | 0.046 | 456.167| 21.358 | 17.011 | 8.640 |
| Ridge                             | 0.046 | 456.106| 21.357 | 17.007 | 8.646 |
| Elastic Net                       | 0.046 | 456.167| 21.358 | 17.011 | 8.640 |
| Polynomial Regression Lasso       | 0.090 | 435.191| 20.861 | 16.534 | 8.404 |
| Polynomial Regression Ridge       | 0.092 | 434.135| 20.836 | 16.505 | 8.386 |
| Polynomial Regression Elastic Net | 0.090 | 435.191| 20.861 | 16.534 | 8.404 |


#### Validation Set

| Algorithm                         | R2    | MSE     | RMSE   | MAE    | MAPE  |
|-----------------------------------|-------|---------|--------|--------|-------|
| Linear Regression                 | 0.040 | 458.447 | 21.411 | 17.040 | 8.683 |
| Decision Tree Regressor           | 0.100 | 429.538 | 20.725 | 16.541 | 8.329 |
| Random Forest Regressor           | 0.902 | 46.620  | 6.828  | 4.790  | 2.570 |
| Polinomial Regression             | 0.081 | 439.046 | 20.953 | 16.620 | 8.508 |
| Lasso                             | 0.041 | 457.794 | 21.396 | 17.035 | 8.667 |
| Ridge                             | 0.041 | 457.823 | 21.397 | 17.034 | 8.673 |
| Elastic Net                       | 0.041 | 457.794 | 21.396 | 17.035 | 8.667 |
| Polinomial Regression Lasso       | 0.080 | 439.482 | 20.964 | 16.623 | 8.525 |
| Polinomial Regression Ridge       | 0.081 | 439.046 | 20.953 | 16.620 | 8.508 |
| Polinomial Regression Elastic Net | 0.080 | 439.482 | 20.964 | 16.623 | 8.525 |


#### Test Set

| Algorithm                         | R2    | MSE     | RMSE   | MAE    | MAPE  |
|-----------------------------------|-------|---------|--------|--------|-------|
| Linear Regression                 | 0.052 | 461.428 | 21.481 | 17.130 | 8.522 |
| Decision Tree Regressor           | 0.090 | 443.296 | 21.055 | 16.830 | 7.886 |
| Random Forest Regressor           | 0.405 | 289.825 | 17.024 | 12.267 | 6.309 |
| Polynomial Regression             | 0.091 | 442.641 | 21.039 | 16.736 | 8.277 |
| Lasso                             | 0.050 | 462.434 | 21.504 | 17.155 | 8.531 |
| Ridge                             | 0.051 | 461.988 | 21.494 | 17.144 | 8.531 |
| Elastic Net                       | 0.050 | 462.434 | 21.504 | 17.155 | 8.531 |
| Polynomial Regression Lasso       | 0.091 | 442.488 | 21.035 | 16.736 | 8.308 |
| Polynomial Regression Ridge       | 0.091 | 442.639 | 21.039 | 16.736 | 8.277 |
| Polynomial Regression Elastic Net | 0.091 | 442.488 | 21.035 | 16.736 | 8.308 |


### Clustering Test:

| Algorithm              | Número de clusters | Average Silhouette Score |
|------------------------|--------------------|--------------------------|
| K-Means                | 3                  | 0.301                    |
| Affinity Propagation   | 2                  | 0.301                    |

## Conclusion

In this Machine Learning project, I was able to gain experience and better understand the limits of algorithms between the underfitting and overfitting states.

Tree-based algorithms are sensitive to the depth of growth and the number of trees in the forest, so choosing the correct values ​​for these parameters prevents the algorithms from entering the overfitting state.

Regression algorithms, on the other hand, are sensitive to the degree of the polynomial. This parameter
controls the limit between the underfitting and overfitting states of these algorithms.

This Machine Learning essay was very important to deepen the understanding of how various classification, regression and clustering algorithms work and what the main control parameters are between the underfitting and overfitting states.

