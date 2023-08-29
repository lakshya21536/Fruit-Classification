Group members:
1) Lakshya Kumar 2021536
2) Maanas Gaur 2021537

Introduction
Fruit classification is a vital challenge in the food industry and agriculture. This project uses various machine learning algorithms to classify fruits based on given features.

Dataset Description
The dataset for the fruit classification problem consists of two files: train.csv and test.csv. The training dataset train.csv includes 4098 columns, with the first and last columns providing 'ID' and target values ('category'), respectively. The remaining columns (n0 to n4095) contain features for classification. The target variable 'category' has 20 distinct classes, such as 'Apple Raw', 'Apple Ripe', 'Banana Raw', and so on. The test dataset test.csv contains unlabeled data with the same structure as the training dataset.

Methodology
Preprocessing Steps
Preprocessing is crucial for improving data quality and model performance. In this project, several preprocessing techniques were explored:

Removal of ID: The 'ID' column was removed as it doesn't contribute to preprocessing or classification.
Local Outlier Factor (LOF) and k-means clustering were tested for outlier detection and grouping but didn't significantly improve accuracy.
Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA) successfully reduced dimensionality and enhanced accuracy. PCA creates new features capturing essential variation, while LDA maximizes class separation.

Classification Models
Two classification algorithms were used in this project:

MLPClassifier: A multi-layer perceptron with different activation functions (logistic and tanh). Hyperparameters like layers, nodes, and optimizers were adjusted.
Logistic Regression: Logistic regression models with different solvers (newton-cg and lbfgs) were employed. Regularization and penalty were also tuned.
Ensemble Method
An ensemble approach was used to improve model performance. The Voting Classifier from scikit-learn combined predictions from multiple classifiers (nn1, nn2, lr1, and lr2) using a hard voting strategy. This approach reduced bias and variance errors, creating a more accurate and robust prediction.

