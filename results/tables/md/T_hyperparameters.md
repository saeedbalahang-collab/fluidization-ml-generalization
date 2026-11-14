| Model | Inputs | Selected hyper-parameters | Inner CV RMSE |
|---|---:|---:|---:|
| XGBoost | Composite | n_estimators = 375, max_depth = 15, learning_rate = 0.2791, subsample = 0.8865, colsample_bytree = 0.8086, gamma = 0.1815, reg_alpha = 0.4374, reg_lambda = 0.7264 | 0.296 |
| XGBoost | Separate | n_estimators = 58, max_depth = 14, learning_rate = 0.3067, subsample = 0.9521, colsample_bytree = 0.7706, gamma = 0.005317, reg_alpha = 0.7861, reg_lambda = 0.396 | 0.272 |
| RF | Composite | n_estimators = 464, max_depth = 5, min_samples_split = 7, min_samples_leaf = 2, max_features = log2, bootstrap = True | 0.317 |
| RF | Separate | n_estimators = 174, max_depth = 15, min_samples_split = 5, min_samples_leaf = 1, max_features = None, bootstrap = True | 0.246 |
| SVR | Composite | kernel = rbf, C = 264.6, epsilon = 0.283, gamma = 0.7802 | 0.290 |
| SVR | Separate | kernel = rbf, C = 60.21, epsilon = 0.03506, gamma = 0.1173 | 0.180 |
