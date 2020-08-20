# Boston-Housing

The Boston Housing data set contains the housing values in the suburbs of Boston, which was obtained from the study "Hedonic Prices and the Demand for Clean Air" by David Harrison and Daniel L. Rubenfeld published in 1978. The paper examines possible issues with using housing market data to measure the willingness to pay for clean air. The dataset contains 14 attributes and 506 observations. The purpose of the study is to compare and contrast different regression models to accurately predict the median house value.  

The dataset was split into training and test samples based on a 70,30 split. The important variables which would be critical to the analysis were identified through some initial exploratory analysis and a simple linear regression model was fitted. Step wise regression methods were employed based on AIC and BIC selection criteria for variable selection and as expected the BIC method yielded a model with a smaller number of predictors. This model was selected as the final output from the linear regression analysis.

The same dataset was then modeled using Decision trees and other advanced tree models such as Bagging, Boosting and Random Forests. Advanced models such GAM and Neural Networks were also modeled

## Major Findings

The best linear regression model was found to be the one obtained through a backward regression process employing the BIC criterion. Even though the AIC and BIC values of this model was comparable to that from the forward regression/stepwise regression methods, this model was selected based as it was more parsimonious employing only 10 variables to calculate the median prices. The Decision tree gave a comparably better result as expected after subjecting to a pruning based on an optimum cp value. The advanced tree models were found to return better results employing bootstrap and aggregation practices. The Random forest model seems to work best even though it has higher MSE values since the insample and out of sample values don’t differ by much, indicating a lower variance. This can be attributed to the decorrelation property of the Random forests algorithm. The boosting procedure can be further explored by employing much more efficient methods in estimating the tuning parameters. The GAM was found to perform much better than the basic linear regression model and even some of the advanced tree models which indicates that some of the variables in the dataset exhibits non-linearity. The Neural Network model was found to return satisfactory results even when the current package at our disposal was constrained by the number of hidden layers we can include in the model and since these network packages are quite sensitive to the initial values, multiple random initial values were fixed to make sure the convergence is global rather than local. On doing so, a far improvement was observed in the test MSE values
