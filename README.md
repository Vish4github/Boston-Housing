# Boston-Housing

The Boston Housing data set contains the housing values in the suburbs of Boston, which was obtained from the study "Hedonic Prices and the Demand for Clean Air" by David Harrison and Daniel L. Rubenfeld published in 1978. The paper examines possible issues with using housing market data to measure the willingness to pay for clean air. The dataset contains 14 attributes and 506 observations. The purpose of the study is to compare and contrast different regression models to accurately predict the median house value.  

The dataset was split into training and test samples based on a 70,30 split. The important variables which would be critical to the analysis were identified through some initial exploratory analysis and a simple linear regression model was fitted. Step wise regression methods were employed based on AIC and BIC selection criteria for variable selection and as expected the BIC method yielded a model with a smaller number of predictors. This model was selected as the final output from the linear regression analysis.

The same dataset was then modeled using Decision trees and other advanced tree models such as Bagging, Boosting and Random Forests. Advanced models such GAM and Neural Networks were also modeled
