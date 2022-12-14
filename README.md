# PHAS0077

## Introduction
Diabetes is a chronic disease caused by the lack of insulin or the inability of using insulin, with hyperglycemia as the main sign. The research of this project uses machine learning to predict the disease of structured medical data. These techniques were applied to the diabetes dataset to predict whether diabetes patients will be readmitted within one month after discharge. Diabetes is one of the nine major diseases threatening human health. Accurate prediction of the risk of diabetes patients if go back to the hospital can help doctors make better decisions, improve the quality of hospital care, save medical resources and reduce medical cost, which has certain practical significance.

This project first conducted a large number of relevant literatures research, and investigated the mainstream research methods of disease prediction based on machine learning. Each process of disease prediction is introduced in detail, including data preprocessing, the principle and advantages and disadvantages of mainstream machine learning algorithms, evaluation indicators, model fusion and other related theories and technologies.

After collecting the data set of diabetes and preprocessing the data, the risk prediction models were established by using 4 algorithms of machine learning, logistic regression, support vector machine, xgboost and lightgbm. The fusion of the four models was carried out by using the voting method to realize the risk prediction of diabetes. The prediction results were compared and analyzed to evaluate the prediction effect of each model and achieved a good result.

## file
EDA.py: Exploratory data analysis

diabete_prediction.ipynb: data processing, feature engineering, modelling and evaluation

## library
numpy

pandas

sklearn

lightgbm

xgboost

matplotlib

seaborn

BinaryEncoder

SMOTE
## run
Download the data, install the dependencies, and just run it
