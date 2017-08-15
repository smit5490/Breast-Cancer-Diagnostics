# Breast Cancer Diagnostics
This respository contains Python and R code to explore, clean, and perform machine learning on breast mass data in the UCI Machine Learning Repository located [here](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)). The ultimate goal of this project is to produce a robust classification algorithm that can correctly identify a benign/malignant tumor based on its features. 

Nearly the same data science pipeline is written in R and Python to provide a side-by-side comparison of their exploratory data analysis and machine learning syntax and capabilities. There are four notebooks in this repository:

*Python*
* Breast Cancer Detection - EDA & Data Cleaning.ipynb
* Breast Cancer Detection - Model Training.ipynb

*R*
* Breast Cancer Detection - EDA & Data Cleaning.Rmd
* Breast Cancer Detection - Model Training.Rmd

Start with either EDA & Data Cleaning script. A cleaned dataframe (cleaned_df.csv) is produced by these files and is used in the Model Training scripts. The Model Training scripts implement four machine learning algorithms:

* L1-Regularized Logistic Regression
* L2-Regularized Logistic Regression
* Random Forest
* Boosted Trees

The final models are stored in final_model.rds and final_model.pkl for R and Python, respectively.

