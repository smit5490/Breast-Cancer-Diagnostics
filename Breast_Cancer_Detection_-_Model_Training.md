# Breast Cancer Detection - Model Training
Robert Smith  
8/9/2017  


This R Markdown notebook contains model training code for the breast mass data in the UCI Machine Learning Repository located [here](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)). The ultimate goal of this project is to produce a robust classification algorithm that can correctly identify a benign/malignant tumor based on its features. The metric used will be the highest area under the ROC curve.

Remember how many of the features exhibited multicollinearity? In this notebook, we will apply four models to our data that are robust to this artifact:  
  * L1 Regularized Logistic Regression  
  * L2 Regularized Logistic Regression  
  * Random Forests  
  * Boosted Trees  

We have already explored and cleaned our data set. We will begin with our cleaned_df:

```r
df<-read.csv("cleaned_df.csv",header=TRUE)
#Cast diagnosis as a factor variable
df$diagnosis<-factor(df$diagnosis,levels=c("1","0"))
```

Build Testing & Training Sets:

```r
library(caret);library(ROCR)

#Create test & training sets
set.seed(1234)
inTrain<-createDataPartition(df$diagnosis, p=0.8, list=FALSE)
train_set <- df[inTrain,]
test_set <- df[-inTrain,]

#Build DataFrame to store results
model_results<-data.frame(Model=character(),CV_Accuracy=double(),Test_Accuracy=double(),Test_AUROC=double())
```

Train L1-Regularized Logistic Regression Model:

```r
set.seed(1234)

#Perform L1 regularized logistic regression across 100 lambdas
tune_grid<-expand.grid(alpha=1,
                       lambda=10^seq(3,-3,length=100))

#Perform 10-Fold Cross-Validation
train_control <- trainControl(method="cv", number=10)

#Train Model
l1_model<-train(diagnosis~.,
                data=train_set,
                method="glmnet",
                tuneGrid=tune_grid,
                trControl=train_control)

#Cross-Validated Training Accuracy
l1_cv_accuracy<-max(l1_model$results$Accuracy)

#Predict on test_set
l1_predictions<-predict(l1_model,test_set,"prob")

#Calculate Area Under ROC Curve
ROCRpred<-prediction(data.frame(l1_predictions[,1]),test_set['diagnosis'])
ROCRperf <- performance(ROCRpred,'auc')
l1_AUROC<-ROCRperf@y.values[[1]]

#Calculate Test Set Accuracy
l1_CFM<-confusionMatrix(predict(l1_model,test_set),test_set$diagnosis)
l1_test_accuracy<-as.numeric(l1_CFM$overall[1])

#Append to model_results dataframe
model_results<-rbind(model_results,
                     data.frame(Model="l1",CV_Accuracy=l1_cv_accuracy,
                          Test_Accuracy=l1_test_accuracy,Test_AUROC=l1_AUROC))
```

Train L2-Regularized Logistic Regression Model:

```r
set.seed(1234)

#Perform L2 regularized logistic regression across 100 lambdas
tune_grid<-expand.grid(alpha=0,
                       lambda=10^seq(3,-3,length=100))

#Perform 10-Fold Cross-Validation
train_control <- trainControl(method="cv",number=10)

#Train Model
l2_model<-train(diagnosis~.,
                data=train_set,
                method="glmnet",
                tuneGrid=tune_grid,
                trControl=train_control)

#Cross-Validated Training Accuracy
l2_cv_accuracy<-max(l2_model$results$Accuracy)

#Predict on test_set
l2_predictions<-predict(l2_model,test_set,"prob")

#Calculate Area Under ROC Curve
ROCRpred<-prediction(data.frame(l2_predictions[,1]),test_set['diagnosis'])
ROCRperf <- performance(ROCRpred,'auc')
l2_AUROC<-ROCRperf@y.values[[1]]

#Calculate Test Set Accuracy
l2_CFM<-confusionMatrix(predict(l2_model,test_set),test_set$diagnosis)
l2_test_accuracy<-as.numeric(l2_CFM$overall[1])

#Append to model_results dataframe
model_results<-rbind(model_results,
                     data.frame(Model="l2",CV_Accuracy=l2_cv_accuracy,
                          Test_Accuracy=l2_test_accuracy,Test_AUROC=l2_AUROC))
```

Train Random Forest Model:

```r
set.seed(1234)

#Trying 3 different levels of feature consideration at each split
tune_grid<-expand.grid(mtry=c(round(sqrt(30),0),10,20))

#Perform 10-Fold Cross-Validation
train_control <- trainControl(method="cv", number=10)

#Will use default growth of 500 trees
rf_model<-train(factor(diagnosis)~.,
                data=train_set,
                method="rf",
                tuneGrid=tune_grid,
                trControl=train_control)

#Cross-Validated Accuracy
rf_cv_accuracy<-max(rf_model$results$Accuracy)

#Predict on test set
rf_predictions<-predict(rf_model,test_set,"prob")

#Calculate Area Under ROC Curve
ROCRpred<-prediction(data.frame(rf_predictions[,1]),test_set['diagnosis'])
ROCRperf <- performance(ROCRpred,'auc')
rf_AUROC<-ROCRperf@y.values[[1]]

#Calculate Test Set Accuracy
rf_CFM<-confusionMatrix(predict(rf_model,test_set),test_set$diagnosis)
rf_test_accuracy<-as.numeric(rf_CFM$overall[1])

#Append to model_results dataframe
model_results<-rbind(model_results,
                     data.frame(Model="rf",CV_Accuracy=rf_cv_accuracy,
                          Test_Accuracy=rf_test_accuracy,Test_AUROC=rf_AUROC))
```

Train Boosted Tree Model:

```r
set.seed(1234)

#Define Hyperparameter Grid
tune_grid<-expand.grid(n.trees=c(100,200),
                       interaction.depth=c(1,3,5),
                       shrinkage=c(0.05,0.1,0.2),
                       n.minobsinnode=1)

#Perform 10-Fold Cross-Validation
train_control <- trainControl(method="cv", number=10)

#Train Model
gb_model<-train(factor(diagnosis)~.,
                data=train_set,method="gbm",
                tuneGrid=tune_grid,
                trControl=train_control,
                verbose=F)

#Cross-Validated Accuracy
gb_cv_accuracy<-max(gb_model$results$Accuracy)

#Predict on test set
gb_predictions<-predict(gb_model,test_set,"prob")

#Calculate Area Under ROC Curve
ROCRpred<-prediction(data.frame(gb_predictions[,1]),test_set['diagnosis'])
ROCRperf <- performance(ROCRpred,'auc')
gb_AUROC<-ROCRperf@y.values[[1]]

#Calculate Test Set Accuracy
gb_CFM<-confusionMatrix(predict(gb_model,test_set),test_set$diagnosis)
gb_test_accuracy<-as.numeric(gb_CFM$overall[1])

#Append to model_results dataframe
model_results<-rbind(model_results,
                     data.frame(Model="gb",CV_Accuracy=gb_cv_accuracy,
                          Test_Accuracy=gb_test_accuracy,Test_AUROC=gb_AUROC))
```

Summary of Model Results:

```r
print(model_results)
```

```
##   Model CV_Accuracy Test_Accuracy Test_AUROC
## 1    l1   0.9759420     0.9823009  0.9989940
## 2    l2   0.9758937     0.9823009  1.0000000
## 3    rf   0.9628019     0.9380531  0.9857478
## 4    gb   0.9737681     0.9380531  0.9906103
```

The model with the highest cross-validated accuracy is L1-regularized logistic regression. The model with the highest test accuracy is a tie between L1 and L2-regularized logistic regression. However, the area under the ROC curve is marginally higher for L2-regularized logistic regression. Let's look at the confusion matrix:

```r
l2_CFM$table
```

```
##           Reference
## Prediction  1  0
##          1 40  0
##          0  2 71
```
The model correctly identify 95% of malignant tumors (e.g. sensitivity) and 100% of benign tumors (specificity). Let's export our winning model!


```r
saveRDS(l2_model,"final_model.rds")
```
