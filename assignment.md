---
title: "Machine Learning Course Project"
author: "Marcel Mos"
date: "22 Sep 2015"
output: html_document
---

## Introduction

## Loading data and cleaning of data

First the data is loaded and the following is done:

1. Some variables only contain values if a new window is started. These columns are not good predictors as they are not present for most rows and are eliminated from the data set.
2. The training set is split in a training and validation set with a 70/30 ratio


```r
library(caret)
library(kernlab)
#repeatability
set.seed(32599)
data <- read.csv('pml-training.csv', na.strings=c("", "NA", "#DIV/0!"))

#remove variables that are only populated for new windows
countisnas <- colSums(is.na(data))
data <- data[,countisnas==0]
#first 7 variables are removed (names, dates, window numbers)
data <- data[,8:dim(data)[2]]

#split the data set in a training and validation set
val <- createDataPartition(y=data$classe, p=0.70, list=FALSE)
validation <- data[val,]
training <- data[-val,]
```

## Model Selection

The problem at hand is a classification problem. Based on a number of numeric variables an acitivity has to be classified, with 5 possible classes as outcome. 

Models that will be considered are recursive partitioning and random forest. These models will be fitted using 5-fold cross validation. The best model will be tested on the validation set. 

### cross validation
The train function from the 'caret' package will be used to train the models. The control parameters are set globally. The parameter used is 5-fold cross validation


```r
tc <- trainControl(method="cv", number = 5)
```

### Recursive partitioning
The first model tried is recursive partitioning:


```r
fit.rpart <- train(classe ~ ., data=training, method="rpart", trControl=tc)
fit.rpart
```

```
## CART 
## 
## 5885 samples
##   52 predictor
##    5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 4709, 4708, 4707, 4709, 4707 
## Resampling results across tuning parameters:
## 
##   cp          Accuracy   Kappa       Accuracy SD  Kappa SD  
##   0.04013299  0.4971963  0.34402156  0.01422963   0.01904549
##   0.04986939  0.4297354  0.23552381  0.06743878   0.11521875
##   0.11042508  0.3131678  0.04379866  0.03918771   0.06011034
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was cp = 0.04013299.
```

The table above shows an accuracy for the best model of 0.497 or an out of sample error rate of 50.3%. Recursive partitioning does not look as a promising candidate. 

### Random Forest
The second model tried is random forest:


```r
fit.rf <- train(classe ~ ., data=training, method="rf", trControl=tc)
fit.rf$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 1.73%
## Confusion matrix:
##      A    B    C   D    E class.error
## A 1664    5    3   1    1 0.005973716
## B   23 1101   15   0    0 0.033362599
## C    1    8 1012   5    0 0.013645224
## D    0    0   18 941    5 0.023858921
## E    0    1    6  10 1065 0.015711645
```

The table above shows a better out of sample error rate of 1.73%. Random forest is preferable as model. 

### Out of Sample Error Rate
Using cross-validation as done above delivers an out of sample error rate of 1.73%. We have reserved 30% of our training set for final validation. 

Applying the trained model to the validation set will verify the estimation of the error rate:


```r
preds.rf <- predict(fit.rf, validation)
confusionMatrix(preds.rf, validation$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3869   37    1    0    0
##          B   20 2578   49    3    4
##          C   13   38 2333   62    6
##          D    2    3   13 2182   11
##          E    2    2    0    5 2504
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9803          
##                  95% CI : (0.9778, 0.9825)
##     No Information Rate : 0.2843          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.975           
##  Mcnemar's Test P-Value : 1.743e-09       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9905   0.9699   0.9737   0.9689   0.9917
## Specificity            0.9961   0.9931   0.9895   0.9975   0.9992
## Pos Pred Value         0.9903   0.9714   0.9515   0.9869   0.9964
## Neg Pred Value         0.9962   0.9928   0.9944   0.9939   0.9981
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2816   0.1877   0.1698   0.1588   0.1823
## Detection Prevalence   0.2844   0.1932   0.1785   0.1610   0.1829
## Balanced Accuracy      0.9933   0.9815   0.9816   0.9832   0.9954
```
The table above verifies the accuracy, The validated out of sample error rate is 1-0.9803= 1.97%. This is significantly close. 



