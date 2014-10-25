Machine_Learning
================
---
title: "Practical Machine Learning Project"

---
## Data
The data for this assignment comes from [here](http://groupware.les.inf.puc-rio.br/har), and contains information from belt, forearm, arm, and dumbbell accelerometers.  The data are split into a training group (19,622) observations and testing group (20 observations).  Participants in this study were asked to do a "Dumbbell Biceps Curl" in five different ways, including using correct form and four common mistakes.  Participants were equipped with censors on the arm, belt and dumbbell itself.  

## Procedure Followed
First, I split the training set into 90/10 subsamples.  
```{r}
set.seed(614)
library(lattice); library(ggplot2); library(caret)
pml.training <- read.csv("~/R/Coursera /Machine Learning/Project/pml-training.csv")
inTrain <- createDataPartition(y=pml.training$classe, p=0.9, list=FALSE)
training <- pml.training[inTrain,]
testing <- pml.training[-inTrain,]
```
Note: To run this code, you will need to switch what is inside `read.csv("")` to your location of the data.   The 90 percent subsample is used to train the modle, and the 10 percent sample is used for cross-validation.  I chose this simple cross-validation rather than using something like K-fold via the `cv.folds` option to cut down on execution time, which was already quite lengthy.  Next, I implement a Stochastic Gradient Boosting algorithm via the `gbm` package.
```{r}
ptm <- proc.time()
modFit <- train(classe ~ user_name + pitch_arm + yaw_arm + roll_arm + roll_belt + pitch_belt + yaw_belt + gyros_belt_x + gyros_belt_y + gyros_belt_z + accel_belt_x + accel_belt_y + accel_belt_z + magnet_belt_x + magnet_belt_y + magnet_belt_z + gyros_arm_x + gyros_arm_y + gyros_arm_z + accel_arm_x + accel_arm_y + accel_arm_z + magnet_arm_x + magnet_arm_y + magnet_arm_z + roll_dumbbell + pitch_dumbbell + yaw_dumbbell, method="gbm", data=training, verbose=FALSE)
proc.time() - ptm
```
To capture the execution time I've used `ptm` and `proc.time()`.  On my 2012 Macbook Air (1.8 GHz Intel Core i5 and 8GM) the execution time was roughly 24 minutes.
```{r}
print(modFit)
predictTr <- predict(modFit,training)
table(predictTr, training$classe)
```
The model correctly classifies 93.6 percent of the observations in the training sample using 150 trees.  The "roll_belt"" and "yaw_belt"" features were by far the most important in terms of variable influence.  
```{r}
summary(modFit,n.trees=150)
```

A plot of these top two features colored by outcome demonstrates their relative importance.  
```{r}
qplot(roll_belt, yaw_belt,colour=classe,data=training)
```
Even though these are the top features, they're still not great predictors in their own right.  Nonetheless, you can see some bunching in this simple plot.  This confirms the choice of a boosting algorithm as a good choice given the large set of relatively weak predictors.  This next plot further demonstrates the improved performance gained by using boosting iterations.

```{r}
ggplot(modFit)
```

Next, I check the performance on the 10 percent subsample to get an estimate of the algorithm's out-of-sample performance.
```{r}
predictTe <- predict(modFit,testing)
table(predictTe, testing$classe)
```
The algorithm actually peforms only does slightly worse on the testing subset than it did on the full training set, correctly classifying 93.4 percent of the observations.

## Predicting on the Test Set
Finally, I use the algorithm to predict using the testing set.  The results are run through the `pml_write_files()` function from the course Coursera site, and stored for submission.  
```{r}
pml.testing <- read.csv("~/R/Coursera /Machine Learning/Project/pml-testing.csv")
answers <- as.character(predict(modFit, pml.testing))
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(answers)
```
After submitting these answers, it turns out that the algorithm correctly predicted the outcome for 20/20 observations further confirming its strong out-of-sample classification accuracy.  
