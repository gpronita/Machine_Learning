Machine_Learning
================

## Data
The data for this assignment comes from [here](http://groupware.les.inf.puc-rio.br/har), and contains information from belt, forearm, arm, and dumbbell accelerometers.  The data are split into a training group (19,622) observations and testing group (20 observations).  Participants in this study were asked to do a "Dumbbell Biceps Curl" in five different ways, including using correct form and four common mistakes.  Participants were equipped with censors on the arm, belt and dumbbell itself.  

## Procedure Followed
Splitting of Test Set  
```{r}
set.seed(614)
library(lattice); library(ggplot2); library(caret)
pml.training <- read.csv("~/R/Coursera /Machine Learning/Project/pml-training.csv")
inTrain <- createDataPartition(y=pml.training$classe, p=0.9, list=FALSE)
training <- pml.training[inTrain,]
testing <- pml.training[-inTrain,]
```
```{r}
ptm <- proc.time()
modFit <- train(classe ~ user_name + pitch_arm + yaw_arm + roll_arm + roll_belt + pitch_belt + yaw_belt + gyros_belt_x + gyros_belt_y + gyros_belt_z + accel_belt_x + accel_belt_y + accel_belt_z + magnet_belt_x + magnet_belt_y + magnet_belt_z + gyros_arm_x + gyros_arm_y + gyros_arm_z + accel_arm_x + accel_arm_y + accel_arm_z + magnet_arm_x + magnet_arm_y + magnet_arm_z + roll_dumbbell + pitch_dumbbell + yaw_dumbbell, method="gbm", data=training, verbose=FALSE)
proc.time() - ptm
```

```{r}
print(modFit)
predictTr <- predict(modFit,training)
table(predictTr, training$class)
```
The model correctly classifies 93% of the data.

```{r}
summary(modFit,n.trees=150)
```

A plot of these top two features colored by outcome demonstrates their relative importance.  
```{r}
qplot(roll_belt, yaw_belt,colour=classe,data=training)
```

## Predicting on the Test Set

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
The algorithm correctly predicted the outcome for all observations. 
