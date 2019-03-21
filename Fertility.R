# library
library(caret)
library(randomForest)
library(ggcorrplot)
library(GGally)
library(e1071)
library(ROCR)
library(pROC)
library(RCurl)
library(dplyr)
library(readr)
library(corrplot)
library(ggvis)

#import data set
dataFertil = read_csv("Fertility.csv")
dataFertil = na.omit(dataFertil)

#change data output to factor
dataFertil$output = as.factor(dataFertil$output)
str(dataFertil)
dim(dataFertil)
summary(dataFertil)

prop.table(table(dataFertil$output))

#make numeric column
dataFactorFertil = dataFertil
dataFactorFertil$`childish-disease` = as.numeric(dataFactorFertil$`childish-disease`)
dataFactorFertil$trauma = as.numeric(dataFactorFertil$trauma)
dataFactorFertil$`surgical-intervention` = as.numeric(dataFactorFertil$`surgical-intervention`)
dataFactorFertil$fevers = as.numeric(dataFactorFertil$fevers)
dataFactorFertil$smoking = as.numeric(dataFactorFertil$smoking)
dataFactorFertil$output = as.numeric(dataFactorFertil$output)

# scatter plot
dataFertil %>% ggvis(~age, ~season, fill = ~output) %>% layer_points()
dataFertil %>% ggvis(~smoking, ~`surgical-intervention`, fill = ~output) %>% layer_points()

#overall Corr
cor(dataFactorFertil, dataFactorFertil$output)
cor(dataFactorFertil$output, dataFactorFertil$season)
#correlation
corr = cor(dataFactorFertil[,0:10])
corrplot(corr)

barplot(table(dataFertil$output),
        main="Fertility", col="black")

#make numeric factor

dataFactorFertil = dataFertil
dataFactorFertil$`childish-disease` = as.factor(dataFactorFertil$`childish-disease`)
dataFactorFertil$trauma = as.factor(dataFactorFertil$trauma)
dataFactorFertil$`surgical-intervention` = as.factor(dataFactorFertil$`surgical-intervention`)
dataFactorFertil$fevers = as.factor(dataFactorFertil$fevers)
dataFactorFertil$smoking = as.factor(dataFactorFertil$smoking)
dataFactorFertil$output = as.factor(dataFactorFertil$output)

fertil = dataFactorFertil #add labels only for plot
levels(fertil$output) = c("Normal","Altered")
levels(fertil$season) = c("Spring","Summer","Fall", "Winter")
mosaicplot(fertil$output ~ fertil$season,
           main="Fate", shade=FALSE,color=TRUE,
           xlab="Fertility", ylab="season")

boxplot(fertil$age ~ fertil$output,
        main="Fate by Age",
        ylab="Age",xlab="Fertility")

boxplot(fertil$season ~ fertil$output,
        main="Fate by Season",
        ylab="Season",xlab="Fertility")

boxplot(fertil$alcoholic ~ fertil$output,
        main="Fate by alcoholic",
        ylab="alcoholic",xlab="Fertility")

boxplot(fertil$age ~ fertil$output,
        main="Fate by Age",
        ylab="Age",xlab="Fertility")

boxplot(fertil$sitting ~ fertil$output,
        main="Fate by sitting",
        ylab="sitting",xlab="Fertility")

## Creating training and test data
set.seed(10)
trainIndexFertil <- createDataPartition(dataFertil$output, p = 0.7, list = FALSE, times = 1)
trainFertil <- dataFertil[ trainIndexFertil,]
testFertil <- dataFertil[-trainIndexFertil,]
nrow(trainFertil)/(nrow(testFertil)+nrow(trainFertil)) #checking whether really 80% -> OK

AUC = list()
Accuracy = list()

##Logistic regression
logRegModel <- train(output ~ ., data=trainFertil, method = 'glm', family = 'binomial')
logRegPrediction <- predict(logRegModel, testFertil)
logRegPredictionprob <- predict(logRegModel, testFertil, type='prob')[2]
logRegConfMat <- confusionMatrix(logRegPrediction, testFertil$output)

#ROC Curve
AUC$logReg <- roc(as.numeric(testFertil$output),as.numeric(as.matrix((logRegPredictionprob))))$auc
Accuracy$logReg <- logRegConfMat$overall['Accuracy']  #found names with str(logRegConfMat)  

### Applying learning models
fitControl <- trainControl(method="cv",
                           number = 5,
                           preProcOptions = list(thresh = 0.99), # threshold for pca preprocess
                           classProbs = TRUE,
                           summaryFunction = twoClassSummary)

## Random Forest
model_rf <- train(output~ .,
                  trainFertil,
                  method="ranger",
                  metric="ROC",
                  #tuneLength=10,
                  #tuneGrid = expand.grid(mtry = c(2, 3, 6)),
                  preProcess = c('center', 'scale'),
                  trControl=fitControl)

## testing for random forets
pred_rf <- predict(model_rf, testFertil)
cm_rf <- confusionMatrix(pred_rf, testFertil$output)
cm_rf
RFPredictionprob = predict(model_rf,testFertil,type="prob")[, 2]
AUC$RF <- roc(as.numeric(testFertil$output),as.numeric(as.matrix((RFPredictionprob))))$auc
Accuracy$RF <- cm_rf$overall['Accuracy']  

##Naive Bayes
model_nb <- train(output~.,
                  trainFertil,
                  method="nb",
                  metric="ROC",
                  preProcess=c('center', 'scale'),
                  trace=FALSE,
                  trControl=fitControl)
pred_nb <- predict(model_nb, testFertil)
cm_nb <- confusionMatrix(pred_nb, testFertil$output)
cm_nb

RFPredictionprob = predict(model_nb,testFertil,type="prob")[, 2]
AUC$RF <- roc(as.numeric(testFertil$output),as.numeric(as.matrix((RFPredictionprob))))$auc
Accuracy$RF <- cm_rf$overall['Accuracy']  

## Boosted tree
set.seed(10)
objControl <- trainControl(method='cv', number=10,  repeats = 10)
gbmGrid <-  expand.grid(interaction.depth =  c(1, 5, 9),
                        n.trees = (1:30)*50,
                        shrinkage = 0.1,
                        n.minobsinnode =10)
# run model
boostModel <- train(output ~ .,data=trainFertil, method='gbm',
                    trControl=objControl, tuneGrid = gbmGrid, verbose=F)
# See model output in Appendix to get an idea how it selects best model
#trellis.par.set(caretTheme())
#plot(boostModel)
boostPrediction <- predict(boostModel, testFertil)
boostPredictionprob <- predict(boostModel, testFertil, type='prob')[2]
boostConfMat <- confusionMatrix(boostPrediction, testFertil$output)

#ROC Curve
AUC$boost <- roc(as.numeric(testFertil$output),as.numeric(as.matrix((boostPredictionprob))))$auc
Accuracy$boost <- boostConfMat$overall['Accuracy']  

#Stochastic gradient boosting
feature.names=names(dataFertil)

for (f in feature.names) {
  if (class(dataFertil[[f]])=="factor") {
    levels <- unique(c(dataFertil[[f]]))
    dataFertil[[f]] <- factor(dataFertil[[f]],
                              labels=make.names(levels))
  }
}
set.seed(10)
inTrainRows <- createDataPartition(dataFertil$output,p=0.8,list=FALSE)
trainFertil2 <- dataFertil[inTrainRows,]
testFertil2 <-  dataFertil[-inTrainRows,]


fitControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 10,
                           ## Estimate class probabilities
                           classProbs = TRUE,
                           ## Evaluate performance using
                           ## the following function
                           summaryFunction = twoClassSummary)

set.seed(10)
gbmModel <- train(output ~ ., data = trainFertil2,
                  method = "gbm",
                  trControl = fitControl,
                  verbose = FALSE,
                  tuneGrid = gbmGrid,
                  ## Specify which metric to optimize
                  metric = "ROC")
gbmPrediction <- predict(gbmModel, testFertil2)
gbmPredictionprob <- predict(gbmModel, testFertil2, type='prob')[2]
gbmConfMat <- confusionMatrix(gbmPrediction, testFertil2$output)
#ROC Curve
AUC$gbm <- roc(as.numeric(testFertil2$output),as.numeric(as.matrix((gbmPredictionprob))))$auc
Accuracy$gbm <- gbmConfMat$overall['Accuracy']

#Support Vector Machine
set.seed(10)
svmModel <- train(output ~ ., data = trainFertil2,
                  method = "svmRadial",
                  trControl = fitControl,
                  preProcess = c("center", "scale"),
                  tuneLength = 8,
                  metric = "ROC")
svmPrediction <- predict(svmModel, testFertil2)
svmPredictionprob <- predict(svmModel, testFertil2, type='prob')[2]
svmConfMat <- confusionMatrix(svmPrediction, testFertil2$output)
#ROC Curve
AUC$svm <- roc(as.numeric(testFertil2$output),as.numeric(as.matrix((svmPredictionprob))))$auc
Accuracy$svm <- svmConfMat$overall['Accuracy']  

#Result

row.names <- names(Accuracy)
col.names <- c("AUC", "Accuracy")
cbind(as.data.frame(matrix(c(AUC,Accuracy),nrow = 5, ncol = 2,
                           dimnames = list(row.names, col.names))))


# AS Regression

##h2o
library(h2o)
h2o.init(nthreads = -1) ## nthreads untuk ngurangin nodes di pc nya

d_Train_hf <- as.h2o(trainFertil)
model_h2o_automl <- h2o.automl(y = "output", training_frame = d_Train_hf, max_models = 5)
model_h2o_automl

library(fossil)

#create a hypothetical clustering outcome with 2 distinct clusters 
g1 <- sample(1:2, size=10, replace=TRUE) 
g2 <- sample(1:3, size=10, replace=TRUE) 
rand.index(g1, g2)

