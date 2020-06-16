#This project is based on breast cancer diagnostic dataset from Kaggle for predictive analysis.
#It is a classification problem and our goal is to predict the type of tumor whether it is benign 
#or malignant based on other given data. Here, Random Forest Algorithm and Artificial Neural Network
#(ANN) are used in R for the predictions.

setwd('C:\\Sandip\\Machine-Learning\\Practice-Datasets\\Breast-Cancer-Prediction')
getwd()
dataset_orig = read.csv('datasets_180_408_data.csv')
head(dataset_orig)
str(dataset_orig)
summary(dataset_orig)
#----------------------------------------------------------------------------

dataset = dataset_orig[2:32]
head(dataset)
levels(factor(dataset$diagnosis))

dataset$diagnosis = factor(dataset$diagnosis, levels = c('B', 'M'), labels = c(1, 2))
head(dataset)
#--------Exploratory data analysis------------------------------------------

library(ggplot2)
s <- ggplot(data=dataset_orig, aes(x=diagnosis))
s + geom_histogram(stat = 'count', fill='Green', colour='Black') + xlab('Type of Cancer, B=Benign, M=Malignant')

install.packages("corrplot")
library(corrplot)
mydata.cor = cor(dataset[2:31])
corrplot(mydata.cor)

#----------------------------------------------------------------------------
library(caTools)
set.seed(123)
split = sample.split(dataset$diagnosis, SplitRatio = 0.8)
training_set = subset(dataset, split==TRUE)
test_set = subset(dataset, split==FALSE)

training_set[, 2:31] = scale(training_set[, 2:31])
test_set[, 2:31] = scale(test_set[, 2:31])
#-------------------Random Forest--------------------------------------------------
library(randomForest)
set.seed(1234)
classifier = randomForest(x=training_set[,2:31], y = training_set$diagnosis, ntree = 10)
y_pred_randomForest = predict(classifier, newdata = test_set[,2:31], type = 'class')

y_pred_randomForest

cm = table(test_set[,1], y_pred_randomForest)

Accuracy_of_testset_randomForest = ((cm[1,1] + cm[2,2])/(cm[1,1] + cm[1,2] + cm[2,1] + cm[2,2]))*100

y_pred_randomForest_train = predict(classifier, newdata = training_set[,2:31], type = 'class')

cm_train = table(training_set[,1], y_pred_randomForest_train)

Accuracy_of_trainingset_randomForest = ((cm_train[1,1] + cm_train[2,2])/(cm_train[1,1] + cm_train[1,2] + cm_train[2,1] + cm_train[2,2]))*100

#--------------Artificial Neural Network(ANN)--------------------------------------------------------------

library(h2o)
h2o.init()
classifier = h2o.deeplearning(y='diagnosis', training_frame = as.h2o(training_set), 
                              validation_frame = as.h2o(test_set), activation = 'Rectifier', 
                              hidden = c(16,16), epochs = 100, train_samples_per_iteration = -2)

y_pred_ANN = h2o.predict(classifier, newdata = as.h2o(test_set[2:31]))

y_pred_ANN = as.vector(y_pred_ANN[1])

cm_test_ANN = table(test_set[,1], y_pred_ANN)

Accuracy_of_testset_ANN = ((cm_test_ANN[1,1] + cm_test_ANN[2,2])/(cm_test_ANN[1,1] + cm_test_ANN[1,2] + cm_test_ANN[2,1] + cm_test_ANN[2,2]))*100

y_pred_ANN_train = h2o.predict(classifier, newdata = as.h2o(training_set[2:31]))

y_pred_ANN_train = as.vector(y_pred_ANN_train[1])

cm_train_ANN = table(training_set[,1], y_pred_ANN_train)

Accuracy_of_trainingset_ANN = ((cm_train_ANN[1,1] + cm_train_ANN[2,2])/(cm_train_ANN[1,1] + cm_train_ANN[1,2] + cm_train_ANN[2,1] + cm_train_ANN[2,2]))*100



