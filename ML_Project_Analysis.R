## Machie Learnign Course Project

## set the working environment


## read the train and test data into variables testd and traind respectively
testd= read.csv("pml-testing.csv", sep=",", header=T,na.strings=c("NA",""))
traind= read.csv("pml-training.csv", sep=",", header=T)

## clean the data a bit. 
## remove the columns or variables which have NA for all the records. 
## results in 60 columns
traind = traind[, colSums(traind)==0]

## remove the columns that has timestamp, user name and window 
## results in 54 columns
traind = traind[,-c(grep("timestamp|X|user_name|new_window",names(traind)))]

## remove num_window column as this is not measured from the devises, but a serial number
## result is 53 columns
traind = traind[,-1]

## Determining the predictor variables/ columns that have higher predicting capacity
M = abs(cor(traind[,-53]))
diag(M) = 0
KeyPredictors = rownames(which(M > 0.8,arr.ind=T))

## split the training data into traind and traind_test sets
trainIndex = createDataPartition(y = traind$classe, p=0.7,list=FALSE)
traind = traind[trainIndex,]
traind_test =traind[-trainIndex,]

## creat the modoel 1 and apply the training set
modFit1 <- train(trainData$classe ~.,data = trainData,method="rpart")
modFit1

## create a model 2 and apply to the training set. RandomForest Method is applied
tc = trainControl(method = "repeatedcv", number = 3, repeats = 3, verboseIter=T, returnResamp='all')
modFit2 = train(traind$classe ~.,data = traind,method="rf", trControl=tc)
PredFit = predict(modFit2, data=traind_test)
table(PredFit, traind_test$classe)

## apply prediction model to the 20 test cases
test_results = predict(modFit2, testd)

