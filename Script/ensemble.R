# Load Dependeicies
library(caret)
library(randomForest)
library(readr)
library(doMC)
registerDoMC(cores = 6)

# Set seed
set.seed(1337)

# Load data
train <- read_csv("../input/train.csv")
test  <- read_csv("../input/test.csv")

# Categorical Variables
catVars <- c('Product_Info_1', 'Product_Info_2', 'Product_Info_3',
             'Product_Info_5', 'Product_Info_6', 'Product_Info_7',
             'Employment_Info_2', 'Employment_Info_3', 'Employment_Info_5',
             'InsuredInfo_1', 'InsuredInfo_2', 'InsuredInfo_3', 'InsuredInfo_4',
             'InsuredInfo_5', 'InsuredInfo_6', 'InsuredInfo_7',
             'Insurance_History_1', 'Insurance_History_2',
             'Insurance_History_3', 'Insurance_History_4',
             'Insurance_History_7', 'Insurance_History_8',
             'Insurance_History_9', 'Family_Hist_1', 'Medical_History_2',
             'Medical_History_3', 'Medical_History_4', 'Medical_History_5',
             'Medical_History_6', 'Medical_History_7', 'Medical_History_8',
             'Medical_History_9', 'Medical_History_10', 'Medical_History_11',
             'Medical_History_12', 'Medical_History_13', 'Medical_History_14',
             'Medical_History_16', 'Medical_History_17', 'Medical_History_18',
             'Medical_History_19', 'Medical_History_20', 'Medical_History_21',
             'Medical_History_22', 'Medical_History_23', 'Medical_History_25',
             'Medical_History_26', 'Medical_History_27', 'Medical_History_28',
             'Medical_History_29', 'Medical_History_30', 'Medical_History_31',
             'Medical_History_33', 'Medical_History_34', 'Medical_History_35',
             'Medical_History_36', 'Medical_History_37', 'Medical_History_38',
             'Medical_History_39', 'Medical_History_40', 'Medical_History_41')

for (var in catVars) {
  train[[var]] <- factor(train[[var]])
  test[[var]] <- factor(test[[var]])
}


# Dummy Variables
dummyVars <- paste('Medical_Keyword_', 1:48, sep = '')

for (var in dummyVars) {
  train[[var]] <- factor(train[[var]])
  test[[var]] <- factor(test[[var]])
}

train$Id <- NULL

feature.names <- names(train)[1:(ncol(train)-1)]
fitControl <- trainControl(method="cv", number=3, verboseIter=T)
rfFit <- train(Response ~ .,
               method="rf", data=train, ntree=50, importance=TRUE,
               do.trace=10, trControl=fitControl, na.action=na.roughfix)

pred <- predict(rfFit, test[,-1])
submit = data.frame(Id = test$Id, Response = pred)

write.csv(submit, "../submissions/rf_starter.csv", row.names = FALSE, quote = FALSE)
