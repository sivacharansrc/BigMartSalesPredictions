# REFERENCES
# https://www.analyticsvidhya.com/blog/2016/05/h2o-data-table-build-models-large-data-sets/
# https://www.analyticsvidhya.com/blog/2015/10/regression-python-beginners/
# https://www.analyticsvidhya.com/blog/2016/04/complete-tutorial-tree-based-modeling-scratch-in-python/
# https://www.analyticsvidhya.com/blog/2016/03/introduction-deep-learning-fundamentals-neural-networks/

###### PREPARING ENVIRONMENT ####
rm(list = ls())
options(scipen = 999)
library(data.table)
library(dplyr)
#library(ggplot2)
library(summaryR)
library(h2o)
#library(plotly)

####### READING DATA ######
# View(t(summaryR(data.frame(train)))) | View(t(summaryR(data.frame(test))))


train <- fread("./src/train.csv", sep = ",", header = T, na.strings = c("",NA," ","NA", "N/A"))
test <- fread("./src/test.csv", sep = ",", header = T, na.strings = c("",NA," ","NA", "N/A"))

####### DATA MANIPULATION AND FEATURE ENGINEERING #########
itemIdentifierList <- unique(test$Item_Identifier)

clean.train <- train[Item_Identifier %in% itemIdentifierList,]

test <- test[,Item_Outlet_Sales := mean(clean.train$Item_Outlet_Sales, na.rm = T)]

cleanData <- rbind(clean.train, test)

# str(cleanData)
# summary(cleanData)

# Backup copy of Item Identifier, and Outlet Identifier

ItemIdDf <- data.frame(ItemId = cleanData$Item_Identifier,
                       ItemIDCode = as.integer(as.factor(cleanData$Item_Identifier)))
ItemIdDf <- unique(ItemIdDf)

outletIDDf <- data.frame(ItemId = cleanData$Outlet_Identifier ,
                         ItemIDCode = as.integer(as.factor(cleanData$Outlet_Identifier)))
outletIDDf <- unique(outletIDDf)

## Converting all character columns to integers:

cleanData$Item_Identifier <- as.integer(as.factor(cleanData$Item_Identifier))
cleanData$Item_Fat_Content <- as.integer(as.factor(cleanData$Item_Fat_Content))
cleanData$Item_Type <- as.integer(as.factor(cleanData$Item_Type))
cleanData$Outlet_Identifier <- as.integer(as.factor(cleanData$Outlet_Identifier))
cleanData$Outlet_Size <- as.integer(as.factor(cleanData$Outlet_Size))
cleanData$Outlet_Location_Type <- as.integer(as.factor(cleanData$Outlet_Location_Type))
cleanData$Outlet_Type <- as.integer(as.factor(cleanData$Outlet_Type))

cleanData[, Item_Weight_NA := ifelse(is.na(Item_Weight),0,1) ][, Outlet_Size_NA := ifelse(is.na(Outlet_Size),0,1)][,No_of_Yrs := 2013 - Outlet_Establishment_Year]

cleanData[is.na(Item_Weight), Item_Weight:= -999][is.na(Outlet_Size), Outlet_Size:= -999]
cleanData[,Outlet_Establishment_Year:= NULL]

## SEPERATING TRAIN AND TEST CLEAN DATA

clean.train <- cleanData[1:nrow(clean.train),]
clean.test <- cleanData[-(1:nrow(clean.train)),]

View(head(cleanData))
####### PREDICTION MODELS #########

localH2o <- h2o.init(nthreads = -1)

train.h2o <- as.h2o(clean.train)
test.h2o <- as.h2o(clean.test)

names(train.h2o)

xIndep <- c(1,2,4:7,9,10, 12,14)
yDep <- 11

reg.model <- h2o.glm(x=xIndep, y=yDep, training_frame = train.h2o, family = "gaussian")
h2o.performance(reg.model)

## RANDOM FOREST

rf.model <- h2o.randomForest(y=yDep, x=xIndep, training_frame = train.h2o,
                             ntrees = 1400, mtries = 5, max_depth = 7, nbins = 30 , seed = 1001)

#?h2o.randomForest

#h2o.varimp(rf.model)

h2o.performance(rf.model)

rf.pred <- as.data.frame(h2o.predict(rf.model, test.h2o))

#View(head(gbm.pred))

df <- data.frame(Item_Identifier = test$Item_Identifier,
                 Outlet_Identifier = test$Outlet_Identifier,
                 Item_Outlet_Sales = rf.pred$predict)

### GRADIENT BOOSTING

gbm.model <- h2o.gbm(y=yDep, x=xIndep, training_frame = train.h2o, 
                     ntrees = 1000, max_depth = 8, learn_rate = 0.05, seed = 100)


h2o.performance(gbm.model)

## PREDICT TEST USING THE GBM MODEL

gbm.pred <- as.data.frame(h2o.predict(gbm.model, test.h2o))

#View(head(gbm.pred))

df <- data.frame(Item_Identifier = test$Item_Identifier,
                 Outlet_Identifier = test$Outlet_Identifier,
                 Item_Outlet_Sales = gbm.pred$predict)


#write.csv(df, "./Output/BigMartSales_Siva_1_Submission.csv", row.names = F)

