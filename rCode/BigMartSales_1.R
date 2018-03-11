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
library(dummies)

####### READING DATA ######
# View(t(summaryR(data.frame(train)))) | View(t(summaryR(data.frame(test)))) | View(head(train))


train <- fread("./src/train.csv", sep = ",", header = T, na.strings = c("",NA," ","NA", "N/A"))
test <- fread("./src/test.csv", sep = ",", header = T, na.strings = c("",NA," ","NA", "N/A"))

####### DATA MANIPULATION AND FEATURE ENGINEERING #########
# str(cleanData)
# summary(cleanData)

itemIdentifierList <- unique(test$Item_Identifier)
clean.train <- train[Item_Identifier %in% itemIdentifierList,]
test <- test[,Item_Outlet_Sales := mean(clean.train$Item_Outlet_Sales, na.rm = T)]
cleanData <- rbind(clean.train, test)

# Backup copy of Item Identifier, and Outlet Identifier

ItemIdDf <- data.frame(ItemId = cleanData$Item_Identifier,
                       ItemIDCode = as.integer(as.factor(cleanData$Item_Identifier)))
ItemIdDf <- unique(ItemIdDf)

outletIDDf <- data.frame(ItemId = cleanData$Outlet_Identifier ,
                         ItemIDCode = as.integer(as.factor(cleanData$Outlet_Identifier)))
outletIDDf <- unique(outletIDDf)

## Converting all character columns to integers: names(cleanData) View(head(cleanData))

cleanData[,Item_Weight_Fixed1 := mean(Item_Weight, na.rm = T), Item_Identifier][,Item_Weight_Fixed:= ifelse(is.na(Item_Weight),Item_Weight_Fixed1,Item_Weight)]
cleanData[,Item_Fat_Content_Fixed := ifelse(Item_Fat_Content %in% c("Regular", "reg"),"Regular","Low Fat")]
cleanData$Item_Identifier <- as.integer(as.factor(cleanData$Item_Identifier))
cleanData[,Sales_By_Item_Type := sum(Item_Outlet_Sales, na.rm = T),Item_Type]
cleanData <- data.table(dummy.data.frame(cleanData, names = c("Item_Fat_Content_Fixed", "Outlet_Type", "Outlet_Location_Type"), sep="_"))
cleanData$Item_Type <- as.integer(as.factor(cleanData$Item_Type))
cleanData$Outlet_Identifier <- as.integer(as.factor(cleanData$Outlet_Identifier))
cleanData$Outlet_Size <- as.integer(as.factor(cleanData$Outlet_Size))
cleanData[, Outlet_Size_NA := ifelse(is.na(Outlet_Size),0,1)][,No_of_Yrs := 2013 - Outlet_Establishment_Year]
cleanData[is.na(Outlet_Size), Outlet_Size:= -999]
cleanData[,c("Outlet_Establishment_Year", "Item_Weight_Fixed1", "Item_Weight", "Item_Fat_Content"):= NULL]

## SEPERATING TRAIN AND TEST CLEAN DATA   View(head(cleanData))

clean.train <- cleanData[1:nrow(clean.train),]
clean.test <- cleanData[-(1:nrow(clean.train)),]

####### PREDICTION MODELS #########

### PERFORMING LINEAR REGRESSION FOR THE DATASET

# INITIATING H2O

# localH2o <- h2o.init(nthreads = -1)
localH2O <- h2o.init(ip='localhost', nthreads=-1,
                     min_mem_size='10G', max_mem_size='20G')
# h2o.shutdown(prompt = T)

# CONVERTING DATA SET TO H2O FORMAT | names(train.h2o) names(train) head(train.h2o)
train.h2o <- as.h2o(clean.train)
test.h2o <- as.h2o(clean.test)

## SETTING COLUMN POSITIONS TO DEP AND INDEP VARIABLES

xIndep <- c(1:13, 15, 18:20)
yDep <- 14

## RANDOM FOREST

rf.model <- h2o.randomForest(y=yDep, x=xIndep, training_frame = train.h2o,
                             ntrees = 1000, mtries = 5, max_depth = 13, seed = 1001)

# h2o.varimp(rf.model)

h2o.performance(rf.model)

## PREDICT TEST USING THE RF MODEL

train.pred <- as.data.frame(h2o.predict(rf.model, train.h2o))
test.pred <- as.data.frame(h2o.predict(rf.model, test.h2o))

#View(head(gbm.pred))

train.ensemble <- clean.train
train.ensemble$Predictions <- train.pred$predict
train.ensemble.h2o <- as.h2o(train.ensemble)

test.ensemble <- clean.test
test.ensemble$Predictions <- test.pred$predict
test.ensemble.h2o <- as.h2o(test.ensemble)

## SETTING COLUMN POSITIONS TO DEP AND INDEP VARIABLES names(train.ensemble.h2o)

xIndep <- c(1,4,6,7,9,10, 12,15,16)
yDep <- 11

## RANDOM FOREST

rf.model <- h2o.randomForest(y=yDep, x=xIndep, training_frame = train.ensemble.h2o,
                             ntrees = 1400, mtries = 5, max_depth = 7, nfolds = 3, seed = 1001)

#h2o.varimp(rf.model)

h2o.performance(rf.model)

rf.pred <- as.data.frame(h2o.predict(rf.model, test.ensemble.h2o))

#View(head(gbm.pred))

df <- data.frame(Item_Identifier = test$Item_Identifier,
                 Outlet_Identifier = test$Outlet_Identifier,
                 Item_Outlet_Sales = rf.pred$predict)


#write.csv(df, "./Output/BigMartSales_Siva_17_Submission.csv", row.names = F)


### GRADIENT BOOSTING

gbm.model <- h2o.gbm(y=yDep, x=xIndep, training_frame = train.h2o, 
                     ntrees = 1400, max_depth = 8, learn_rate = 0.1, seed = 100)


h2o.performance(gbm.model)

## PREDICT TEST USING THE GBM MODEL

gbm.pred <- as.data.frame(h2o.predict(gbm.model, test.h2o))

#View(head(gbm.pred))

df <- data.frame(Item_Identifier = test$Item_Identifier,
                 Outlet_Identifier = test$Outlet_Identifier,
                 Item_Outlet_Sales = gbm.pred$predict)


#write.csv(df, "./Output/BigMartSales_Siva_17_Submission.csv", row.names = F)

######## DATA ANALYSIS ##########

names(train)

db <- train
db[,Item_Weight_NA:= ifelse(is.na(Item_Weight),0,1)]

table(db$Item_Weight_NA,db$Outlet_Type)
db1 <- db[,.(Total_Sales = sum(Item_Outlet_Sales, na.rm = T)), by = .(Outlet_Type, Item_Weight_NA)]
db[Outlet_Type == "Grocery Store" & !is.na(Item_Weight), 
   .(Mean_Weight = mean(Item_Weight, na.rm=T),
     Median_Weight = median(Item_Weight, na.rm = T),
     Min_Weight = min(Item_Weight, na.rm=T),
     Max_Weight = max(Item_Weight, na.rm=T)), by= Item_Type]

db[,Item_Weight_Fixed1 := mean(Item_Weight, na.rm = T), Item_Identifier][,Item_Weight_Fixed:= ifelse(is.na(Item_Weight),Item_Weight_Fixed1,Item_Weight)]

head(db[,.(Item_Identifier, Item_Weight, Item_Weight_Fixed, Item_Weight_Fixed1)], 30)

db[,Item_Fat_Content_Fixed := ifelse(Item_Fat_Content %in% c("Regular", "reg"),"Regular","Low Fat")]

View(db[,Item_Fat_Content, Item_Fat_Content_Fixed])

