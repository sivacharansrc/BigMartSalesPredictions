# REFERENCES
# https://www.analyticsvidhya.com/blog/2016/05/h2o-data-table-build-models-large-data-sets/
# https://www.analyticsvidhya.com/blog/2015/10/regression-python-beginners/
# https://www.analyticsvidhya.com/blog/2016/04/complete-tutorial-tree-based-modeling-scratch-in-python/
# https://www.analyticsvidhya.com/blog/2016/03/introduction-deep-learning-fundamentals-neural-networks/

###### PREPARING ENVIRONMENT ####
#install.packages(c("h2o", "dummies"))
rm(list = ls())
options(scipen = 999)
library(data.table)
library(dplyr)
#library(ggplot2)
library(summaryR)
library(h2o)
#library(plotly)
library(dummies)

### DEFINING MODE FUNCTION 

Mode <- function(x, na.rm = TRUE) {
  if(na.rm){
    x = x[!is.na(x)]
  }
  
  ux <- unique(x)
  return(ux[which.max(tabulate(match(x, ux)))])
}


summariseCategCols <- function(x,y){
  factCols <- NULL
  df <- NULL
  df1 <- NULL
  z <- x
  x <- as.data.frame(x)
  for (i in 1:ncol(x)) {
    if(class(x[,i]) %in% c("factor","character")){
      factCols <- c(names(x[i]),factCols)
    }
  }
  for (i in 1:length(factCols)){
    colName <- factCols[i]
    respVar <- which(colnames(x) == y)
    if(length(unique(x[[which(colnames(x) == colName)]])) < 51) {
      #df1 <-  z[,.(Total_Sum = sum(z[,respVar], na.rm=T), Total_Count = .N), colName]
      df1 <-  z[,.(Total_Count = .N, Count_Percent = round((.N*100/nrow(z)),0)), colName]
      names(df1)[1] <- "Category_Values"
      df1$Category <- colName
      df1 <- df1[,c(4,1,2,3)]
      df <- rbind(df,df1)
    }
  }
  return(df)
}

#summariseCategCols(train, "Item_Outlet_Sales")


####### READING DATA ######
# View(t(summaryR(data.frame(train)))) | View(t(summaryR(data.frame(test)))) | View(head(train))


train <- fread("./src/train.csv", sep = ",", header = T, na.strings = c("",NA," ","NA", "N/A"))
test <- fread("./src/test.csv", sep = ",", header = T, na.strings = c("",NA," ","NA", "N/A"))


### DATA CLEANING ####

# CLEANING UP TRAIN DATA TO PICK ONLY THE ITEM IDENTIFIERS AVAILABLE IN TEST

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

##  names(cleanData) View(head(cleanData))  summary(cleanData)  str(cleanData) View(cleanData)

## RECODING VALUES IN THE ITEM FAT CONTENT

cleanData[,Item_Fat_Content_Fixed := ifelse(Item_Fat_Content %in% c("Regular", "reg"),"Regular","Low Fat")]

#### IMPUTING MISSING VALUES  ####

# OUTLET SIZE AND ITEM WEIGHT HAS MISSING VALUES
# OUTLET SIZE = MODE OF OUTLET SIZE BY OUTLET TYPE
# ITEM WEIGHT = MEAN OF ITEM WEIGHT BY ITEM TYPE
# ITEM VISIBIILTY DOES NOT HAVE MISSING VALUES BUT HAS 0 WHICH CANNOT BE THE CASE. 
# CONVERTING ALL ITEM VISIBILITY WITH 0 AS NA, AND IMPUTING THE MISSING VALUES BY CALCULATING MEAN OF ITEM VISIBILITY BY OUTLET AND ITEM TYPE


### OUTLET SIZE
cleanData[,Outlet_Size_Fixed := ifelse(!is.na(Outlet_Size), Outlet_Size, Mode(Outlet_Size)), Outlet_Type]


### ITEM WEIGHT
cleanData[,Item_Weight_Fixed1 := mean(Item_Weight, na.rm = T), Item_Identifier][,Item_Weight_Fixed:= ifelse(is.na(Item_Weight),Item_Weight_Fixed1,Item_Weight)]

### ITEM VISIBILITY

cleanData[Item_Visibility == 0, Item_Visibility := NA]
cleanData[,Item_Visibility_Fixed := ifelse(!is.na(Item_Visibility), Item_Visibility, mean(Item_Visibility, na.rm = T)), .(Outlet_Identifier, Item_Type)]


### FEATURE ENGINEERING ####

cleanData[, Item_Type_New := ifelse(Item_Type %in% c("Soft Drinks", "Hard Drinks"),
                                    "Drinks",
                                    ifelse(Item_Type %in% c("Household", "Baking Goods","Health and Hygiene", "Others"),
                                           "Others", "Food"))]

## CREATING NA REFERENCE COLS WHEREVER APPLICABLE
cleanData[, Outlet_Size_NA := ifelse(is.na(Outlet_Size),0,1)][, Item_Visibility_NA := ifelse(is.na(Item_Visibility),0,1)][, Item_Weight_NA := ifelse(is.na(Item_Weight),0,1)]

## CREATING NEW COLUMN - NO OF YRS THE OUTLET IS OPERATIONAL
cleanData[,No_of_Yrs := 2013 - Outlet_Establishment_Year]

## HOT ENCODING
cleanData <- data.table(dummy.data.frame(cleanData, names = c("Item_Fat_Content_Fixed", "Outlet_Type", "Outlet_Location_Type", "Outlet_Size_Fixed", "Item_Type_New"), sep="_"))

cleanData$Item_Identifier <- as.integer(as.factor(cleanData$Item_Identifier))

cleanData$Item_Type <- as.integer(as.factor(cleanData$Item_Type))

cleanData$Outlet_Identifier <- as.integer(as.factor(cleanData$Outlet_Identifier))

#cleanData[,Sales_By_Item_Type := sum(Item_Outlet_Sales, na.rm = T),Item_Type]

# View(cleanData) names(cleanData)

cleanData[, c("Outlet_Establishment_Year", "Item_Weight_Fixed1", "Item_Weight", "Item_Fat_Content",
              "Item_Visibility", "Outlet_Size") := NULL]


## SEPERATING TRAIN AND TEST CLEAN DATA   View(head(cleanData))

clean.train <- cleanData[1:nrow(clean.train),]
clean.test <- cleanData[-(1:nrow(clean.train)),]

####### PREDICTION MODELS #########

### PERFORMING LINEAR REGRESSION FOR THE DATASET

# INITIATING H2O

# localH2o <- h2o.init(nthreads = -1)
# rm(localH2)
# localH2O <- h2o.init(ip='localhost', nthreads=-1,
#                     min_mem_size='8G', max_mem_size='12G')
# h2o.shutdown(prompt = T)

# CONVERTING DATA SET TO H2O FORMAT | names(train.h2o) names(train) head(train.h2o)
train.h2o <- as.h2o(clean.train)
test.h2o <- as.h2o(clean.test)

## SETTING COLUMN POSITIONS TO DEP AND INDEP VARIABLES

xIndep <- c(1:4, 8:11,13:19, 25,26)
yDep <- 12

## RANDOM FOREST
rm(rf.model)
rf.model <- h2o.randomForest(y=yDep, x=xIndep, training_frame = train.h2o,
                             ntrees = 500, mtries = 10, max_depth = 20, seed = 1001)

# View(h2o.varimp(rf.model))

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
                     ntrees = 1400, max_depth = 16, learn_rate = 0.1, seed = 100)


h2o.performance(gbm.model)

## PREDICT TEST USING THE GBM MODEL

gbm.pred <- as.data.frame(h2o.predict(gbm.model, test.h2o))

#View(head(gbm.pred))

df <- data.frame(Item_Identifier = clean.train$Item_Identifier,
                 Outlet_Identifier = clean.train$Outlet_Identifier,
                 Actual_Sales = clean.train$Item_Outlet_Sales,
                 Predicted_Sales = gbm.pred$predict)

df <- data.frame(Item_Identifier = test$Item_Identifier,
                 Outlet_Identifier = test$Outlet_Identifier,
                 Item_Outlet_Sales = gbm.pred$predict)

train[, .(Avg_Sales = mean(Item_Outlet_Sales)), Outlet_Identifier][order(Outlet_Identifier)]
data.table(df)[, .(Avg_Sales = mean(Item_Outlet_Sales)), Outlet_Identifier][order(Outlet_Identifier)]
df$


#write.csv(df, "./Output/BigMartSales_Siva_18_Submission.csv", row.names = F)

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

