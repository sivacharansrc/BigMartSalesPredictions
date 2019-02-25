import sys
sys.modules[__name__].__dict__.clear()

import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('expand_frame_repr', False)
import numpy as np
from sklearn.linear_model import LinearRegression

#df = pd.read_csv("~\\Git Clones\\BigMartSalesPredictions\\src\\train.csv")
df = pd.read_csv("~\\Documents\\Python Projects\\BigMartSalesPredictions\\src\\train.csv")
# df = pd.read_csv("~\\Documents\\Python Projects\\BigMartSalesPredictions\\src\\test.csv")
df.head()

# PERFORMING UNI-VARIATE ANALYSIS

# UNIVARIATE ANALYSIS FOR ITEM WEIGHT
plt.subplot(1,2,1)
df['Item_Weight'].plot.box()
plt.subplot(1,2,2)
df['Item_Weight'].plot.hist()
plt.show()
df['Item_Weight'].isnull().value_counts()

# OBSERVATIONS
# Box Plot - The distribution looks fine. There are no outliers. Range is from 5 to 23
# Histogram - Kind of right skewed, but not very apparent. The transformation doesn't look to help very much here
# Missing Items - There are missing items in this variable that needs to be fixed

# Final - Fix Missing Values

# UNIVARIATE ANALYSIS FOR Item_Visibility
plt.subplot(1,2,1)
df['Item_Visibility'].plot.box()
plt.subplot(1,2,2)
df['Item_Visibility'].plot.hist()
plt.show()
df['Item_Visibility'].isnull().value_counts()

# OBSERVATIONS
# Box Plot - There are many outliers in the data. Any data above 0.196 looks as outlier. No lower level outliers though
# Histogram - Right Skewed. Log will not work as the min value is 0, and log (0) will be Inf. Sqrt transformation works
# Missing Values - There are no missing values

# Final - Fix outliers and data distribution

# UNI-VARIATE ANALYSIS FOR Item_MRP
plt.subplot(1,2,1)
df['Item_MRP'].plot.box()
plt.subplot(1,2,2)
df['Item_MRP'].plot.hist()
plt.show()
df['Item_MRP'].isnull().value_counts()


# OBSERVATIONS
# Box Plot - Data looks good without any outliers
# Histogram - Not a good normal distribution, but no action required
# Missing Values - No missing values

# Final - No action required

# UNI-VARIATE ANALYSIS FOR Outlet_Establishment_Year
plt.subplot(1,2,1)
df['Outlet_Establishment_Year'].plot.box()
plt.subplot(1,2,2)
df['Outlet_Establishment_Year'].plot.hist()
plt.show()
df['Outlet_Establishment_Year'].isnull().value_counts()

# OBSERVATIONS
# Box Plot - No issues
# Histogram - No issues
# Missing Values = No missing values

# Final - No issue with data. However, feature engineering can be performed for something like "No of Years in Market"
# since the establishment year does not much much sense for the outlet sales


# UNI-VARIATE ANALYSIS FOR ITEM WEIGHT
plt.subplot(1, 2, 1)
df['Item_Weight'].plot.box()
plt.subplot(1, 2, 2)
df['Item_Weight'].plot.hist()
plt.show()

# OBSERVATIONS
# Box Plot - The distribution looks fine. There are no outliers. Range is from 5 to 23
# Histogram - Kind of right skewed, but not very apparent. The transformation doesn't look to help very much here

# CATEGORICAL VARIABLES - Item_Fat_Content, Item_Type,  Outlet_Size, Outlet_Location_Type, Outlet_Type

# UNI-VARIATE ANALYSIS FOR Item_Fat_Content
df['Item_Fat_Content'].isnull().value_counts()
df['Item_Fat_Content'].value_counts()

# OBSERVATIONS
# Missing Values - No missing values
# Variable Transformation - Variable Transformation should be done.
#  Low Fat, LF, low fat = Low Fat, Regular and reg = Regular

# UNI-VARIATE ANALYSIS FOR Item_Type
df['Item_Type'].isnull().value_counts()
df['Item_Type'].value_counts()

# OBSERVATIONS
# Missing Values - No missing values
# Variable Transformation - See if feature engineering can be done like categorizing foods to healthy, junk foods etc

# UNI-VARIATE ANALYSIS FOR Outlet_Size
df['Outlet_Size'].isnull().value_counts()
df['Outlet_Size'].value_counts()

# OBSERVATIONS
# Missing Values - Almost 30 % data missing
# Variable Transformation - check if missing outlet size can be determined by finding the average sales
# by outlet size/ outlet location type

# UNI-VARIATE ANALYSIS FOR Outlet_Location_Type
df['Outlet_Location_Type'].isnull().value_counts()
df['Outlet_Location_Type'].value_counts()

# OBSERVATIONS
# Missing Values - No missing values
# Variable Transformation - Not applicable

# UNI-VARIATE ANALYSIS FOR Outlet_Type
df['Outlet_Type'].isnull().value_counts()
df['Outlet_Type'].value_counts()

# OBSERVATIONS
# Missing Values - No missing values
# Variable Transformation - Not Applicable

# FIXING ALL VARIABLES AFTER UNI-VARIATE ANALYSIS

# Item_Weight - Missing Values
# Item_Visibility - Perform sqrt transformation. Any point > 0.196 is outlier
# Outlet_Establishment_Year - Feature Engineering for No of Years in Market
# Item_Fat_Content - Variable transform the values
# Item_Type - Check for feature engineering like healthy foods, junk foods etc
# Outlet_Size - 30% missing data. Can outlet size be determined using the item_outlet_sales?

def feature_engineer (df):
    df_clean = df
    df_clean = df_clean.reset_index(drop=True)
    # FIXING MISSING VALUES FOR ITEM WEIGHT

    df_clean['Item_Identifier'][df_clean['Item_Weight'].isnull()].head()
    df_clean[['Item_Identifier', 'Item_Weight']][df_clean['Item_Identifier'].isin(['FDP10', 'DRI11', 'FDW12', 'FDC14'])].head(20)
    # From the above analysis, the item_weight is directly related to the Item_Identifier. So, the weight can be obtained from Identifier
    df_clean['Item_Weight_varTransformed'] = df_clean['Item_Weight']
    item_weight_container = df_clean[['Item_Identifier', 'Item_Weight_varTransformed']][~df_clean['Item_Weight_varTransformed'].isnull()].drop_duplicates()
    df_missing = df_clean[df_clean['Item_Weight_varTransformed'].isnull()]
    df_missing = df_missing.drop(columns='Item_Weight_varTransformed')
    df_clean = df_clean[~df_clean.index.isin(df_missing.index)]
    df_missing = df_missing.merge(item_weight_container, how='left', on='Item_Identifier')
    df_clean = df_clean.append(df_missing, sort=True)
    df_clean.loc[df_clean['Item_Weight_varTransformed'].isnull(), 'Item_Weight_varTransformed'] = df_clean['Item_Weight_varTransformed'].mean()
    # df_clean['Item_Weight_varTransformed'].plot.hist()
    df_clean.reset_index(drop=True, inplace=True)
    # THERE ARE NO MISSING VALUES NOR OUTLIERS IN THE ITEM_WEIGHT TRANSFORMED DATA


    # FIXING OUTLIERS FOR ITEM VISIBILITY AND PERFORMING SQRT TRANSFORMATION
    # item_visibility_mean = df_clean.groupby('Item_Type')['Item_Visibility'].mean().reset_index()
    df_clean['Item_Visibility_varTransformed'] = df_clean['Item_Visibility']
    Item_Visibility_varTransformed_median = df_clean.groupby('Item_Type').agg({'Item_Visibility_varTransformed': 'median'}).reset_index()  # This is good for multiple columns
    lower_limit = np.percentile(df_clean['Item_Visibility_varTransformed'], 25) - (1.5 * (np.percentile(df_clean['Item_Visibility_varTransformed'], 75) - np.percentile(df_clean['Item_Visibility_varTransformed'], 25)))
    upper_limit = np.percentile(df_clean['Item_Visibility_varTransformed'], 75) + (1.5 * (np.percentile(df_clean['Item_Visibility_varTransformed'], 75) - np.percentile(df_clean['Item_Visibility_varTransformed'], 25)))
    df_item_visibility_outliers = df_clean[(df_clean['Item_Visibility_varTransformed'] < lower_limit) | (df_clean['Item_Visibility_varTransformed'] > upper_limit)]
    df_clean = df_clean[~ df_clean.index.isin(df_item_visibility_outliers.index)]
    df_item_visibility_outliers = df_item_visibility_outliers.drop(columns='Item_Visibility_varTransformed')
    df_item_visibility_outliers = df_item_visibility_outliers.merge(Item_Visibility_varTransformed_median, on='Item_Type', how='left')
    df_clean = df_clean.append(df_item_visibility_outliers, sort=True)
    lower_limit = np.percentile(df_clean['Item_Visibility_varTransformed'], 25) - (1.5 * (np.percentile(df_clean['Item_Visibility_varTransformed'], 75) - np.percentile(df_clean['Item_Visibility_varTransformed'], 25)))
    upper_limit = np.percentile(df_clean['Item_Visibility_varTransformed'], 75) + (1.5 * (np.percentile(df_clean['Item_Visibility_varTransformed'], 75) - np.percentile(df_clean['Item_Visibility_varTransformed'], 25)))
    # df_clean[(df_clean['Item_Visibility_varTransformed'] < lower_limit) | (df_clean['Item_Visibility_varTransformed'] > upper_limit)].__len__() # CHECKING THE NUMBER OF OUTLIERS
    df_clean['Item_Visibility_varTransformed'] = np.sqrt(df_clean['Item_Visibility_varTransformed'])
    df_clean.reset_index(drop=True, inplace=True)
    # THE OUTLIERS ARE REDUCED FROM 144 TO 29. ALSO THE ACTUAL DISTRIBUTION WAS RIGHT SKEWED. THIS IS NOW FIXED BY TAKING THE SQRT TRANSFORMATION

    # FEATURE ENGINEERING FOR OUTLET ESTABLISHMENT YEAR

    df_clean['No_Of_Years_Established_varTransformed'] = 2013 - df_clean['Outlet_Establishment_Year']
    df_clean[['Outlet_Identifier', 'Outlet_Establishment_Year', 'No_Of_Years_Established_varTransformed']].head(30)
    df_clean['No_Of_Years_Established_varTransformed'].isnull().value_counts()

    # THE NO OF YEARS ESTABLISHED CAN BE USED IN MODEL INSTEAD OF ESTABLISHMENT YEAR. THERE ARE NO MISSING VALUES AND THIS FEATURE IS GOOD TO BE USED

    # FIX THE ITEM FAT CONTENT VARIABLE

    df_clean['Item_Fat_Content_varTransformed'] = df_clean['Item_Fat_Content']
    df_clean['Item_Fat_Content_varTransformed'].value_counts()
    df_clean.loc[df_clean['Item_Fat_Content_varTransformed'].isin(['LF', 'low fat']), 'Item_Fat_Content_varTransformed'] = 'Low Fat'
    df_clean.loc[df_clean['Item_Fat_Content_varTransformed'] == 'reg', 'Item_Fat_Content_varTransformed'] = 'Regular'
    df_clean['Item_Fat_Content_varTransformed'].value_counts()

    # THE TYPOS IN THE ITEM FAT CONTENT VARIABLE IS SUCCESSFULLY FIXED

    # CHECK FEATURE ENGINEERING FOR FEATURE TYPE - PENDING

    # OUTLET SIZE DATA IS MISSING. CHECK FROM OUTLET IDENTIFIER OR USING ITEM OUTLET SALES AS A PROXY
    df_clean['Outlet_Size_varTransformed'] = df_clean['Outlet_Size']
    # df_clean['Outlet_Identifier'][df_clean['Outlet_Size'].isnull()].head(20)
    # pd.crosstab(df_clean['Outlet_Type'], df_clean['Outlet_Size_varTransformed'])
    # From the cross tab, we can assume that all Grocery Stores are small, Supermarket2 and 3 are mostly medium, while Supermarket 1 can be of any size
    df_clean.loc[(df_clean['Outlet_Size_varTransformed'].isnull()) & (df_clean['Outlet_Type'] == 'Grocery Store'), 'Outlet_Size_varTransformed'] = 'Small'
    df_clean.loc[(df_clean['Outlet_Size_varTransformed'].isnull()) & (df_clean['Outlet_Type'].isin(['Supermarket Type3', 'Supermarket Type2'])),
                 'Outlet_Size_varTransformed'] = 'Medium'

    # WE YET NEED TO FIX THE VALUES FOR SUPERMARKET TYPE1
    # df_clean.groupby(['Outlet_Size_varTransformed', 'Outlet_Type', 'Outlet_Location_Type']).agg({'Item_Outlet_Sales': 'sum'}).reset_index()
    # NO CONCLUSION CAN BE REACHED WITH THE OUTLET SALES

    pd.crosstab(df_clean['Outlet_Size_varTransformed'][df_clean['Outlet_Type'] == 'Supermarket Type1'], df_clean['Outlet_Location_Type'][df_clean['Outlet_Type'] == 'Supermarket Type1'])
    # FROM THE NUMBERS LET US ASSUME THAT TIER 3 IS HIGH, TIER 2 IS MEDIUM AND TIER 1 IS SMALL
    # df_clean['Outlet_Size_varTransformed'][df_clean['Outlet_Type'] == 'Supermarket Type2'].mode()[0]
    # df_clean[['Outlet_Identifier', 'Outlet_Location_Type']][df_clean['Outlet_Size_varTransformed'].isnull()].drop_duplicates()

    df_clean.loc[(df_clean['Outlet_Size_varTransformed'].isnull()) & (df_clean['Outlet_Location_Type'] == 'Tier 1'), 'Outlet_Size_varTransformed'] = 'Small'
    df_clean.loc[(df_clean['Outlet_Size_varTransformed'].isnull()) & (df_clean['Outlet_Location_Type'] == 'Tier 2'), 'Outlet_Size_varTransformed'] = 'Medium'
    df_clean.loc[(df_clean['Outlet_Size_varTransformed'].isnull()) & (df_clean['Outlet_Location_Type'] == 'Tier 3'), 'Outlet_Size_varTransformed'] = 'High'

    # ALL VARIABLES ARE FIXED FOR OUTLIERS AND MISSING DATA

    df_final = df_clean.drop(['Item_Identifier', 'Outlet_Identifier', 'Item_Fat_Content', 'Item_Visibility', 'Item_Weight', 'Outlet_Establishment_Year',
                             'Outlet_Size'], 1)
    df_final.reset_index(drop=True, inplace=True)
    return (df_final);


df_final = feature_engineer(df)
df_final.drop(['Item_Visibility_varTransformed', 'Item_Weight_varTransformed'], 1, inplace=True)
df_final.drop(['Item_Type'], 1, inplace=True)
df_final.head()
df_final = pd.get_dummies(df_final)

# SAMPLING DATA

def  sampleSplit(df):
    x_train = df.sample(frac=0.8, random_state=1)
    x_test = df.loc[~df_final.index.isin(x_train.index)]
    y_train = x_train['Item_Outlet_Sales']
    x_train = x_train.drop('Item_Outlet_Sales', 1)
    y_test = x_test['Item_Outlet_Sales']
    x_test = x_test.drop('Item_Outlet_Sales', 1)
    return x_train, y_train, x_test, y_test;

x_train, y_train, x_test, y_test = sampleSplit(df_final)

# PERFORMING LINEAR REGRESSION ON THE DATA SET


lm = LinearRegression()

# x_train.fillna(0, inplace=True)
# y_test.fillna(0, inplace=True)

lm.fit(x_train, y_train)
predictions = lm.predict(x_test)
predictions[predictions < 0] = 0


# Performance of the model:

r_square_test = lm.score(x_test, y_test)  # r square is 0.5681433001673957
r_square_train = lm.score(x_train, y_train)  # r square is 0.561736750416446
rmse_test = np.sqrt(np.mean(np.power((np.array(y_test) - np.array(predictions)), 2)))  # 1154.6830929638688
rmse_train = np.sqrt(np.mean(np.power((np.array(y_train) - np.array(lm.predict(x_train))), 2)))  # 1121.0895369240686

print("R Square Test Data: " + str(r_square_test) + ", R Square Train Data: " + str(r_square_train) + "\n RMSE Test Data: " + str(rmse_test) + ", RMSE Train Data: " + str(rmse_train))

# PREPARING TEST DATA

test_data = feature_engineer(df)
test_data.drop(['Item_Type'], 1, inplace=True)
df_final.head()
test_data = pd.get_dummies(test_data)
test_pred = lm.predict(test_data)
test_pred[test_pred < 0] = 0

outputFile = df[['Item_Identifier', 'Outlet_Identifier']]
outletSales = pd.DataFrame({'Item_Outlet_Sales': test_pred})
final_data = outputFile.join(outletSales)

outputFile.to_csv("~\\Documents\\Python Projects\\BigMartSalesPredictions\\Output\\BigMartSales_Siva_18_Submission.csv")
outletSales.to_csv("~\\Documents\\Python Projects\\BigMartSalesPredictions\\Output\\BigMartSales_Siva_19_Submission.csv")

plt.subplot(1, 2, 1)
plt.hist(y_test)
plt.subplot(1, 2, 2)
plt.hist(predictions)
plt.show()