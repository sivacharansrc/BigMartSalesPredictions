import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('expand_frame_repr', False)
import numpy as np

df = pd.read_csv("~\\Git Clones\\BigMartSalesPredictions\\src\\train.csv")
df = pd.read_csv("~\\Documents\\Python Projects\\BigMartSalesPredictions\\src\\train.csv")
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

df_clean = df

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

df_clean['Item_Weight_varTransformed'].plot.hist()

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

# THE OUTLIERS ARE REDUCED FROM 144 TO 29. ALSO THE ACTUAL DISTRIBUTION WAS RIGHT SKEWED. THIS IS NOW FIXED BY TAKING THE SQRT TRANSFORMATION