# Function definition is here
def feature_engineer( df ):
    df_clean = df
    df_clean = df_clean.reset_index(drop=True)
    # FIXING MISSING VALUES FOR ITEM WEIGHT
    df_clean['Item_Weight_varTransformed'] = df_clean['Item_Weight']
    item_weight_container = df_clean[['Item_Identifier', 'Item_Weight_varTransformed']][~df_clean['Item_Weight_varTransformed'].isnull()].drop_duplicates()
    df_missing = df_clean[df_clean['Item_Weight_varTransformed'].isnull()]
    df_missing = df_missing.drop(columns='Item_Weight_varTransformed')
    df_clean = df_clean[~df_clean.index.isin(df_missing.index)]
    df_missing = df_missing.merge(item_weight_container, how='left', on='Item_Identifier')
    df_clean = df_clean.append(df_missing, sort=True)
    df_clean.loc[df_clean['Item_Weight_varTransformed'].isnull(), 'Item_Weight_varTransformed'] = df_clean['Item_Weight_varTransformed'].mean()
    df_clean.reset_index(drop=True, inplace=True)
    # THERE ARE NO MISSING VALUES NOR OUTLIERS IN THE ITEM_WEIGHT TRANSFORMED DATA

    # FIXING OUTLIERS FOR ITEM VISIBILITY AND PERFORMING SQRT TRANSFORMATION
    df_clean['Item_Visibility_varTransformed'] = df_clean['Item_Visibility']
    Item_Visibility_varTransformed_median = df_clean.groupby('Item_Type').agg({'Item_Visibility_varTransformed': 'median'}).reset_index()
    lower_limit = np.percentile(df_clean['Item_Visibility_varTransformed'], 25) - (1.5 * (np.percentile(df_clean['Item_Visibility_varTransformed'], 75) - np.percentile(df_clean['Item_Visibility_varTransformed'], 25)))
    upper_limit = np.percentile(df_clean['Item_Visibility_varTransformed'], 75) + (1.5 * (np.percentile(df_clean['Item_Visibility_varTransformed'], 75) - np.percentile(df_clean['Item_Visibility_varTransformed'], 25)))
    df_item_visibility_outliers = df_clean[(df_clean['Item_Visibility_varTransformed'] < lower_limit) | (df_clean['Item_Visibility_varTransformed'] > upper_limit)]
    df_clean = df_clean[~ df_clean.index.isin(df_item_visibility_outliers.index)]
    df_item_visibility_outliers = df_item_visibility_outliers.drop(columns='Item_Visibility_varTransformed')
    df_item_visibility_outliers = df_item_visibility_outliers.merge(Item_Visibility_varTransformed_median, on='Item_Type', how='left')
    df_clean = df_clean.append(df_item_visibility_outliers, sort=True)
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

    # THE TYPOS IN THE ITEM FAT CONTENT VARIABLE IS SUCCESSFULLY FIXED

    # CHECK FEATURE ENGINEERING FOR FEATURE TYPE - PENDING

    # OUTLET SIZE DATA IS MISSING. CHECK FROM OUTLET IDENTIFIER OR USING ITEM OUTLET SALES AS A PROXY
    df_clean['Outlet_Size_varTransformed'] = df_clean['Outlet_Size']
    df_clean.loc[(df_clean['Outlet_Size_varTransformed'].isnull()) & (df_clean['Outlet_Type'] == 'Grocery Store'), 'Outlet_Size_varTransformed'] = 'Small'
    df_clean.loc[(df_clean['Outlet_Size_varTransformed'].isnull()) & (df_clean['Outlet_Type'].isin(['Supermarket Type3', 'Supermarket Type2'])), 'Outlet_Size_varTransformed'] = 'Medium'

    # WE YET NEED TO FIX THE VALUES FOR SUPERMARKET TYPE1

    df_clean.loc[(df_clean['Outlet_Size_varTransformed'].isnull()) & (
                df_clean['Outlet_Location_Type'] == 'Tier 1'), 'Outlet_Size_varTransformed'] = 'Small'
    df_clean.loc[(df_clean['Outlet_Size_varTransformed'].isnull()) & (
                df_clean['Outlet_Location_Type'] == 'Tier 2'), 'Outlet_Size_varTransformed'] = 'Medium'
    df_clean.loc[(df_clean['Outlet_Size_varTransformed'].isnull()) & (
                df_clean['Outlet_Location_Type'] == 'Tier 3'), 'Outlet_Size_varTransformed'] = 'High'

    # ALL VARIABLES ARE FIXED FOR OUTLIERS AND MISSING DATA

    df_final = df_clean[['Item_Identifier', 'Item_MRP', 'Item_Outlet_Sales', 'Item_Visibility_varTransformed',
                         'Item_Weight_varTransformed', 'Item_Type',
                         'Outlet_Identifier', 'Outlet_Location_Type', 'Outlet_Type',
                         'No_Of_Years_Established_varTransformed',
                         'Item_Fat_Content_varTransformed', 'Outlet_Size_varTransformed']]
    return (df_final);

df = pd.read_csv("~\\Documents\\Python Projects\\BigMartSalesPredictions\\src\\train.csv")
# df = pd.read_csv("~\\Documents\\Python Projects\\BigMartSalesPredictions\\src\\test.csv")

train_data = feature_engineer(df)

train_data.head()