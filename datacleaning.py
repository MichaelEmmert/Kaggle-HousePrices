import numpy as np
import pandas as pd
from houseprice_functions import ratings_to_ord

#read_csv
HousePrices = pd.read_csv("data/test.csv")

################### Mike code
#electrical | converted to ordinal | imputing NA to 0
elec_ = {'SBrkr':5, 'FuseF':3, 'FuseA':4, 'FuseP':2, 'Mix':1, 'Na':0}
HousePrices.Electrical = HousePrices.Electrical.fillna('Na')
HousePrices.Electrical = HousePrices.Electrical.apply(lambda x: list(elec_.values())[list(elec_.keys()).index(x)])

#Central Air | converted to bool | NO missingness
HousePrices.CentralAir = HousePrices.CentralAir.apply(lambda x: 1 if x == 'Y' else 0)

#Heating | Bool Gas or not | No missingness
heating_ = {"GasA":1,"GasW":1,"Floor":0,"Grav":0,"OthW":0,"Wall":0}
HousePrices.Heating = HousePrices.Heating.apply(lambda x: list(heating_.values())[list(heating_.keys()).index(x)])

#HeatingQC | Ordinal Categorical | No missingness
ratings_to_ord(df = HousePrices,col = 'HeatingQC',inplace = True)

#### Garage ####
#Garage Quality and condition comb | ordingal catergorical | Na Mapped to 0 assuming NA are no garage
ratings_to_ord(df = HousePrices,col = 'GarageQual',inplace = True)
ratings_to_ord(df = HousePrices,col = 'GarageCond',inplace = True)
HousePrices['garage_score'] = HousePrices.GarageCond + HousePrices.GarageQual

#Garage Finish - UNUSED
gfin = {"Fin":1,"RFn":1,"Unf":0,"Na":0}
HousePrices.GarageFinish = HousePrices.GarageFinish.fillna('Na')
HousePrices.GarageFinish = HousePrices.GarageFinish.apply(lambda x: list(gfin.values())[list(gfin.keys()).index(x)])

# Garage Area
HousePrices['GarageArea'] = HousePrices['GarageArea'].fillna(0)


#Garage Type | Dummified - dropping 'Attchd'| NA converted to no garage
HousePrices.GarageType = HousePrices.GarageType.fillna('No_garage')
garage_type_dummy = pd.get_dummies(HousePrices.GarageType).drop('Attchd',axis = 1)

#### Basement ####

##########
#### TotalBsmtSF
HousePrices['TotalBsmtSF'] = HousePrices['TotalBsmtSF'].fillna(0)
#### GRLivArea
HousePrices['GrLivArea'] = HousePrices['GrLivArea'].fillna(0)
#HousePrices['TotalSF'] = HousePrices['GrLivArea'] + HousePrices['TotalBsmtSF']
##########

# BsmtUnfSF Finished/ unfished basementv | percent between 0 and 1 | if Na, zero percent
HousePrices['finishedbsmt'] = 1 - HousePrices['BsmtUnfSF']/HousePrices['TotalBsmtSF']
HousePrices['finishedbsmt'] = HousePrices['finishedbsmt'].fillna(0) #to avoid divide by zero error


##### 
#Fence | Dummy - dropped no fence | imputed NA to mean no fence
fence_dict = {'MnPrv':'b_fence','MnWw':'b_fence','Na':'n_fence','GdWo':'g_fence','GdPrv':'g_fence'}
HousePrices.Fence = HousePrices.Fence.fillna('Na')
HousePrices.Fence = HousePrices.Fence.apply(lambda x: list(fence_dict.values())[list(fence_dict.keys()).index(x)])
fence_dummy = pd.get_dummies(HousePrices.Fence).drop('n_fence',axis = 1)

cleaned_columns = HousePrices[['Id','CentralAir','HeatingQC','garage_score','Heating','Electrical',\
                               'GarageArea','finishedbsmt','GrLivArea','TotalBsmtSF']]
cleaned_columns = cleaned_columns.merge(garage_type_dummy,how = "outer",left_index=True,right_index=True)
cleaned_columns_ME = cleaned_columns.merge(fence_dummy,how = "outer",left_index=True,right_index=True)

### Notes ###
# TotalBsmtSF - nothing done to it
# GarageCars / GarageArea - Garage Area only
# g_fence is good fence b_fence is bad fence

#garageT_ = {'Attchd':, 'Detchd':1, 'BuiltIn':2, 'CarPort':1, 'Na':0, 'Basment':2, '2Types':2}
#HousePrices.GarageType = HousePrices.GarageType.fillna('Na')
#HousePrices.GarageType = HousePrices.GarageType.apply(lambda x: list(garageT_.values())[list(garageT_.keys()).index(x)]

############## Vince Code
#Select continuous features in the living and recreation category
Living_Rec_Cont = HousePrices[["Id",
"BedroomAbvGr",
"BsmtFullBath",
"BsmtHalfBath",
"FullBath",
"HalfBath",
# Features dropped
# "1stFlrSF",
# "2ndFlrSF",
# "TotRmsAbvGrd",
# "KitchenAbvGr", 
# "PoolArea",
# "WoodDeckSF",
]]

# Total bath room - basement bathrooms are counted 75% of normal bathrooms
# Then remove bathroom cols that are no longer needed


Living_Rec_Cont["BsmtFullBath"] = Living_Rec_Cont["BsmtFullBath"].fillna(0)
Living_Rec_Cont["BsmtHalfBath"] = Living_Rec_Cont["BsmtHalfBath"].fillna(0)
Living_Rec_Cont["HalfBath"] = Living_Rec_Cont["HalfBath"].fillna(0)
Living_Rec_Cont["FullBath"]  = Living_Rec_Cont["FullBath"].fillna(0)
Living_Rec_Cont["TotalBath"] = Living_Rec_Cont["BsmtFullBath"]*0.75 + Living_Rec_Cont["FullBath"] + \
                               Living_Rec_Cont["BsmtHalfBath"]*0.75 + Living_Rec_Cont["HalfBath"]

Living_Rec_Cont_Final = Living_Rec_Cont.drop(["BsmtFullBath","BsmtHalfBath","FullBath","HalfBath"], axis =1)

#Select continuous features in the living and recreation category

Living_Rec_Cat_Raw = HousePrices[["Id","KitchenQual","BsmtCond","BsmtFinType1","BsmtFinType2","BsmtExposure",
"BsmtQual","PoolQC","PoolArea"]]


# Convert Pool and Pool_Ex (whether in exellent condition) to binary
Living_Rec_Cat = Living_Rec_Cat_Raw.assign(Pool = np.select([Living_Rec_Cat_Raw['PoolArea'] > 0], "1", "0"))
Living_Rec_Cat = Living_Rec_Cat.assign(Pool_Ex = np.select([Living_Rec_Cat['PoolQC'] == "Ex"], "1", "0")) 

# Check null values after convertion - commented out for now
# Living_Rec_Cat.isnull().sum()


##########
#### TotalBsmtSF
HousePrices['TotalBsmtSF'] = HousePrices['TotalBsmtSF'].fillna(0)
HousePrices['GrLivArea'] = HousePrices['GrLivArea'].fillna(0)
HousePrices['TotalSF'] = HousePrices['GrLivArea'] + HousePrices['TotalBsmtSF']
##########

# Fills na with 0
Living_Rec_Cat=Living_Rec_Cat.fillna(0)

# Check null values after fillna - commented out for now
# Living_Rec_Cat.isnull().sum()

# Create a dictionary of ordinal categories 
Bsmt_Desc_Dict = {"Ex":5, "Gd":4 ,"TA":3,"Av":3,"Fa":2, "Mn":2, "Po":1, "No":1, "GLQ":6,"ALQ":5,"BLQ":4,"Rec":3,"LwQ":2,"Unf":1}

# Converts the following columns to ordinal categories
Col_Convert = ["BsmtCond","BsmtFinType1","BsmtFinType2","BsmtExposure","BsmtQual","KitchenQual"]
for col in Col_Convert:
    convert_index = Living_Rec_Cat.loc[Living_Rec_Cat[col]!=0].index
    Living_Rec_Cat.loc[convert_index,col] =  Living_Rec_Cat.loc[convert_index][[col]].apply(lambda x: Bsmt_Desc_Dict.get(x[0]),axis = 1)

# Consolidate new basement related columns
Living_Rec_Cat["Basm_Quality"] = Living_Rec_Cat.BsmtCond + Living_Rec_Cat.BsmtExposure + Living_Rec_Cat.BsmtQual
 
Living_Rec_Cat["BsmtFin_Quality"] = Living_Rec_Cat.BsmtFinType1 + Living_Rec_Cat.BsmtFinType2 


Living_Rec_Cat_Final = Living_Rec_Cat[["Id","KitchenQual","Basm_Quality","BsmtFin_Quality","Pool","Pool_Ex"]]

# Write out the output
living_rec_all = Living_Rec_Cont_Final.merge(Living_Rec_Cat_Final, on = "Id")




############# sunny code
cols = ['Id','ExterQual','ExterCond','OverallQual','OverallCond','MasVnrType',
       'RoofMatl','SaleCondition','OpenPorchSF', 'ScreenPorch', '3SsnPorch',
        'WoodDeckSF','MoSold', 'YrSold', 'EnclosedPorch', 'YearRemodAdd',
       'Exterior1st','HouseStyle']

def exterior_score(dataframe):
    qual_ = {"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1,"Na": 0}
    for col in ['ExterQual','ExterCond']:
        dataframe[col] = dataframe[col].apply(lambda x: int(qual_[x]))
    dataframe['Ext_Score'] = dataframe['ExterQual'] + dataframe['ExterCond']
    

def overall_qual_score(dataframe):
    dataframe['OverallScore'] = dataframe['OverallQual'] + dataframe['OverallCond']
    

def vnr_binary(dataframe):
    dataframe['MasVnrType'] = dataframe['MasVnrType'].fillna('None')
    dataframe['MasVnrType'] = dataframe['MasVnrType'].apply(lambda x: 0 if x=='None' else 1)
    

def roofmat_binary(dataframe):
    dataframe['RoofMatl'] = dataframe['RoofMatl'].fillna('Na')
    dataframe['RoofMatl'] = dataframe['RoofMatl'].apply(lambda x: 1 if x=='CompShg' else 0)
    

def salecond_binary(dataframe):
    dataframe['SaleCondition'] = dataframe['SaleCondition'].fillna('Na')
    dataframe['SaleCondition'] = dataframe['SaleCondition'].apply(lambda x: 1 if x=='Abnorml' else 0)
    

def total_porchdeckSF(dataframe):
    dataframe['Total_PorchDeckSF']  = dataframe['OpenPorchSF'] + dataframe ['ScreenPorch'] + dataframe['3SsnPorch'] + dataframe['WoodDeckSF']
    
    
sunny_cols = HousePrices[cols]

exterior_score(sunny_cols)
overall_qual_score(sunny_cols)
vnr_binary(sunny_cols)
roofmat_binary(sunny_cols)
salecond_binary(sunny_cols)
total_porchdeckSF(sunny_cols)

#Dummifying Exeterior 
x = pd.get_dummies(sunny_cols['Exterior1st']).drop(['VinylSd'], axis='columns')
sunny_cols1 = sunny_cols.merge(x, left_index=True, right_index=True)

#Dummifying HouseStyle
x1 = pd.get_dummies(sunny_cols1['HouseStyle']).drop(['1Story'], axis='columns')
sunny_cols2 = sunny_cols1.merge(x1, left_index=True,right_index=True)

sunny_clean = sunny_cols2.drop(['ExterQual','ExterCond','OverallQual','OverallCond','OpenPorchSF',
                                'ScreenPorch','3SsnPorch','WoodDeckSF','Exterior1st','HouseStyle'],
               axis='columns')

sunny_clean.rename(columns = {'AsbShng':'Ext_AsbShng',
                              'AsphShn':'Ext_AsphShn',
                              'BrkComm':'Ext_BrkComm',
                              'BrkFace':'Ext_BrkFace',
                              'CBlock':'Ext_CBlock',
                              'CemntBd':'Ext_CemntBd',
                              'HdBoard':'Ext_HdBoard',
                              'ImStucc':'Ext_ImStucc',
                              'MetalSd':'Ext_MetalSd',
                              'Plywood':'Ext_Plywood',
                              'Stone':'Ext_Stone',
                              'Stucco':'Ext_Stucco',
                              'Wd Sdng':'Ext_WdSdng',
                              'WdShing':'Ext_WdShing',
                              '1.5Fin':'House_1.5Fin',
                              '1.5Unf':'House_1.5Unf',
                              '2.5Fin':'House_2.5Fin',
                              '2.5Unf':'House_2.5Unf',
                              '2Story':'House_2Story',
                              'SFoyer':'House_SFoyer',
                              'SLvl':'House_SLvl'}, inplace=True)

############# Zafa
# Subset data to focus on  21 features:
my_features = ['Id','Condition1', 'Condition2', 'MSZoning', 'Street', 'Alley', 'Neighborhood', 'FireplaceQu', 'Fireplaces', 'PavedDrive', 'LotFrontage', 'LotArea', 'LowQualFinSF', 'LandContour', 'LotShape', 'LotConfig', 'LandSlope', 'Foundation']
HousePrices = HousePrices[my_features]

# Dummify Condition1, Drop Norm column which is most common
# Rename Columns with Cond prefix to indicate original feature
# Drop Condition1 and Condition2, Shape should be 1460 x 24
HousePrices = pd.concat([HousePrices, pd.get_dummies(HousePrices.Condition1).drop('Norm',1)], 1)
name_dict = {'Artery': 'Cond_Artery', 'Feedr': 'Cond_Feedr','PosA': 'Cond_PosA', 'PosN': 'Cond_PosN','RRAe':'Cond_RRAe','RRAn':'Cond_RRAn','RRNe':'Cond_RRNe', 'RRNn':'Cond_RRNn' }
HousePrices = HousePrices.rename(columns = name_dict).drop(columns=['Condition1','Condition2'])

# Convert Street to Binary where 1 is Paved and 0 is Gravel
HousePrices.Street = HousePrices.Street.str.replace('Pave','1').replace('Grvl','0')
HousePrices.Street = pd.to_numeric(HousePrices.Street)

# Create Binary column where 1 = has alley, 0 = does not have alley
HousePrices.Alley = HousePrices.Alley.str.replace('Pave','1').replace('Grvl','1')
HousePrices.Alley = pd.to_numeric(HousePrices.Alley).fillna(0)

# Dummify Neighborhoods, drop NAmes because it's most common
# Add Hood_ prefix to columns
# Drop Original Neighborhood column
neighborhood_df = pd.get_dummies(HousePrices.Neighborhood).add_prefix('Hood_')
HousePrices = pd.concat([HousePrices,neighborhood_df], 1)
HousePrices = HousePrices.drop(columns=['Neighborhood', 'Hood_NAmes'])

# Change FireplaceQu to score where 0 = no fireplace, Po=1 up to Ex=5
HousePrices.FireplaceQu = HousePrices.replace({'FireplaceQu': {'Ex': 5, 'Gd': 4, 'TA':3, 'Fa':2, 'Po':1, np.nan:0}}, value=None).FireplaceQu

# Dummify PavedDrive (Y,N or P), Add Paved_ prefix and Drop Paved_Y and PavedDrive column
# shape should be 1460x48
paved_df = pd.get_dummies(HousePrices.PavedDrive).add_prefix('Paved_').drop('Paved_Y',1)
HousePrices = pd.concat([HousePrices,paved_df], 1)
HousePrices = HousePrices.drop('PavedDrive',1)

# Drop Lot Frontage Column, 1460x47
HousePrices = HousePrices.drop('LotFrontage',1)

# Drop LowQualFinSF Column to get 1460x46
HousePrices = HousePrices.drop('LowQualFinSF',1)

# Dummify LotConfig column, Shape: 1460x49
lotconfig_df = pd.get_dummies(HousePrices.LotConfig).add_prefix('LotConfig_').drop('LotConfig_Inside',1)
HousePrices = pd.concat([HousePrices,lotconfig_df], 1).drop('LotConfig',1)

# Convert to ordinal scale Reg=0, IR1=1, IR2=2, IR3 =3
HousePrices.LotShape = HousePrices.replace({'LotShape': {'IR3':3, 'IR2':2, 'IR1':1, 'Reg':0}}, value=None).LotShape

# Merge C,A,I into Other category, Merge RL and RP together as RL
HousePrices.replace({'MSZoning': {'C (all)':'Other', 'RP':'RL', 'A':'Other', 'I':'Other'}}, value=None).MSZoning

# Dummify MSZoning after merging groups, shape should be 52
zoning_df = pd.get_dummies(HousePrices.MSZoning).add_prefix('Zone_').drop('Zone_RL',1)
HousePrices = pd.concat([HousePrices,zoning_df], 1).drop('MSZoning',1)

# Dummify Foundation , shape should be 56
foundation_df = pd.get_dummies(HousePrices.Foundation).add_prefix('Found_').drop('Found_PConc',1)
HousePrices = pd.concat([HousePrices,foundation_df], 1).drop('Foundation',1)

# Convert landslope to ordinal 0=Gtl, 1=Mod, 2 = Sev
HousePrices.LandSlope = HousePrices.replace({'LandSlope': {'Sev':2, 'Mod':1, 'Gtl':0}}, value=None).LandSlope

# Dummify LandContour and drop columns, Shape should be 1460x58
contour_df = pd.get_dummies(HousePrices.LandContour).add_prefix('Contour_').drop('Contour_Lvl',1)
HousePrices = pd.concat([HousePrices,contour_df], 1).drop('LandContour',1)

##############################
#Merging individual sections
full_df = cleaned_columns_ME.merge(living_rec_all, on = "Id")
full_df = full_df.merge(HousePrices, on = "Id")
full_df = full_df.merge(sunny_clean, on = "Id" )


hp_clean = full_df.to_csv('data/cleaned_houseprice_test.csv',index = False)