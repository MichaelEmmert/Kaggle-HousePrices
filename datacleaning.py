import numpy as np
import pandas as pd
from houseprice_functions import ratings_to_ord

#read_csv
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
train_test = 0
for HousePrices in [train,test]:
  train_test += 1
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

  #cleaned_columns = HousePrices[['Id','CentralAir','HeatingQC','garage_score','Heating','Electrical',\
  #                               'GarageArea','finishedbsmt','GrLivArea','TotalBsmtSF']]
  HousePrices = HousePrices.merge(garage_type_dummy,how = "outer",left_index=True,right_index=True)
  HousePrices = HousePrices.merge(fence_dummy,how = "outer",left_index=True,right_index=True)

  # Total bath room - basement bathrooms are counted 75% of normal bathrooms
  # Then remove bathroom cols that are no longer needed
  HousePrices["BsmtFullBath"] = HousePrices["BsmtFullBath"].fillna(0)
  HousePrices["BsmtHalfBath"] = HousePrices["BsmtHalfBath"].fillna(0)
  HousePrices["HalfBath"] = HousePrices["HalfBath"].fillna(0)
  HousePrices["FullBath"]  = HousePrices["FullBath"].fillna(0)
  HousePrices["TotalBath"] = HousePrices["BsmtFullBath"]*0.75 + HousePrices["FullBath"] + \
  HousePrices["BsmtHalfBath"]*0.75*0.5 + HousePrices["HalfBath"]*0.5

  # Convert Pool and Pool_Ex (whether in exellent condition) to binary
  HousePrices = HousePrices.assign(Pool = np.select([HousePrices['PoolArea'] > 0], "1", "0"))
  HousePrices = HousePrices.assign(Pool_Ex = np.select([HousePrices['PoolQC'] == "Ex"], "1", "0")) 

  ##########
  #### TotalBsmtSF
  HousePrices['TotalBsmtSF'] = HousePrices['TotalBsmtSF'].fillna(0)
  HousePrices['GrLivArea'] = HousePrices['GrLivArea'].fillna(0)
  HousePrices['TotalSF'] = HousePrices['GrLivArea'] + HousePrices['TotalBsmtSF']
  ##########

  # Fills na with 0
  HousePrices=HousePrices.fillna(0)

  # Check null values after fillna - commented out for now
  # Living_Rec_Cat.isnull().sum()

  # Create a dictionary of ordinal categories 
  Bsmt_Desc_Dict = {"Ex":5, "Gd":4 ,"TA":3,"Av":3,"Fa":2, "Mn":2, "Po":1, "No":1, "GLQ":6,"ALQ":5,"BLQ":4,"Rec":3,"LwQ":2,"Unf":1}

  # Converts the following columns to ordinal categories
  Col_Convert = ["BsmtCond","BsmtFinType1","BsmtFinType2","BsmtExposure","BsmtQual","KitchenQual"]
  for col in Col_Convert:
      convert_index = HousePrices.loc[HousePrices[col]!=0].index
      HousePrices.loc[convert_index,col] =  HousePrices.loc[convert_index][[col]].apply(lambda x: Bsmt_Desc_Dict.get(x[0]),axis = 1)

  # Consolidate new basement related columns
  HousePrices["Basm_Quality"] = HousePrices.BsmtCond + HousePrices.BsmtExposure + HousePrices.BsmtQual
   
  HousePrices["BsmtFin_Quality"] = HousePrices.BsmtFinType1 + HousePrices.BsmtFinType2

  qual_ = {"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1,"Na": 0}
  HousePrices['ExterQual'] = HousePrices['ExterQual'].apply(lambda x: int(qual_[x]))
  HousePrices['ExterCond'] = HousePrices['ExterCond'].apply(lambda x: int(qual_[x]))
  HousePrices['Ext_Score'] = HousePrices['ExterQual'] + HousePrices['ExterCond']   

  HousePrices['OverallScore'] = HousePrices['OverallQual'] + HousePrices['OverallCond']
      
  HousePrices['MasVnrType'] = HousePrices['MasVnrType'].fillna('None')
  HousePrices['MasVnrType'] = HousePrices['MasVnrType'].apply(lambda x: 0 if x=='None' else 1)


  HousePrices['RoofMatl'] = HousePrices['RoofMatl'].fillna('Na')
  HousePrices['RoofMatl'] = HousePrices['RoofMatl'].apply(lambda x: 1 if x=='CompShg' else 0)
      
  HousePrices['SaleCondition'] = HousePrices['SaleCondition'].fillna('Na')
  HousePrices['SaleCondition'] = HousePrices['SaleCondition'].apply(lambda x: 1 if x=='Abnorml' else 0)
      
  HousePrices['Total_PorchDeckSF']  = HousePrices['OpenPorchSF'] + HousePrices['ScreenPorch'] + HousePrices['3SsnPorch'] + HousePrices['WoodDeckSF']
      
  #Dummifying Exeterior
  x = pd.get_dummies(HousePrices['Exterior1st']).drop(['VinylSd'], axis='columns')
  HousePrices = HousePrices.merge(x, left_index=True, right_index=True)

  #Dummifying HouseStyle
  x1 = pd.get_dummies(HousePrices['HouseStyle']).drop(['1Story'], axis='columns')
  HousePrices = HousePrices.merge(x1, left_index=True,right_index=True)

  HousePrices.rename(columns = {'AsbShng':'Ext_AsbShng','AsphShn':'Ext_AsphShn','BrkComm':'Ext_BrkComm',
                                'BrkFace':'Ext_BrkFace','CBlock':'Ext_CBlock','CemntBd':'Ext_CemntBd',
                                'HdBoard':'Ext_HdBoard','ImStucc':'Ext_ImStucc','MetalSd':'Ext_MetalSd',
                                'Plywood':'Ext_Plywood','Stone':'Ext_Stone','Stucco':'Ext_Stucco',
                                'Wd Sdng':'Ext_WdSdng','WdShing':'Ext_WdShing','1.5Fin':'House_1.5Fin',
                                '1.5Unf':'House_1.5Unf','2.5Fin':'House_2.5Fin','2.5Unf':'House_2.5Unf',
                                '2Story':'House_2Story','SFoyer':'House_SFoyer','SLvl':'House_SLvl'}, inplace=True)

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

  # Dummify LotConfig column, Shape: 1460x49
  lotconfig_df = pd.get_dummies(HousePrices.LotConfig).add_prefix('LotConfig_').drop('LotConfig_Inside',1)
  HousePrices = pd.concat([HousePrices,lotconfig_df], 1).drop('LotConfig',1)

  # Convert to ordinal scale Reg=0, IR1=1, IR2=2, IR3 =3
  HousePrices.LotShape = HousePrices.replace({'LotShape': {'IR3':3, 'IR2':2, 'IR1':1, 'Reg':0}}, value=None).LotShape

  # Merge C,A,I into Other category, Merge RL and RP together as RL
  HousePrices.MSZoning = HousePrices.replace({'MSZoning': {'C (all)':'Other', 'RP':'RL', 'A':'Other', 'I':'Other'}}, value=None).MSZoning

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

  HousePrices = HousePrices.drop(["MSSubClass","Utilities","BldgType","YearBuilt","RoofStyle",
    "Exterior2nd","MasVnrArea","BsmtQual","BsmtCond","BsmtExposure",
    "BsmtFinType1","BsmtFinSF1","BsmtFinType2","BsmtFinSF2","BsmtUnfSF",
    "1stFlrSF","2ndFlrSF","BsmtFullBath","BsmtHalfBath","FullBath",
    "HalfBath","KitchenAbvGr","TotRmsAbvGrd","Functional","GarageType",
    "GarageYrBlt","GarageFinish","GarageCars","GarageQual","GarageCond",
    "PoolArea","PoolQC","Fence","MiscFeature","MiscVal","SaleType",
    "TotalSF",'ExterQual','ExterCond','OverallQual','OverallCond','OpenPorchSF',
    'ScreenPorch','3SsnPorch','WoodDeckSF','Exterior1st','HouseStyle','LotFrontage',
    'LowQualFinSF','PavedDrive'],axis = 1)
  if train_test == 1:
    HousePrices = HousePrices.drop(["SalePrice","Ext_ImStucc", "Ext_Stone","House_2.5Fin"],axis = 1)
    HousePrices.to_csv('data/cleaned_houseprice.csv',index = False)
  elif train_test == 2:
    HousePrices = HousePrices.drop(["Zone_0"],axis = 1)
    HousePrices.to_csv('data/cleaned_houseprice_test.csv',index = False)