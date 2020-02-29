import numpy as np
import pandas as pd

def ratings_to_ord(df,col,inplace = False):
    '''
    This Function takes a dataframe and a column of that dataframe and returns and converts it to ordinal 
    df:
    col:
    inplace: 
    '''
    df[col] = df[col].fillna('Na')
    qual_ = {"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1,"Na": 0}
    if inplace == False:
        return df[col].apply(lambda x: list(qual_.values())[list(qual_.keys()).index(x)])
    elif inplace == True:
        df[col] = df[col].apply(lambda x: list(qual_.values())[list(qual_.keys()).index(x)])

def outlier_selecter(df,outlier_column , num_sd = 3, min_unique = 20, drop_zeros = True, for_test = False):
    '''
    creates a list of outliers to feed into the imputer
    '''
    outlier_dict = {}
    outlier_dict_test = {}
    for col in outlier_column:
        if col != 'Id':
            #Making sure the column has at least 20 unique values to prevent binary outlier imputation
            if (df[col].nunique() >= min_unique):
                #imputes mean without dropping zeros
                if drop_zeros == False:
                    mean = df.loc[:,col].mean()
                    sd   = df.loc[:,col].std()
                    outlier_bound_high = mean + sd*num_sd
                    outlier_bound_low  = mean - sd*num_sd
                #imputes mean with zeros dropped
                elif drop_zeros == True:
                    mean_no_z = df.loc[(df.loc[:,col] != 0)].loc[:,col].mean()
                    sd_no_z = df.loc[(df.loc[:,col] != 0)].loc[:,col].std()
                    outlier_bound_high = mean_no_z + sd_no_z*num_sd
                    outlier_bound_low  = mean_no_z - sd_no_z*num_sd
                # finds the indexes of the outlies 
                outliers_idx = df.index[df.loc[:,col].apply(lambda x: (x < outlier_bound_low) or (x > outlier_bound_high))].tolist()
                #appending the column name and the outliers in that column to a dictionary
                outlier_dict_test[col] = [outlier_bound_low, outlier_bound_high]
                if len(outliers_idx) != 0:
                    outlier_dict[col] = outliers_idx
    if for_test == True:
        return outlier_dict_test
    else:
        return outlier_dict

def outlier_imputation(df_train,df_test,index_values, col = "",method = "drop_row",decimals = 0, drop_zeros = True):
    '''
    df: dataframe
    index values: an integer or list of ints that indicate the rows that need to be imputed
    column: if mutatng using the values from a column input column name
    method : the method of imputation "drop", "mean", "median", "mode"
    decimals: num of decimals to include in the rounding, default 0
    '''
    idx_v = []
    before = []
    after = []
    #value_switch = pd.DataFrame([idx_v,before,after],columns = ['Id',f'{col} Before Imputation',f'{col} After Imputation'])
    if type(index_values) == int:
        index_values = [index_values]
    for idx in index_values:
        drop_ = []
        idx_v.append(idx)
        col_ = df_train.loc[:,col]
        col_drop = df_train.loc[:,col].loc[(col_ != 0)]
        before.append(df_test.loc[:,col].iloc[idx])
        #drop
        if method == "drop_row":
            if idx not in drop_:
                drop_.append(idx)
                df_test.drop(idx,inplace = True)
                after.append('The whole row is gone')
        #Mean
        elif method == "mean":
            if drop_zeros == False:
                df_test.loc[:,col].iloc[idx] = col_.mean()
            elif drop_zeros == True:
                df_test.loc[:,col].iloc[idx] = col_drop.mean()
        #Median
        elif method == "median":
            if drop_zeros == False:
                df_test.loc[:,col].iloc[idx] = col_.median()
            elif drop_zeros == True:
                df_test.loc[:,col].iloc[idx] = col_drop.median()
        #Mode
        elif method == "mode":
            if drop_zeros == False:
                df_test.loc[:,col].iloc[idx] = col_.mode()
            elif drop_zeros == True:
                df_test.loc[:,col].iloc[idx] = col_drop.mode()
        #Random
        elif method == "random":
            if drop_zeros == False:
                df_test.loc[:,col].iloc[idx] = col_.sample()
            elif drop_zeros == True:
                df_test.loc[:,col].iloc[idx] = np.random.choice(col_drop)
        after.append(round(df_test.loc[:,col].iloc[idx],decimals))
    #creating a dataframe to see the new value change for the imputation
    value_switch = pd.DataFrame(
        {
        'Id'                       : idx_v,
         f'{col} Before Imputation': before,
         f'{col} After Imputation' : after
        }
    )
    print(value_switch.to_string())
    print('='*60,'\n')



def k_neighbors(df_train,df_test,imputed_column,index_values,neihgbor_coulmn,k):
    df = df_train
    ic = imputed_column
    nc = neihgbor_coulmn
    for idx in index_values:
        # imputed columns neighbor value
        before = df.loc[:,ic].iloc[idx]
        icn_value = df.loc[:,nc].iloc[idx]
        # index of k closest neighbor column values 
        kn_index = np.abs(icn_value - df.loc[:,nc]).drop(idx).sort_values().head(k).index
        # the mean of the index values ic value
        mean_vic = df.loc[:,ic].iloc[kn_index].mean()
        # replace the ic index with the new mean value
        df_test.loc[:,ic].iloc[idx] = mean_vic
        after = df_test.loc[:,ic].iloc[idx]
        print(
            "-"*20,
            f"\nimputed on: {ic}",
            f"\nNeighbor calculation: {nc}",
            "\nID:",idx,
            "\nBefore:",before,
            "\nAfter:", after, 
            '\n',
            "-"*20
        )


