import numpy as np
import polars as pl

def drop_unuse_cols_by_pandas (df, unuse_cols):
    for col in unuse_cols :
        if col in df.columns.values :
            del df[col]
    return df    

def drop_unuse_cols_by_polars (df, unuse_cols):
    for col in unuse_cols :
        if col in df.columns :
            df = df.drop(col)
    return df  

def select_part_df_by_id(df, name, val_set) :
    return df[df[name].isin(val_set)].reset_index(drop=True)


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    cols_ = [col for col in list(df) if col not in ['cid', 'vid']]
    for col in cols_:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                       df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def reduce_memory_usage_pl(df, name):
    """ Reduce memory usage by polars dataframe {df} with name {name} by changing its data types.
        Original pandas version of this function: https://www.kaggle.com/code/arjanso/reducing-dataframe-memory-size-by-65 """
    print(f"Memory usage of dataframe {name} is {round(df.estimated_size('mb'), 2)} MB")
    Numeric_Int_types = [pl.Int8,pl.Int16,pl.Int32,pl.Int64]
    Numeric_Float_types = [pl.Float32,pl.Float64]    
    Cate_types = [pl.Categorical, pl.Utf8]
    #for col in df.columns:
    for col in [col for col in df.columns if col != 'did'] :
        col_type = df[col].dtype
        if col_type not in Numeric_Int_types + Numeric_Float_types + Cate_types :
            continue
        c_min = df[col].min()
        c_max = df[col].max()
        if col_type in Numeric_Int_types:
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df = df.with_columns(df[col].cast(pl.Int8))
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df = df.with_columns(df[col].cast(pl.Int16))
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df = df.with_columns(df[col].cast(pl.Int32))
            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                df = df.with_columns(df[col].cast(pl.Int64))
        elif col_type in Numeric_Float_types:
            try :
                #if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                #    df = df.with_columns(df[col].cast(pl.Float16))
                #elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                #    df = df.with_columns(df[col].cast(pl.Float32))
                #else:
                    #df[col] = df[col].astype(np.float64)                   
                #    df = df.with_columns(df[col].cast(pl.Float64))
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df = df.with_columns(df[col].cast(pl.Float32))
                else:
                    df = df.with_columns(df[col].cast(pl.Float32))
                    #pass
            except :
                print ("reduce memory exception: ", name, col, c_min, c_max)
                pass
        elif col_type == pl.Utf8 :
            df = df.with_columns(df[col].cast(pl.Categorical))
        else:
            pass
    print(f"Memory usage of dataframe {name} became {round(df.estimated_size('mb'), 2)} MB")
    return df