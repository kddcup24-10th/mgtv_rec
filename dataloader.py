import polars as pl
import pandas as pd
import gc
from tqdm import tqdm

from data_utils import make_pipline
from mem_utils import reduce_mem_usage

def load_data(data_path):
    df_short_click_playtime_set = []
    df_short_show_set = []
    for day in tqdm(['day_1', 'day_2', 'day_3', 'day_4', 'day_5', 'day_6', 'day_7', 'day_8', 'day_9', 'day_10', 'day_11', 'day_12',]) :
        flag = int(day.split("_")[1])
        df_short_click_playtime = pl.read_csv(f'{data_path}/{day}/short_click_playtime.csv')
        df_short_click_playtime = df_short_click_playtime.with_columns (
            pl.lit(flag).alias('day')
        )
        df_short_click_playtime_set.append(df_short_click_playtime)
    
        df_short_show = pl.read_csv(f'{data_path}/{day}/short_show.csv')  
        df_short_show = df_short_show.with_columns (
            pl.lit(flag).alias('day')
        )
        df_short_show_set.append(df_short_show)  
    
    df_short_click_playtime = pl.concat(df_short_click_playtime_set)
    df_short_click_playtime = df_short_click_playtime.sort(['did', 'click_time'])
    
    df_short_show = pl.concat(df_short_show_set)
    df_short_show = df_short_show.sort(['did', 'show_time'])
    
    df_short_show = df_short_show.with_columns(pl.col("show_time").str.strptime(pl.Datetime, '%Y-%m-%d %H:%M:%S'))
    df_short_click_playtime = df_short_click_playtime.with_columns(pl.col("click_time").str.strptime(pl.Datetime, '%Y-%m-%d %H:%M:%S'))
    
    df_candidate_did_A = pl.read_csv(f'{data_path}/df_candidate_did_A.csv')
    df_candidate_vid_A = pl.read_csv(f'{data_path}/df_candidate_vid_A.csv')

    vid_info = pl.read_csv(f'{data_path}/vid_info.csv')
    vid_info = vid_info.with_columns(pl.col("online_time").str.strptime(pl.Datetime, '%Y-%m-%d %H:%M:%S'))
    
    #TODO: 加了合集id信息 考虑加入更多
    df_short_show = df_short_show.join(vid_info.select(['vid', 'cid']), on='vid', how='left')
    df_short_click_playtime = df_short_click_playtime.join(vid_info.select(['vid', 'cid']), on='vid', how='left')
    
    # 去除没有明星信息的电影 去除前：3469878  去除后： 3314914
    df_tags = vid_info.select(['vid', 'stars']).to_pandas()
    df_tags = df_tags[df_tags['stars'].notnull()].reset_index(drop=True)
    
    # 转vid_to_tags为字典 且设为vid: [star1, star2]的格式
    vid_to_tags = df_tags[['vid', 'stars']].set_index('vid')['stars'].apply(lambda x : x.replace('[', '').replace(']', '').replace(' ', '').split(',')).apply(lambda x : [int(i) for i in x]).to_dict()
    
    
    
    
    
    
    
    train_day = 11
    stage = 'train'
    df_train_label = df_short_click_playtime.filter(pl.col('day') == train_day)
    df_train_history_short_click_playtime = df_short_click_playtime.filter(pl.col('day') < train_day)
    df_train_history_short_show = df_short_show.filter(pl.col('day') < train_day)
    
    df_candidate_did_train = df_train_label.select('did').unique()
    df_candidate_vid_train = df_train_label.select('vid').unique()
    
    
    df_train_sample = make_pipline(df_train_history_short_click_playtime, 
                               df_train_history_short_show, 
                               df_train_label, 
                               df_candidate_did_train, 
                               df_candidate_vid_train, vid_info, stage=stage) 
    
    df_train_sample = reduce_mem_usage(df_train_sample.to_pandas(), verbose=True)
    
    valid_day = 12
    stage = 'valid'
    df_valid_label = df_short_click_playtime.filter(pl.col('day') == valid_day)
    df_valid_history_short_click_playtime = df_short_click_playtime.filter(pl.col('day') < valid_day)
    df_valid_history_short_show = df_short_show.filter(pl.col('day') < valid_day)
    df_candidate_did_valid = df_valid_label.select('did').unique()
    df_candidate_vid_valid = df_valid_label.select('vid').unique()
    df_valid_sample = make_pipline(df_valid_history_short_click_playtime, 
                               df_valid_history_short_show, 
                               df_valid_label, 
                               df_candidate_did_valid, 
                               df_candidate_vid_valid, vid_info, stage=stage) 
    #df_valid_sample = reduce_memory_usage_pl(df_valid_sample, name='df_valid_sample')
    df_valid_sample = reduce_mem_usage(df_valid_sample.to_pandas(), verbose=True)
    
    
    stage = 'test'
    df_test_label = pl.DataFrame()
    df_test_sample = make_pipline(df_short_click_playtime, 
                               df_short_show, 
                               df_test_label, 
                               df_candidate_did_A, 
                               df_candidate_vid_A, 
                               vid_info, stage=stage) 
    #df_valid_sample = reduce_memory_usage_pl(df_valid_sample, name='df_valid_sample')
    df_test_sample = reduce_mem_usage(df_test_sample.to_pandas(), verbose=True)

    gc.collect()
    
    return df_train_sample, df_valid_sample, df_valid_label, df_test_sample, vid_to_tags
    