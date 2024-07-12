import polars as pl
import numpy as np


def agg_expr(agg, group_name):
    expr_max = [pl.max(col).alias(f"max_{group_name}_{col}") for col, methods in agg.items() for method in methods if method.lower() == 'max']
    expr_mean = [pl.mean(col).alias(f"mean_{group_name}_{col}") for col, methods in agg.items() for method in methods if method.lower() == 'mean']
    expr_min = [pl.min(col).alias(f"min_{group_name}_{col}") for col, methods in agg.items() for method in methods if method.lower() == 'min']
    expr_std = [pl.std(col).alias(f"std_{group_name}_{col}") for col, methods in agg.items() for method in methods if method.lower() == 'std']
    expr_count = [pl.count(col).alias(f"count_{group_name}_{col}") for col, methods in agg.items() for method in methods if method.lower() == 'count']
    expr_sum = [pl.sum(col).alias(f"sum_{group_name}_{col}") for col, methods in agg.items() for method in methods if method.lower() == 'sum']
    return expr_max + expr_mean + expr_std + expr_min + expr_count + expr_sum


def make_vid_pair_sim (df, vid_info,  group_col, use_cols, agg) :
    df_left = df.select(group_col + use_cols)#.rename({'vid' : 'left_vid', 'play_time' : 'left_play_time'})
    df_right = df.select(group_col + use_cols)#.rename({'vid' : 'right_vid', 'play_time' : 'right_play_time'})
    
    # 做笛卡尔积
    df_left = df_left.join(df_right, on=group_col, how='left')
    #df_left = df_left.filter(
    #    pl.col('vid') != pl.col('vid_right')
    #)
    df_features = df_left.group_by(['vid', 'vid_right'], maintain_order=True).agg(agg_expr(agg, 'vid_pair_sim'))
    return df_features

#召回构建训练样本
def create_data_sample (df_history_short_click_playtime, 
                        df_label,
                        history_seq_num,
                        df_candidate_did, 
                        df_candidate_vid, 
                        df_pair_vid_sim,
                        top_pair_sim_N=50, 
                        hots_N=10,
                        stage='train') :
    # 第一路召回：item2item 以同时访问两个视频的用户数来衡量相似度
    df_sample = df_history_short_click_playtime.filter(
        pl.col('did').is_in(df_candidate_did['did'].unique().to_list())
    ).group_by('did').tail(history_seq_num) # 获取candidate用户的历史history_seq_num条数据
    df_sample = df_sample.join(df_pair_vid_sim.group_by('vid').head(top_pair_sim_N), on='vid', how='left') #
    # 这是为了构造更多的样本作为负样本 因此采用vid_right替换vid  
    df_sample = df_sample.drop('vid')
    df_sample = df_sample.rename({'vid_right' : 'vid'})
    df_sample = df_sample.select(['did', 'vid', 'count_vid_pair_sim_did', 'sum_vid_pair_sim_play_time_right'])

    # 第二路召回： 召回热门物品
    # 每个did补label日最近一天的热门vid
    hots_N_vids = df_history_short_click_playtime.filter(
        pl.col('day') == df_history_short_click_playtime['day'].max()
    )['vid'].value_counts().sort('count', descending=True)['vid'].to_list()[0:hots_N]
    df_added_sample = df_candidate_did.with_columns(
        pl.lit(hots_N_vids).alias('hots_N_vids')
    )
    # 补热门
    # 本质上也是在增加样本 增加热门物品的样本对
    df_added_sample = df_added_sample.explode('hots_N_vids').rename({'hots_N_vids' : 'vid'})
    
    # 第三路召回：item2item 以同时访问两个视频的时长来衡量相似度
    # df_pair_vid_sim = df_pair_vid_sim.sort(['vid', 'count_vid_pair_sim_did'], descending=True)
    # df_added_sample3 = df_history_short_click_playtime.filter(
    #     pl.col('did').is_in(df_candidate_did['did'].unique().to_list())
    # ).group_by('did').tail(history_seq_num) # 获取candidate用户的历史history_seq_num条数据
    # df_added_sample3 = df_added_sample3.join(df_pair_vid_sim.group_by('vid').head(top_pair_sim_N), on='vid', how='left') #
    # df_added_sample3 = df_added_sample3.drop('vid')
    # df_added_sample3 = df_added_sample3.rename({'vid_right' : 'vid'})
    # df_added_sample3 = df_added_sample3.select(['did', 'vid', 'count_vid_pair_sim_did', 'sum_vid_pair_sim_play_time_right'])
    
    print("Recall Size", df_sample.shape, df_added_sample.shape)
    df_sample = pl.concat([df_sample, df_added_sample], how="diagonal").unique(('did', 'vid'), keep='first', maintain_order=True)
    print("Recall Size after unique", df_sample.shape)
    
    df_sample = df_sample.fill_null(0)
    
    #只保留候选vid的样本
    df_sample = df_sample.filter (
        pl.col('vid').is_in(df_candidate_vid['vid'].to_list())
    )
    #测试集不必构造label
    if stage != 'test' :
        df_label = df_label.select(['did', 'vid']).unique()
        df_label = df_label.with_columns(
            pl.lit(1).alias('label')
        )
        df_sample = df_sample.join(df_label, on=['did', 'vid'], how='left')

        #构建正负样本
        df_sample = df_sample.with_columns(
            pl.col('label').fill_null(0)
        )    
    return df_sample



def make_features (df_sample, df_history_short_click_playtime, df_history_short_show, vid_info):
    agg = {
        'vid' : ['count'], 
        'play_time' : ['mean', 'std', 'max', 'min'],
    }
    df_did_click_features = df_history_short_click_playtime.group_by('did').agg(agg_expr(agg, 'did_history_click'))
    agg = {
        'did' : ['count'],
        'play_time' : ['mean', 'std', 'max', 'min'],
    }
    df_vid_click_features = df_history_short_click_playtime.group_by('vid').agg(agg_expr(agg, 'vid_history_click'))
    
    agg = {
        'did' : ['count'],
        'play_time' : ['mean', 'std', 'max', 'min'],
    }
    df_cid_click_features = df_history_short_click_playtime.group_by('cid').agg(agg_expr(agg, 'cid_history_click'))    
    
    agg = {
        'play_time' : ['count'],
    }
    df_did_vid_click_features = df_history_short_click_playtime.group_by(['did', 'vid']).agg(agg_expr(agg, 'did_vid_history_click'))
    
    agg = {
        'play_time' : ['count', 'mean', 'std', 'max', 'min'],
    }
    df_did_cid_click_features = df_history_short_click_playtime.group_by(['did', 'cid']).agg(agg_expr(agg, 'did_cid_history_click'))
    
    agg = {
        'vid' : ['count'], 
    }
    df_did_show_features = df_history_short_show.group_by('did').agg(agg_expr(agg, 'did_history_show'))
    
    agg = {
        'did' : ['count'], 
    }
    df_vid_show_features = df_history_short_show.group_by('vid').agg(agg_expr(agg, 'vid_history_show'))

    agg = {
        'did' : ['count'], 
    }
    df_cid_show_features = df_history_short_show.group_by('cid').agg(agg_expr(agg, 'cid_history_show'))
    
    agg = {
        'show_time' : ['count'],
    }
    df_did_vid_show_features = df_history_short_show.group_by(['did', 'vid']).agg(agg_expr(agg, 'did_vid_history_show'))
    
    agg = {
        'show_time' : ['count'],
    }
    df_did_cid_show_features = df_history_short_show.group_by(['did', 'cid']).agg(agg_expr(agg, 'did_cid_history_show'))    
    
    df_sample = df_sample.join(vid_info.select(['vid', 'cid', 'is_intact', 'classify_id', 'serialno', 'duration', 'online_time']), 
                               on='vid', 
                               how='left')
    df_sample = df_sample.with_columns(
        ((df_history_short_show['show_time'].max() - pl.col('online_time')).alias('online_time_gap') / (1_000_000 * 60 * 60 * 24)).cast(pl.Int64)
    )

    df_sample = df_sample.join(df_did_click_features, on='did', how='left')
    df_sample = df_sample.join(df_vid_click_features, on='vid', how='left')
    df_sample = df_sample.join(df_cid_click_features, on='cid', how='left')
    df_sample = df_sample.join(df_did_vid_click_features, on=['did', 'vid'], how='left')
    df_sample = df_sample.join(df_did_cid_click_features, on=['did', 'cid'], how='left')
    df_sample = df_sample.join(df_did_show_features, on='did', how='left')
    df_sample = df_sample.join(df_vid_show_features, on='vid', how='left')
    df_sample = df_sample.join(df_cid_show_features, on='cid', how='left')
    df_sample = df_sample.join(df_did_vid_show_features, on=['did', 'vid'], how='left')
    df_sample = df_sample.join(df_did_cid_show_features, on=['did', 'cid'], how='left')

    df_sample = df_sample.with_columns(
        (pl.col('count_vid_history_click_did') / (pl.col('count_vid_history_show_did') + 0.01)).alias('vid_ctr'),
        (pl.col('count_cid_history_click_did') / (pl.col('count_cid_history_show_did') + 0.01)).alias('cid_ctr'),
    )
    return df_sample

def make_pipline(df_history_short_click_playtime, df_history_short_show, df_label, df_candidate_did, df_candidate_vid, vid_info, stage) :
    # input: 历史的短视频点击数据， 历史的短视频曝光数据， label, 候选user, 候选vid， vid info, stage
    # output: sample_data
    agg = {
        'did' : ['count'], 
        'play_time_right' : ['sum'],
    }
    df_pair_click_sim = make_vid_pair_sim(df_history_short_click_playtime, vid_info, 
                                           group_col=['did'], 
                                           use_cols=['vid', 'play_time'],
                                           agg=agg) 
    df_pair_click_sim = df_pair_click_sim.sort(['vid', 'count_vid_pair_sim_did'], descending=True) # 两两vid之间的相似度 通过同时点击两个vid的did的count来表示

    df_sample = create_data_sample(df_history_short_click_playtime,
                                          df_label,
                                          history_seq_num=10,
                                          df_candidate_did=df_candidate_did, 
                                          df_candidate_vid=df_candidate_vid, 
                                          df_pair_vid_sim=df_pair_click_sim,
                                          top_pair_sim_N=50, 
                                          hots_N=20,
                                          stage=stage)

    df_sample = make_features (df_sample, df_history_short_click_playtime, 
                                     df_history_short_show, vid_info)
    return df_sample
