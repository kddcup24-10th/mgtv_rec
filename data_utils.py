import polars as pl
import numpy as np
import pandas as pd
import ast


def get_did_vid_label(df_label_click, df_label_show, data_path):
    df_label_click = pd.DataFrame(df_label_click, columns=df_label_click.columns)
    df_label_show = pd.DataFrame(df_label_show, columns= df_label_show.columns)
    df_label_show_ct=df_label_show.groupby(['did', 'vid'])['session_id'].count().reset_index().rename(columns={'session_id': 'show_counts'})
    df_label_click_ct=df_label_click.groupby(['did', 'vid'])['session_id'].count().reset_index().rename(columns={'session_id': 'click_counts'})
    df_label_click_viewtime=df_label_click.groupby(['did', 'vid'])['play_time'].sum().reset_index().rename(columns={'play_time': 'sum_play_time'})
    vid_info = pd.read_csv(f'{data_path}/vid_info.csv')
    df_label_click_viewtime_vid=df_label_click_viewtime.merge(vid_info[['vid','duration']],on=['vid'],how='left')
    df_label_click_viewtime_vid['play_time_rate'] = 1.0*df_label_click_viewtime_vid['sum_play_time']/df_label_click_viewtime_vid['duration']
    df_label_click_viewtime_vid['play_time_rate'] = df_label_click_viewtime_vid['play_time_rate'].apply(lambda x: 1 if x > 1 else x)
    df_label_click_final=df_label_click_ct.merge(df_label_click_viewtime_vid,on=['did','vid'],how='left')
    df_show_click_final=df_label_show_ct[['did','vid','show_counts']].merge(df_label_click_final[['did','vid','click_counts','play_time_rate']],on=['did','vid'],how='left').fillna({'play_time_rate': 0,'click_counts':0})
    df_show_click_final['click_rate']=1.0*df_show_click_final['click_counts']/df_show_click_final['show_counts']
    df_show_click_final['click_rate'] = df_show_click_final['click_rate'].apply(lambda x: 1 if x > 1 else x)
    df_show_click_final['label']=df_show_click_final['click_rate']*0.5+df_show_click_final['play_time_rate']*0.5
    df_show_click_final_total=df_show_click_final[['did','vid','label']]
    return df_show_click_final_total

def agg_expr(agg, group_name):
    expr_max = [pl.max(col).alias(f"max_{group_name}_{col}") for col, methods in agg.items() for method in methods if method.lower() == 'max']
    expr_mean = [pl.mean(col).alias(f"mean_{group_name}_{col}") for col, methods in agg.items() for method in methods if method.lower() == 'mean']
    expr_min = [pl.min(col).alias(f"min_{group_name}_{col}") for col, methods in agg.items() for method in methods if method.lower() == 'min']
    expr_std = [pl.std(col).alias(f"std_{group_name}_{col}") for col, methods in agg.items() for method in methods if method.lower() == 'std']
    expr_count = [pl.count(col).alias(f"count_{group_name}_{col}") for col, methods in agg.items() for method in methods if method.lower() == 'count']
    expr_sum = [pl.sum(col).alias(f"sum_{group_name}_{col}") for col, methods in agg.items() for method in methods if method.lower() == 'sum']
    return expr_max + expr_mean + expr_std + expr_min + expr_count + expr_sum


def make_vid_pair_sim (df, vid_info,  group_col, use_cols, agg) :
    vid_info = vid_info.select(['vid', 'stars', 'theme', 'kind', 'keyWord'])
    df_left = df.select(group_col + use_cols)#.rename({'vid' : 'left_vid', 'play_time' : 'left_play_time'})
    df_right = df.select(group_col + use_cols)#.rename({'vid' : 'right_vid', 'play_time' : 'right_play_time'})
    # 做笛卡尔积
    df_left = df_left.join(df_right, on=group_col, how='left')
    #df_left = df_left.filter(
    #    pl.col('vid') != pl.col('vid_right')
    #)
    
    df_features = df_left.group_by(['vid', 'vid_right'], maintain_order=True).agg(agg_expr(agg, 'vid_pair_sim'))
    # 连接上vid_info 从而计算两个视频的相似度
    df_features = df_features.join(vid_info, on=['vid'], how='left').join(vid_info, left_on=['vid_right'], right_on=['vid'], how='left')
    # print("df_features.shape:", df_features.shape)
    
    # dp = pd.DataFrame(df_features, columns=df_features.columns)
    # print("dp_shape:", dp.shape)
    # 增加明星信息
    # dp['stars_intersection_size'] = dp.apply(lambda row: len(set(ast.literal_eval(row['stars'])) & set(ast.literal_eval(row['stars_right']))) if pd.notna(row['stars']) and pd.notna(row['stars_right']) else 0, axis=1) 
    # # 增加题材信息
    # dp['theme_intersection_size'] = dp.apply(lambda row: len(set(ast.literal_eval(row['theme'])) & set(ast.literal_eval(row['theme_right']))) if pd.notna(row['theme']) and pd.notna(row['theme_right']) else 0, axis=1) 
    # # 增加种类信息
    # dp['kind_intersection_size'] = dp.apply(lambda row: len(set(ast.literal_eval(row['kind'])) & set(ast.literal_eval(row['kind_right']))) if pd.notna(row['kind']) and pd.notna(row['kind_right']) else 0, axis=1) 
    # # 增加关键词信息
    # dp['keywords_intersection_size'] = dp.apply(lambda row: len(set(ast.literal_eval(row['keyWord'])) & set(ast.literal_eval(row['keyWord']))) if pd.notna(row['keyWord']) and pd.notna(row['keyWord_right']) else 0, axis=1) 
    # print("dp_shape:", dp.shape)
    # # 转polars.DF
    # df_features = pl.DataFrame(dp)
    # print("df_features.shape:", df_features.shape)
    # df_features = df_features.select(['vid', 'vid_right', 'count_vid_pair_sim_did', 'sum_vid_pair_sim_play_time_right', 'stars_intersection_size', 'theme_intersection_size', 'kind_intersection_size', 'keywords_intersection_size'])
    return df_features

def get_intersection_size(stars, stars_right):
    set_stars = set(ast.literal_eval(stars))
    set_stars_right = set(ast.literal_eval(stars_right))
    return len(set_stars & set_stars_right)

def itemcf_did(df_history_short_click_playtime, df_candidate_did, df_pair_vid_sim,history_seq_num, top_pair_sim_N):
    # 第一路召回：item2item 以同时访问两个视频的用户数来衡量相似度
    df_sample = df_history_short_click_playtime.filter(
        pl.col('did').is_in(df_candidate_did['did'].unique().to_list())
    ).group_by('did').tail(history_seq_num) # 获取candidate用户的历史history_seq_num条数据
    
    df_sample = df_sample.join(df_pair_vid_sim.group_by('vid').head(top_pair_sim_N), on='vid', how='left') #
    # 这是为了构造更多的样本作为负样本 因此采用vid_right替换vid  
    df_sample = df_sample.drop('vid')
    df_sample = df_sample.rename({'vid_right' : 'vid'})
    # df_sample = df_sample.select(['did', 'vid', 'count_vid_pair_sim_did', 'sum_vid_pair_sim_play_time_right', 'stars_intersection_size', 'theme_intersection_size', 'kind_intersection_size', 'keywords_intersection_size'])
    df_sample = df_sample.select(['did', 'vid', 'count_vid_pair_sim_did', 'sum_vid_pair_sim_play_time_right'])
    return df_sample

    
def hot_item_recall(df_history_short_click_playtime, df_candidate_did, hots_N):
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
    return df_added_sample

def all_hot_item_recall(df_history_short_click_playtime, df_candidate_did, hots_N):
    # 第二路召回： 召回热门物品
    # 每个did补label日最近一天的热门vid
    hots_N_vids = df_history_short_click_playtime['vid'].value_counts().sort('count', descending=True)['vid'].to_list()[0:hots_N]
    df_added_sample = df_candidate_did.with_columns(
        pl.lit(hots_N_vids).alias('hots_N_vids')
    )
    # 补热门
    # 本质上也是在增加样本 增加热门物品的样本对
    df_added_sample = df_added_sample.explode('hots_N_vids').rename({'hots_N_vids' : 'vid'})
    return df_added_sample

def everyday_top_k_items(df_history_short_click_playtime, df_candidate_did, hots_N):
    max_day = df_history_short_click_playtime['day'].max()
    hots_N_vids = []
    for day in range(max_day - 9, max_day + 1):
        hots_N_vids_day = df_history_short_click_playtime.filter(pl.col('day') == day)['vid'].value_counts().sort('count', descending=True)['vid'].to_list()[0:hots_N]
        hots_N_vids += hots_N_vids_day
    df_added_sample = df_candidate_did.with_columns(
        pl.lit(hots_N_vids).alias('hots_N_vids')
    )
    # 补热门
    # 本质上也是在增加样本 增加热门物品的样本对
    df_added_sample = df_added_sample.explode('hots_N_vids').rename({'hots_N_vids' : 'vid'})
    return df_added_sample
    

def get_recall_size(df_sample, df_label):
    sample_tuples = set(zip(df_sample['did'], df_sample['vid']))
    label_tuples = set(zip(df_label['did'], df_label['vid']))
    # 计算重合个数
    num_overlap = len(sample_tuples.intersection(label_tuples))

    # 计算不重合个数
    num_not_overlap_sample = len(sample_tuples - label_tuples)  # df_sample 中独有的 (did, vid)
    num_not_overlap_label = len(label_tuples - sample_tuples)  # df_label 中独有的 (did, vid)

    # 打印结果
    print(f"重合个数: {num_overlap}")
    print(f"df_sample 中不匹配的个数: {num_not_overlap_sample}")
    print(f"df_label 中不匹配的个数: {num_not_overlap_label}")
    print("包含正样:",num_overlap / df_sample.shape[0], "召回率:", num_overlap / df_label.shape[0])
    return 
    

#召回构建训练样本
def create_data_sample (df_history_short_click_playtime, 
                        df_label_click,
                        df_label_show,
                        history_seq_num,
                        df_candidate_did, 
                        df_candidate_vid, 
                        df_pair_vid_sim,
                        data_path,
                        top_pair_sim_N=50, 
                        hots_N=10,
                        stage='train') :
    print("df_pair_vid_sim.shape:", df_pair_vid_sim.shape)
    df_itemcf_sample = itemcf_did(df_history_short_click_playtime, df_candidate_did, df_pair_vid_sim,history_seq_num, top_pair_sim_N)
    
    # 热门物品召回 最近的热门物品
    df_hot_items_sample = hot_item_recall(df_history_short_click_playtime, df_candidate_did, hots_N)
    
    # TODO: 补热门 全部的热门
    # df_all_hot_items_sample = all_hot_item_recall(df_history_short_click_playtime, df_candidate_did, hots_N)
    
    # TODO: 其他指标进行itemcf
    # df_pair_vid_sim = df_pair_vid_sim.sort(['vid', 'sum_vid_pair_sim_play_time_right'], descending=True)
    # df_itemcf_sample2 = itemcf_did(df_history_short_click_playtime, df_candidate_did, df_pair_vid_sim,history_seq_num, top_pair_sim_N)
    # history_seq_num, top_pair_sim_N = 3, 3
    # df_pair_vid_sim = df_pair_vid_sim.sort(['vid', 'stars_intersection_size'], descending=True)
    # df_itemcf_sample3 = itemcf_did(df_history_short_click_playtime, df_candidate_did, df_pair_vid_sim,history_seq_num, top_pair_sim_N)
    # df_pair_vid_sim = df_pair_vid_sim.sort(['vid', 'theme_intersection_size'], descending=True)
    # df_itemcf_sample4 = itemcf_did(df_history_short_click_playtime, df_candidate_did, df_pair_vid_sim,history_seq_num, top_pair_sim_N)
    # df_pair_vid_sim = df_pair_vid_sim.sort(['vid', 'kind_intersection_size'], descending=True)
    # df_itemcf_sample5 = itemcf_did(df_history_short_click_playtime, df_candidate_did, df_pair_vid_sim,history_seq_num, top_pair_sim_N)
    # df_pair_vid_sim = df_pair_vid_sim.sort(['vid', 'keywords_intersection_size'], descending=True)
    # df_itemcf_sample6 = itemcf_did(df_history_short_click_playtime, df_candidate_did, df_pair_vid_sim,history_seq_num, top_pair_sim_N)
    
    
    print("Recall Size", df_itemcf_sample.shape, df_hot_items_sample.shape)
    df_sample = pl.concat([df_itemcf_sample, df_hot_items_sample], how="diagonal").unique(('did', 'vid'), keep='first', maintain_order=True)
    # df_sample = pl.concat([df_itemcf_sample, df_hot_items_sample, df_itemcf_sample2, df_itemcf_sample3, df_itemcf_sample4, df_itemcf_sample5, df_itemcf_sample6], how="diagonal").unique(('did', 'vid'), keep='first', maintain_order=True)
    print("Recall Size after unique", df_sample.shape)
    
    df_sample = df_sample.fill_null(0)
    
    #只保留候选vid的样本
    df_sample = df_sample.filter (
        pl.col('vid').is_in(df_candidate_vid['vid'].to_list())
    )
    print("candidate_recall_size:", df_sample.shape)
    #测试集不必构造label
    if stage != 'test' :
        # 采用click作为标签
        
        df_label_click = df_label_click.select(['did', 'vid']).unique()
        df_label_click = df_label_click.with_columns(
            pl.lit(1).alias('label')
        )
        
        get_recall_size(df_sample, df_label_click)
        
        df_sample = df_sample.join(df_label_click, on=['did', 'vid'], how='left')

        df_sample = df_sample.with_columns(
            pl.col('label').fill_null(0)
        )   
        # 采用click与show加权作为标签
        # df_label = get_did_vid_label(df_label_click, df_label_show, data_path)
        # df_label = pl.DataFrame(df_label)
        # df_sample = df_sample.join(df_label, on=['did', 'vid'], how='left')
        # df_sample = df_sample.with_columns(
        #     pl.col('label').fill_null(0)
        # )
         
    print("candidate_recall_size with lable:", df_sample.shape, "label size", df_label_click.shape)
    return df_sample



def make_features (df_sample, df_history_short_click_playtime, df_history_short_show, vid_info, user_info):
    agg = {
        'vid' : ['count'], 
        'play_time' : ['count', 'mean', 'std', 'max', 'min', 'sum'],
        'session_id' :['count'],
    }
    df_did_click_features = df_history_short_click_playtime.group_by('did').agg(agg_expr(agg, 'did_history_click'))
    agg = {
        'did' : ['count'],
        'play_time' : ['count', 'mean', 'std', 'max', 'min', 'sum'],
        'session_id' :['count'],
    }
    df_vid_click_features = df_history_short_click_playtime.group_by('vid').agg(agg_expr(agg, 'vid_history_click'))
    
    agg = {
        'did' : ['count'],
        'play_time' : ['count', 'mean', 'std', 'max', 'min', 'sum'],
        'session_id' :['count'],
    }
    df_cid_click_features = df_history_short_click_playtime.group_by('cid').agg(agg_expr(agg, 'cid_history_click'))    
    
    agg = {
        'play_time' : ['count', 'mean', 'std', 'max', 'min', 'sum'],
        'session_id' :['count'],
    }
    df_did_vid_click_features = df_history_short_click_playtime.group_by(['did', 'vid']).agg(agg_expr(agg, 'did_vid_history_click'))
    
    agg = {
        'play_time' : ['count', 'mean', 'std', 'max', 'min', 'sum'],
        'session_id' :['count'],
    }
    df_did_cid_click_features = df_history_short_click_playtime.group_by(['did', 'cid']).agg(agg_expr(agg, 'did_cid_history_click'))
    
    agg = {
        'vid' : ['count'], 
        'session_id' : ['count'],
        'show_time' : ['count']
    }
    df_did_show_features = df_history_short_show.group_by('did').agg(agg_expr(agg, 'did_history_show'))
    
    agg = {
        'did' : ['count'], 
        'session_id' :['count'],
        'show_time' : ['count']
    }
    df_vid_show_features = df_history_short_show.group_by('vid').agg(agg_expr(agg, 'vid_history_show'))

    agg = {
        'did' : ['count'], 
        'session_id' :['count'],
        'show_time' : ['count'],
    }
    df_cid_show_features = df_history_short_show.group_by('cid').agg(agg_expr(agg, 'cid_history_show'))
    
    agg = {
        'show_time' : ['count'],
        'session_id' :['count'],
    }
    df_did_vid_show_features = df_history_short_show.group_by(['did', 'vid']).agg(agg_expr(agg, 'did_vid_history_show'))
    
    agg = {
        'show_time' : ['count'],
        'session_id' :['count'],
    }
    df_did_cid_show_features = df_history_short_show.group_by(['did', 'cid']).agg(agg_expr(agg, 'did_cid_history_show'))    
    
    # df_sample = df_sample.join(vid_info.select(['vid', 'cid', 'is_intact', 'classify_id', 'serialno', 'duration', 'online_time']), 
    #                            on='vid', 
    #                            how='left')
    
    df_sample = df_sample.join(vid_info.select(['vid', 'cid', 'is_intact', 'classify_id', 'serialno', 'duration', 'online_time', 'stars', 'theme', 'kind', 'keyWord', 'series_id']), 
                               on='vid', 
                               how='left')
    
    # df_sample = df_sample.join(vid_info, on='vid', how='left')
    df_sample = df_sample.with_columns(
        ((df_history_short_show['show_time'].max() - pl.col('online_time')).alias('online_time_gap') / (1_000_000 * 60 * 60 * 24)).cast(pl.Int64)
    )
    
    #######################################
    # 增加用户特征表示用户喜欢的明星 主题等是否被当前vid命中
    df_sample = df_sample.join(user_info, on='did', how='left')
    df_sample = pd.DataFrame(df_sample, columns=df_sample.columns)
    df_sample["sim_star_counts"] = df_sample.apply(lambda row: len(set(eval(row["stars"])) & set(map(int, row["stars_right"].split(',')))) if (pd.notna(row["stars"]) and pd.notna(row["stars_right"])) else 0, axis=1)
    df_sample["sim_theme_counts"] = df_sample.apply(lambda row: len(set(eval(row["theme"])) & set(map(int, row["theme_right"].split(',')))) if (pd.notna(row["theme"]) and pd.notna(row["theme_right"])) else 0, axis=1)
    df_sample["sim_kind_counts"] = df_sample.apply(lambda row: len(set(eval(row["kind"])) & set(map(int, row["kind_right"].split(',')))) if (pd.notna(row["kind"]) and pd.notna(row["kind_right"])) else 0, axis=1)
    df_sample["sim_keyWord_counts"] = df_sample.apply(lambda row: len(set(eval(row["keyWord"])) & set(map(int, row["keyWord_right"].split(',')))) if (pd.notna(row["keyWord"]) and pd.notna(row["keyWord_right"])) else 0, axis=1)
    df_sample["sim_classify_id_counts"] = df_sample.apply(lambda row: len(set([row["classify_id"]]) & set(map(int, row["classify_id_right"].split(',')))) if (pd.notna(row["classify_id"]) and pd.notna(row["classify_id_right"])) else 0, axis=1)
    df_sample["sim_cid_counts"] = df_sample.apply(lambda row: len(set([row["cid"]]) & set(map(int, row["cid_right"].split(',')))) if (pd.notna(row["cid"]) and pd.notna(row["cid_right"])) else 0, axis=1)
    df_sample["sim_is_intact_counts"] = df_sample.apply(lambda row: len(set([row["is_intact"]]) & set(map(int, row["is_intact_right"].split(',')))) if (pd.notna(row["is_intact"]) and pd.notna(row["is_intact_right"])) else 0, axis=1)
    df_sample["sim_series_id_counts"] = df_sample.apply(lambda row: len(set([row["series_id"]]) & set(map(int, row["series_id_right"].split(',')))) if (pd.notna(row["series_id"]) and pd.notna(row["series_id_right"])) else 0, axis=1)
    
    df_sample = pl.DataFrame(df_sample)
    
    df_sample = df_sample.drop('stars').drop('stars_right').drop('theme').drop('theme_right').drop('kind').drop('kind_right').drop('keyWord').drop('keyWord_right').drop('classify_id_right').drop('cid_right').drop('is_intact_right').drop('series_id').drop('series_id_right')
    print("特征命中个数： 总采样： stars: theme: kind: keyWord: classify_id:",df_sample.    shape, df_sample.filter(pl.col('sim_star_counts') > 0).shape,
          df_sample.filter(pl.col('sim_theme_counts') > 0).shape,
          df_sample.filter(pl.col('sim_kind_counts') > 0).shape,
          df_sample.filter(pl.col('sim_keyWord_counts') > 0).shape,
          df_sample.filter(pl.col('sim_classify_id_counts') > 0).shape,
          df_sample.filter(pl.col('sim_cid_counts') > 0).shape,
          df_sample.filter(pl.col('sim_is_intact_counts') > 0).shape,
          df_sample.filter(pl.col('sim_series_id_counts') > 0).shape
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

def make_pipline(df_history_short_click_playtime, df_history_short_show, df_label_click, df_label_show, df_candidate_did, df_candidate_vid, vid_info, user_info, stage, data_path) :
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
                                          df_label_click,
                                          df_label_show,
                                          history_seq_num=10,
                                          df_candidate_did=df_candidate_did, 
                                          df_candidate_vid=df_candidate_vid, 
                                          df_pair_vid_sim=df_pair_click_sim,
                                          data_path=data_path,
                                          top_pair_sim_N=50, 
                                          hots_N=20,
                                          stage=stage)

    df_sample = make_features (df_sample, df_history_short_click_playtime, 
                                     df_history_short_show, vid_info, user_info)
    return df_sample
