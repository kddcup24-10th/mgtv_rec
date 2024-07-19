import polars as pl
import pandas as pd
from tqdm import tqdm

def parse_stars(stars_str):
    return eval(stars_str)

def get_user_hobby(dp, hobby_column, topn):
    dp_exploded = dp.explode(hobby_column)
    hobby_counts_per_user = dp_exploded.groupby(['did', hobby_column]).size().reset_index(name='counts')
    df_hobby = pl.DataFrame(hobby_counts_per_user)
    # 获取top5
    df_hobby = df_hobby.sort('counts', descending=True).group_by(['did']).head(topn)
    df_hobby = pd.DataFrame(df_hobby, columns=df_hobby.columns)
    user_hobby_feature = df_hobby.groupby('did', as_index=False).agg({hobby_column:lambda x:','.join(map(str, x))})
    return pl.DataFrame(user_hobby_feature)

def generate_user_info_from_long_behaviors(data_path):
    
    df_short_click_playtime_set, df_long_click_playtime_set = [], []
    df_short_show_set = []
    for day in tqdm(['day_1', 'day_2', 'day_3', 'day_4', 'day_5', 'day_6', 'day_7', 'day_8', 'day_9', 'day_10', 'day_11', 'day_12',]) :
    # for day in tqdm(['day_1', 'day_2', 'day_3', 'day_4', 'day_5', 'day_6', 'day_7', 'day_8', 'day_9', 'day_10', 'day_11',]) :
    # for day in tqdm(['day_1','day_2']) :
        flag = int(day.split("_")[1])
        # df_short_click_playtime = pl.read_csv(f'{data_path}/{day}/short_click_playtime.csv')
        # df_short_click_playtime = df_short_click_playtime.with_columns (
        #     pl.lit(flag).alias('day')
        # )
        # df_short_click_playtime_set.append(df_short_click_playtime)
        df_long_click_playtime = pl.read_csv(f'{data_path}/{day}/user_long_vv_behaviors.csv')
        df_long_click_playtime = df_long_click_playtime.with_columns (
            pl.lit(flag).alias('day')
        )
        df_long_click_playtime_set.append(df_long_click_playtime.select(["did", "vts","vid"]))
    
    
    df_long_click_playtime_set = pl.concat(df_long_click_playtime_set)
    print(df_long_click_playtime_set.shape)
    df_long_click_playtime_set = df_long_click_playtime_set.with_columns(
        pl.col('vts').alias('vts_minutes') / 60 / 1000
    )
    df_long_click_playtime_set = df_long_click_playtime_set.group_by(['did', 'vid']).agg([pl.col('vts_minutes').sum().alias('vts')])
    # 过滤vts < 1的视频 day1数据过滤后只剩28466 少了很多
    # df_long_click_playtime_set = df_long_click_playtime_set.filter(pl.col('vts') > 0.5)
    # 过滤0.5后全部数据298144, 3  不过滤数据225765727
    ################################
    # 过滤候选用户的视频
    for day in tqdm(['day_1', 'day_2', 'day_3', 'day_4', 'day_5', 'day_6', 'day_7', 'day_8', 'day_9', 'day_10', 'day_11', 'day_12',]) :
    # for day in tqdm(['day_1','day_2']) :
        flag = int(day.split("_")[1])
        df_short_click_playtime = pl.read_csv(f'{data_path}/{day}/short_click_playtime.csv')
        df_short_click_playtime = df_short_click_playtime.with_columns (
            pl.lit(flag).alias('day')
        )
        df_short_click_playtime_set.append(df_short_click_playtime)
    df_short_click_playtime = pl.concat(df_short_click_playtime_set)
    df_candidate_did = pl.read_csv(f'{data_path}/df_candidate_did_A.csv')
    # df_long_click_playtime_set = df_long_click_playtime_set.filter(pl.col('did').is_in(df_candidate_did['did'].unique().to_list())) 
    df_long_click_playtime_set = df_long_click_playtime_set.filter(pl.col('did').is_in(df_short_click_playtime['did'].unique().to_list() + df_candidate_did["did"].unique().to_list()))
    ##################
    print(df_long_click_playtime_set.shape) 
    # vts单位不明 直接取用户vts最大的20的视频来构建用户特征
    df_long_click_playtime_set = df_long_click_playtime_set.sort('vts', descending=True).group_by(['did']).head(50)
    
    # 获取vid info
    vid_info = pl.read_csv(f'{data_path}/vid_info.csv')
    df_user_video = df_long_click_playtime_set.join(vid_info, on=['vid'], how='left')
    df_user_video = df_user_video.select(['did', 'vid', 'cid', 'is_intact','series_id' , 'stars', 'theme', 'kind', 'keyWord', 'classify_id']) # classify_id共10个
    
    dp = pd.DataFrame(df_user_video, columns=df_user_video.columns)
    dp["stars"] = dp["stars"].apply(lambda x:eval(x) if pd.notna(x) else [])
    dp["theme"] = dp["theme"].apply(lambda x:eval(x) if pd.notna(x) else [])
    dp["kind"] = dp["kind"].apply(lambda x:eval(x) if pd.notna(x) else [])
    dp["keyWord"] = dp["keyWord"].apply(lambda x:eval(x) if pd.notna(x) else [])
    
    # 获取star特征
    # Day1数据：df_long_click_time: 30485401 df_short_click_time: 205844 unique_star:8021
    # dp_exploded = dp.explode('stars')
    # star_counts_per_user = dp_exploded.groupby(['did', 'stars']).size().reset_index(name='counts')
    # df_stars = pl.DataFrame(star_counts_per_user)
    # # 获取top5明星
    # df_stars = df_stars.sort('counts', descending=True).group_by(['did']).head(5)
    # df_stars = pd.DataFrame(df_stars, columns=df_stars.columns)
    # user_stars_feature = df_stars.groupby('did', as_index=False).agg({'stars':lambda x:','.join(map(str, x))})
    print("start generate feature")
    user_stars_feature = get_user_hobby(dp, "stars", 3) # unique: 总的8021 过滤一分钟后2388
    print('stars ok')
    user_theme_feature = get_user_hobby(dp, "theme", 3) # 过滤一分钟后：173
    print("theme ok")
    user_kind_feature = get_user_hobby(dp, "kind", 3) # 过滤一分钟后： 85
    print("kind ok")
    user_keyWord_feature = get_user_hobby(dp, "keyWord", 3)  # 过滤一分钟后： 2582
    print("keyWord ok")
    user_classify_id_feature = get_user_hobby(dp, "classify_id", 3) # 过滤一分钟后： 10
    print("id ok")
    user_cid_feature = get_user_hobby(dp, "cid", 3)
    print("cid ok")
    user_is_intact_feature = get_user_hobby(dp, "is_intact", 3)
    print("is_intact ok")
    user_series_id_feature = get_user_hobby(dp, "series_id", 3)
    print("series_id ok")
    
    
    user_features = user_stars_feature.join(user_theme_feature, on=['did'], how='left').join(user_kind_feature, on=['did'], how='left').join(user_keyWord_feature, on=['did'], how='left').join(user_classify_id_feature, on=['did'], how='left').join(user_cid_feature, on=['did'], how='left').join(user_is_intact_feature, on=['did'], how='left').join(user_series_id_feature, on=['did'], how='left')
    user_features = pd.DataFrame(user_features, columns=user_features.columns)
    print(user_features.shape) # 222785, 6   全部为770798, 6
    user_features.to_csv(data_path + "/user_info_not_filter_new.csv", index=False)

if __name__ == "__main__":
    data_path = "../A"
    generate_user_info_from_long_behaviors(data_path)
