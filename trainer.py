import os
import time
import pandas as pd
import lightgbm as lgb

from metrics import cal_final_score



def train(df_train_sample, df_valid_sample, df_valid_label, df_test_sample, vid_to_tags):
    num = os.cpu_count()
    print ("cpu_counts: ", num)

    # TODO:parameters search
    lgb_params = {
        #'objective': 'lambdarank',
        #'metric': {'ndcg'},
        #'ndcg_at': [10],
        #'group_column': 'did',      
        'objective': 'binary', #定义的目标函数
        # 'objective': 'regression',  
        #'metric': {'auc', 'binary_logloss'},
        'metric': {'auc'},
        'boosting_type' : 'gbdt',

        'learning_rate': 0.05,
        'max_depth' : 12,
        'num_leaves' : 2 ** 6,

        'min_child_weight' : 10,
        'min_data_in_leaf' : 40,

        'feature_fraction' : 0.70,
        'subsample' : 0.75,
        'seed' : 114,

        'nthread' : -1,
        'bagging_freq' : 1,
        'verbose' : 5,
        #'scale_pos_weight':200
    }
    
    train_label = df_train_sample['label'].values
    valid_label = df_valid_sample['label'].values

    df_train_group = df_train_sample.groupby("did")["did"].count().to_numpy()
    df_valid_group = df_valid_sample.groupby("did")["did"].count().to_numpy()

    unuse_cols_importance = ['did', 'label', 'online_time', 'predict_prob']

    feature_name = df_train_sample.columns.values
    feature_name = [i for i in feature_name if i not in unuse_cols_importance]
    #feature_name = [col for col in feature_name if col not in unuse_cols_importance]
    print(f"Feature length = {len(feature_name)}")
    print ("Feature: ", feature_name)

    print("Model Trainning.")

    trn_data = lgb.Dataset(df_train_sample[feature_name], 
                       label=train_label,)
                       #group=df_train_group)#, categorical_feature=cat_cols)
    val_data = lgb.Dataset(df_valid_sample[feature_name], 
                       label=valid_label,)
                       #group=df_valid_group)#, categorical_feature=cat_cols)

    #print ("train_data : ", train_data.info())
    #print ("test_data : ", train_data.info())
    lgb_model = lgb.train(lgb_params,
                trn_data,
                1000,
                valid_sets=[val_data],
                #categorical_feature=cat_cols, 
                callbacks=[lgb.early_stopping(stopping_rounds=100), 
                           lgb.log_evaluation(period=100, show_stdv=True)])#, feval=self_gauc)        

    df_valid_sample['predict_prob'] = lgb_model.predict(df_valid_sample[feature_name], 
                                                    num_iteration=lgb_model.best_iteration)
    
    path = "baseline_local_model_v1/" + str(int(time.time()))
    print(time.time())#获当前时间的时间戳
    print(path)#获取本地时间
    os.makedirs(path)
    lgb_model.save_model(path + '/model.txt')
    
    
    #评估线下验证集得分
    df_valid_result = df_valid_sample[['did', 'vid', 'predict_prob']].sort_values(by=['did', 'predict_prob'], ascending=False).reset_index(drop=True)
    top_N = 10
    df_valid_result = df_valid_result.groupby('did').head(top_N).reset_index(drop=True)
    df_temp, final_score = cal_final_score (df_valid_result[['did', 'vid']], df_valid_label.select(['did', 'vid']).to_pandas(), vid_to_tags, weight=0.9)
    print ("验证集得分为: ", final_score)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = feature_name
    fold_importance_df["importance"] = lgb_model.feature_importance()
    fold_importance_df["importance_gain"] = lgb_model.feature_importance(importance_type='gain')
    fold_importance_df = fold_importance_df.sort_values(by='importance', ascending=False)
    fold_importance_df.to_csv("fold_importance_df.csv", index=None) 
    
    
    
    
    
    
    
    if 'predict_prob' in df_valid_sample.columns.values :
        del df_valid_sample['predict_prob']
    df_total_sample = pd.concat([df_train_sample, df_valid_sample]).reset_index(drop=True)
    train_label = df_total_sample['label'].values
    
    #df_train_group = df_total_sample.groupby("did")["did"].count().to_numpy()


    feature_name = lgb_model.feature_name()
    print(f"Feature length = {len(feature_name)}")
    print ("Feature: ", feature_name)

    print("Model Trainning.")

    trn_data = lgb.Dataset(df_total_sample[feature_name], 
                       label=train_label,)
                       #group=df_train_group)#, categorical_feature=cat_cols)

    #print ("train_data : ", train_data.info())
    #print ("test_data : ", train_data.info())
    lgb_model = lgb.train(lgb_params,
                      trn_data,
                      int(lgb_model.best_iteration * 1.2),)

    df_test_sample['predict_prob'] = lgb_model.predict(df_test_sample[feature_name], 
                                                    num_iteration=lgb_model.best_iteration)

    #评估线下验证集得分
    df_test_result = df_test_sample[['did', 'vid', 'predict_prob']].sort_values(by=['did', 'predict_prob'], ascending=False).reset_index(drop=True)
    top_N = 10
    df_test_result = df_test_result.groupby('did').head(top_N).reset_index(drop=True)
    #保存答案
    df_test_result[['did', 'vid']].to_csv('mgtv_rec_baseline_v1.csv', index=None)

    path = "baseline_online_model_v1/" + str(int(time.time()))
    print(time.time())#获当前时间的时间戳
    print(path)#获取本地时间
    os.makedirs(path)
    lgb_model.save_model(path + '/model.txt')


    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = feature_name
    fold_importance_df["importance"] = lgb_model.feature_importance()
    fold_importance_df["importance_gain"] = lgb_model.feature_importance(importance_type='gain')
    fold_importance_df = fold_importance_df.sort_values(by='importance', ascending=False)
    fold_importance_df.to_csv("fold_importance_df.csv", index=None)

