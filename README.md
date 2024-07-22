

记录
* Baseline:  本地验证集 0.119 A榜 0.1040
* Baseline + play time 召回   基本同baseline 合理因为召回的item太多基本没变化
* Baseline + item2item明星 题材 keywords等等召回 召回效果并不好
* Baseline + seq=10 sim_N=10 hot_item=15 训练valid auc 0.78 本地验证集 0.1238 A榜: 0.1071
* Baseline + seq=10 sim_n=10 hot_item=20  auc 0.78 本地验证集0.1227 A榜 0.1066
* Baseline + seq=10 sim_n=20 hot_item=20 auc 0.7823 本地验证集合0.1232 A榜 1064
* Baseline + seq=10 sim_n=10 hot_item=15 + user_info(过滤0.5后的userinfo,匹配率并不高 训练匹配率(总采样: stars: theme: kind: keyWord: classify_id: (3352567, 17) (4151, 17) (11963, 17) (9963, 17) (4989, 17) (24103, 17)))   auc 0.7841 本地验证集合0.12469 A榜： 0.1076
* Baseline + seq=10 sim_n=10 hot_item=15  + user_info_all(不过滤时长的userinfo,总采样： stars: theme: kind: keyWord: classify_id: (3199425, 17) (122571, 17) (485980, 17) (635646, 17) (246383, 17) (1097396, 17)   auc 0.804394  本地验证集0.1332  A榜：0.1097
* Baseline + seq=10 sim_n=10 hot_item=15 + user_info_all + 全量特征共67个特征 auc 0.800796本地验证集合 0.1311 A榜 0.1083
* Baseline + seq=10 sim_n=50 hot_item=20 + user_info_all_new + 全量特征 v2 
* Baseline + seq=10 sim_n=10  hot_item=15  + user_info_all_new + 全量特征 + regression v1 auc 0.797467   本地验证集0.13316  A榜 0.1026     本地验证集上升但是A榜没变化 难道是引入用户特征的时候采用了12天长视频数据的原因？

* TODO: lightgcn训练召回模型
* TODO: 更换label


长视频候选用户统计:
* 全部长视频记录中用户为短视频点击的条数： (33109888, 3)
* 用户为候选用户的条数 (2455631,3)
* 用户为a + b的条数  (34556453, 3)
* 用户为全部用户的条数 (428310531, 3) 聚合后(225765727, 3)



召回分析：
* train_data:
label： 219368

* seq_len = 10, sim_N  = 50, hot_item = 20
    * Recall Size (7243467, 4) (3228200, 2)
    * Recall Size after unique (7600293, 4)
    * candidate_recall_size: (6972331, 4)
    * 重合个数: 65579
    * df_sample 中不匹配的个数: 6906752
    * df_label 中不匹配的个数: 153789
    * **包含正样: 0.009405606245601364 召回率: 0.29894515152620255**
    * candidate_recall_size with lable: (6972331, 5) label size (219368, 3)

* seq_len = 10, sim_N  = 20, hot_item = 20 
    * Recall Size (2976127, 4) (3228200, 2)
    * Recall Size after unique (4951339, 4)
    * candidate_recall_size: (4733346, 4)
    * 重合个数: 58897
    * df_sample 中不匹配的个数: 4674449
    * df_label 中不匹配的个数: 160471
    * **包含正样: 0.012442994870858797 召回率: 0.2684849203165457**
    * candidate_recall_size with lable: (4733346, 5) label size (219368, 3)
* seq_len = 10, sim_N  = 10, hot_item = 20
    * Recall Size (1502313, 4) (3228200, 2)
    * Recall Size after unique (4081138, 4)
    * candidate_recall_size: (3980339, 4)
    * 重合个数: 54695
    * df_sample 中不匹配的个数: 3925644
    * df_label 中不匹配的个数: 164673
    * **包含正样: 0.013741291884937439 召回率: 0.2493298931475876**
    * candidate_recall_size with lable: (3980339, 5) label size (219368, 3)
* seq_len = 10 sim_N = 5 , hot_item = 20
    * Recall Size (754563, 4) (3228200, 2)
    * Recall Size after unique (3652135, 4)
    * candidate_recall_size: (3599663, 4)
    * 重合个数: 54695
    * df_sample 中不匹配的个数: 3925644
    * df_label 中不匹配的个数: 164673
    * **包含正样: 0.013741291884937439 召回率: 0.2493298931475876**
    * candidate_recall_size with lable: (3599663, 5) label size (219368, 3)
* seq_len = 10 sim_N = 10 hot_item = 15
    * Recall Size after unique (3300224, 4)
    * candidate_recall_size: (3199425, 4)
    * 重合个数: 51330
    * df_sample 中不匹配的个数: 3148095
    * df_label 中不匹配的个数: 168038
    * **包含正样: 0.016043507817811012 召回率: 0.23399037234236533**
    * candidate_recall_size with lable: (3199425, 5) label size (219368, 3)


* valid data:
seq_len = 10 sim_N = 10 hot_item = 15
    * Recall Size (1594358, 4) (2518380, 2)
    * Recall Size after unique (3451729, 4)
    * candidate_recall_size: (3352567, 4)
    * 重合个数: 49590
    * df_sample 中不匹配的个数: 3302977
    * df_label 中不匹配的个数: 177026
    * **包含正样: 0.014791650696317181 召回率: 0.21882832633176827**
    * candidate_recall_size with lable: (3352567, 5) label size (226616, 3)  


    

