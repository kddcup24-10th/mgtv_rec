

记录
* Baseline:  本地验证集 0.119 A榜 0.1040
* Baseline + play time 召回   基本同baseline 合理因为召回的item太多基本没变化
* Baseline + seq=10 sim_N=10 hot_item=15 训练valid auc 0.78 本地验证集 0.1238 A榜: 0.1071
* Baseline + item2item(相同明星召回) 



召回分析：
* train_data:
label： 219368
seq_len = 10, sim_N  = 50, hot_item = 20
        Recall Size (7243467, 4) (3228200, 2)
        Recall Size after unique (7600293, 4)
        candidate_recall_size: (6972331, 4)
        重合个数: 65579
        df_sample 中不匹配的个数: 6906752
        df_label 中不匹配的个数: 153789
        包含正样: 0.009405606245601364 召回率: 0.29894515152620255
        candidate_recall_size with lable: (6972331, 5) label size (219368, 3)
seq_len = 10, sim_N  = 20, hot_item = 20 
        Recall Size (2976127, 4) (3228200, 2)
        Recall Size after unique (4951339, 4)
        candidate_recall_size: (4733346, 4)
        重合个数: 58897
        df_sample 中不匹配的个数: 4674449
        df_label 中不匹配的个数: 160471
        包含正样: 0.012442994870858797 召回率: 0.2684849203165457
        candidate_recall_size with lable: (4733346, 5) label size (219368, 3)
seq_len = 10, sim_N  = 10, hot_item = 20
        Recall Size (1502313, 4) (3228200, 2)
        Recall Size after unique (4081138, 4)
        candidate_recall_size: (3980339, 4)
        重合个数: 54695
        df_sample 中不匹配的个数: 3925644
        df_label 中不匹配的个数: 164673
        包含正样: 0.013741291884937439 召回率: 0.2493298931475876
        candidate_recall_size with lable: (3980339, 5) label size (219368, 3)
seq_len = 10 sim_N = 5 , hot_item = 20
        Recall Size (754563, 4) (3228200, 2)
        Recall Size after unique (3652135, 4)
        candidate_recall_size: (3599663, 4)
        重合个数: 54695
        df_sample 中不匹配的个数: 3925644
        df_label 中不匹配的个数: 164673
        包含正样: 0.013741291884937439 召回率: 0.2493298931475876
        candidate_recall_size with lable: (3599663, 5) label size (219368, 3)
seq_len = 10 sim_N = 10 hot_item = 15
        Recall Size after unique (3300224, 4)
        candidate_recall_size: (3199425, 4)
        重合个数: 51330
        df_sample 中不匹配的个数: 3148095
        df_label 中不匹配的个数: 168038
        包含正样: 0.016043507817811012 召回率: 0.23399037234236533
        candidate_recall_size with lable: (3199425, 5) label size (219368, 3)


* valid data:
seq_len = 10 sim_N = 10 hot_item = 15
        Recall Size (1594358, 4) (2518380, 2)
        Recall Size after unique (3451729, 4)
        candidate_recall_size: (3352567, 4)
        重合个数: 49590
        df_sample 中不匹配的个数: 3302977
        df_label 中不匹配的个数: 177026
        包含正样: 0.014791650696317181 召回率: 0.21882832633176827
        candidate_recall_size with lable: (3352567, 5) label size (226616, 3)  


    

