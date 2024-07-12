
def RR(ranked_list, ground_list):
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id in ground_list:
            return 1 / (i + 1.0)
    return 0

def merge_answser (df_results, df_answer) :
    #df_results = df_results.sort_values(by=['did', 'rank']).reset_index(drop=True)
    df_results = df_results.groupby('did')['vid'].apply(list).reset_index()
    df_results.rename(columns = {'vid' : 'vid_list'}, inplace=True)
    #df_results = df_results.merge(df_answer[['did', 'vid']], on='did', how='left')
    #df_results['vid'] = df_results['vid'].apply(lambda x : [x])
    df_results['vid'] = df_results['did'].map(df_answer.groupby('did')['vid'].apply(list))
    return df_results

def get_vid_sim (vid1, vid2, vid_to_tags) :
    if vid1 in vid_to_tags and vid2 in vid_to_tags :
        return len(set(vid_to_tags[vid1]) & set(vid_to_tags[vid2])) / len(set(vid_to_tags[vid1]) | set(vid_to_tags[vid2])) 
    return 1

def cal_mrr_score (df_results, df_answer) :
    df_results = merge_answser (df_results, df_answer)
    #df_results['base_model_v1_rr_score'] = df_results[['vid', 'vid_list']].parallel_apply(lambda x : RR(x['vid'], x['vid_list']), axis=1)
    df_results['base_model_v1_rr_score'] = df_results[['vid_list', 'vid']].apply(lambda x : RR(x['vid_list'], x['vid']), axis=1)
    base_model_v1_rr_score = df_results['base_model_v1_rr_score'].mean()
    print ("base_model_v1_rr_score: ", base_model_v1_rr_score)    
    return base_model_v1_rr_score


def compute_diversity_sim(elements, vid_to_tags):
    """
    Compute pairwise absolute differences for a list of elements without repetition.
    """
    similarities = {}
    sim_sum = 0
    for i in range(len(elements)):
        for j in range(i + 1, len(elements)):
            #sim = abs(elements[i] - elements[j])
            sim = get_vid_sim (elements[i], elements[j], vid_to_tags)
            similarities[(i, j)] = 1 - sim
            sim_sum += 1 - sim
    sim_sum *= 1 / (len(elements) * (len(elements) - 1))
    #return similarities, sim_sum
    return similarities, sim_sum

def cal_final_score (df_results, df_answer, vid_to_tags, weight=0.9) :
    df_temps = merge_answser (df_results, df_answer)
    #计算F1_Score
    df_temps['base_model_v1_rr_score'] = df_temps[['vid_list', 'vid']].apply(lambda x : RR(x['vid_list'], x['vid']), axis=1)
    #计算多样性
    df_temps['diversity_score'] = df_temps['vid_list'].apply(lambda x : compute_diversity_sim(x, vid_to_tags)[1])
    df_temps['final_score'] = df_temps[['base_model_v1_rr_score', 'diversity_score']].apply(lambda x : x['base_model_v1_rr_score'] * weight + x['diversity_score'] * (1-weight) if x['base_model_v1_rr_score'] > 0 else x['base_model_v1_rr_score'], axis=1)    
    base_model_v1_rr_score = df_temps['base_model_v1_rr_score'].mean()
    diversity_score = df_temps['diversity_score'].mean()
    final_score = df_temps['final_score'].mean()
    print ("base_model_v1_rr_score:%s", base_model_v1_rr_score)    
    print ("diversity_score:%s", diversity_score)
    print ("final_score:%s", final_score)
    return df_temps, final_score
