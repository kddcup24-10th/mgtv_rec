import random
import numpy as np
import warnings
import polars as pl
import pandas as pd
import json
warnings.filterwarnings("ignore")
from dataloader import load_data
from trainer import train

seed = 3407
random.seed(seed) 
np.random.seed(seed) 

def load_train_data(data_path):
    print("load data")
    train_df = pd.read_csv(data_path + '/' + 'train_df.csv')
    valid_df = pd.read_csv(data_path + '/' + 'valid_df.csv')
    df_valid_lable = pl.read_csv(data_path + '/' + 'df_valid_label.csv')
    test_df = pd.read_csv(data_path + '/' + 'test_df.csv')
    with open(data_path + '/' + 'vid_to_tags.json', 'r') as f:
        vid_to_tags = json.load(f)
    return train_df, valid_df, df_valid_lable, test_df, vid_to_tags
    
def save_train_data(data_path, train_df, valid_df, df_valid_label, test_df, vid_to_tags):
    print("save data")
    train_df.to_csv(data_path + '/' + 'train_df.csv', index=False)
    valid_df.to_csv(data_path + '/' + 'valid_df.csv', index=False)
    df_valid_label.write_csv(data_path + '/' + 'df_valid_label.csv')
    test_df.to_csv(data_path + '/' + 'test_df.csv', index=False)
    with open(data_path + '/' + 'vid_to_tags.json', 'w') as f:
        json.dump(vid_to_tags, f)
    
def main():
    data_path = "../A"
    train_df, valid_df, df_valid_label, test_df, vid_to_tags = load_data(data_path)
    save_train_data(data_path + "/train_data", train_df, valid_df, df_valid_label, test_df, vid_to_tags)
    train_df, valid_df, df_valid_label, test_df, vid_to_tags = load_train_data(data_path + "/train_data")
    print(train_df.shape, valid_df.shape, df_valid_label.shape, test_df.shape, len(vid_to_tags))
    print("Train/Valid/Test Size:", train_df.shape, valid_df.shape, test_df.shape)
    train(train_df, valid_df, df_valid_label, test_df, vid_to_tags)


if __name__ == "__main__":
    print("start")

    main()
    