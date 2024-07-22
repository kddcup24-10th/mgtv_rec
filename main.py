import random
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from dataloader import load_data
from trainer import train

seed = 3407
random.seed(seed) 
np.random.seed(seed) 

def main():
    data_path = "../A"
    data_path_wu = '../data_v4/A'

    data_path = data_path_wu

    train_df, valid_df, df_valid_label, test_df, vid_to_tags = load_data(data_path)
    print("Train/Valid/Test Size:", train_df.shape, valid_df.shape, test_df.shape)
    train(train_df, valid_df, df_valid_label, test_df, vid_to_tags)


if __name__ == "__main__":
    print("start")

    main()
    