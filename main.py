import pandas as pd
from dataset_reader import load_data
from trainer import train



if __name__ == "__main__":
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 5000)
    pd.set_option('display.width', 4000)

    train_x, train_y = load_data()


    #data.to_csv("data.csv", sep=',', index=True, decimal='.')


    train(train_x, train_y)

    
