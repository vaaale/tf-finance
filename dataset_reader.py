import quandl
import pandas as pd
import pickle
import os.path
from features import fut_features,stock_features
import numpy as np

DEBUG = False

TOKEN = "WM8sJvsKsnrHmRA_TkZD"
fut_ticker_list   = ["CHRIS/ICE_T1", "CHRIS/ICE_T2", "CHRIS/ICE_T3", "CHRIS/ICE_T4", "CHRIS/ICE_T5", "CHRIS/ICE_T6"]
stock_ticker_list = ["YAHOO/OL_NHY","YAHOO/OL_STL","YAHOO/OL_MHG","YAHOO/OL_TEL","YAHOO/OL_VEI"]

#fut_ticker_list   = ["CHRIS/ICE_T1"]
#stock_ticker_list = ["YAHOO/OL_NHY"]

# Open   High    Low  Settle  Change   Wave   Volume  Prev. Day Open Interest  EFP Volume  EFS Volume  Block Volume
FUT_COL_IND   = [0, 1, 2, 3, 4, 6]
STOCK_COL_IND = [0, 1, 2, 5, 4]


def download_data():
    labels = pd.DataFrame()
    data = pd.DataFrame()
    for t in fut_ticker_list:
        print("Loading {0}".format(t))
        new_data = quandl.get(t, authtoken=TOKEN).iloc[:, FUT_COL_IND]
        new_data.columns = ['open', 'high', 'low', 'close', 'change', 'volume']
        new_data, new_labels = fut_features(new_data)
        data = pd.concat([data, new_data.iloc[:, 6:]], axis=1)
        #data = pd.concat([data, new_data], axis=1)
        labels = pd.concat([labels, new_labels], axis=1)


    for s in stock_ticker_list:
        print("Loading {0}".format(s))
        new_data = quandl.get(s, authtoken=TOKEN).iloc[:, STOCK_COL_IND]
        new_data.columns = ['open', 'high', 'low', 'close', 'volume']
        new_data, new_labels = stock_features(new_data)
        data = pd.concat([data, new_data.iloc[:, 5:]], axis=1)
        #data = pd.concat([data, new_data], axis=1)
        labels = pd.concat([labels, new_labels], axis=1)

    data = data.fillna(0)
    labels = labels.fillna(0)

    if not DEBUG:
        with open("data.pickle", "wb") as f:
            pickle.dump((data, labels), f)

    return data, labels


def load_data():
    if not DEBUG and os.path.isfile("data.pickle"):
        print("Loading from pickled data...")
        with open("data.pickle", "rb") as f:
            train_x, train_y =  pickle.load(f)
    else:
        print("Downloading data....")
        train_x, train_y = download_data()


    return train_x.values.tolist(), train_y.values.tolist()


def data_iterator(batch_size, seq_len):
    """ A simple data iterator """
    offset = 1585
    train_x, train_y = load_data()
    train_x = train_x[offset:]
    train_y = train_y[offset:]

    tot_len = len(train_x)
    round_offset = int(tot_len / (batch_size *  seq_len)) * (batch_size *  seq_len)
    train_x = train_x[(tot_len - round_offset):]
    train_y = train_y[(tot_len - round_offset):]
    print "Length ", tot_len
    print "Offset ", round_offset
    print "New length", len(train_x)
    while True:
        #batch_size = 128
        for batch_idx in range(0, len(train_x), batch_size*seq_len):
            images_batch = train_x[batch_idx:batch_idx+batch_size * seq_len]
            labels_batch = train_y[batch_idx:batch_idx+batch_size]
            yield np.array(images_batch), np.array(labels_batch)



if __name__ == "__main__":
    DEBUG = False
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    fut_ticker_list   = ["CHRIS/ICE_T1"]
    stock_ticker_list = ["YAHOO/OL_NHY"]

    iter = data_iterator(100, 10)
    x, y = iter.next()



