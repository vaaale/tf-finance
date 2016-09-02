import pandas as pd


def fut_features(data):
    labels = pd.DataFrame()
    data = pd.concat([data, data.pct_change()], axis=1)
    data.columns = ['open', 'high', 'low', 'close', 'change', 'volume',
                    'open_pct', 'high_pct', 'low_pct', 'close_pct', 'change_pct', 'volume_pct']

    data['open_close_delta'] = (data.close - data.open) / data.open
    data['close_open_delta'] = (data.open - data.close.shift(1)) / data.open
    data['open_high_delta'] = (data.high - data.open) / data.open
    data['open_low_delta'] = (data.low - data.open) / data.open
    data['high_close_delta'] = (data.high - data.close) / data.high
    data['low_close_delta'] = (data.low - data.close) / data.low
    data['high_low_delta'] = (data.high - data.low) / data.high
    data['vol_delta'] = (data.volume - data.volume.shift(1)) / data.volume

    #labels['tgt_open'] = data.open.shift(-1)
    #labels['tgt_close'] = data.close.shift(-1)
    labels['tgt_open_delta'] = data.open.shift(-1) - data.open
    labels['tgt_close_delta'] = data.close.shift(-1) - data.close

    #data = data.iloc[:, 6:]

    return data, labels


def stock_features(data):
    labels = pd.DataFrame()
    data = pd.concat([data, data.pct_change()], axis=1)
    data.columns = ['open', 'high', 'low', 'close', 'volume',
                    'open_pct', 'high_pct', 'low_pct', 'close_pct', 'volume_pct']

    data['open_close_delta'] = (data.close - data.open) / data.open
    data['close_open_delta'] = (data.open - data.close.shift(1)) / data.open
    data['open_high_delta'] = (data.high - data.open) / data.open
    data['open_low_delta'] = (data.low - data.open) / data.open
    data['high_close_delta'] = (data.high - data.close) / data.high
    data['low_close_delta'] = (data.low - data.close) / data.low
    data['high_low_delta'] = (data.high - data.low) / data.high
    data['vol_delta'] = (data.volume - data.volume.shift(1)) / data.volume

    #labels['tgt_open'] = data.open.shift(-1)
    #labels['tgt_close'] = data.close.shift(-1)
    labels['tgt_open_delta'] = data.open.shift(-1) - data.open
    labels['tgt_close_delta'] = data.close.shift(-1) - data.close

    #data = data.iloc[:,5:]

    return data, labels
