import json
import os
import pandas as pd


def create_dataframe(first_path, second_path):
    news, is_rumour = [], []
    for _dir in os.listdir(first_path):
        new_pth_1 = first_path + '/' + _dir + '/reactions'
        data_source = json.load(open(first_path + '/' + _dir + '/' + 'source-tweet/' + _dir + '.json'))
        news.append(data_source['text'])
        is_rumour.append(0)
        for _file in os.listdir(new_pth_1):
            data_reactions = json.load(open(new_pth_1 + '/' + _file))
            news.append(data_reactions['text'])
            is_rumour.append(0)

    for _dir in os.listdir(second_path):
        new_pth_0 = second_path + '/' + _dir + '/reactions'
        data_source = json.load(open(second_path + '/' + _dir + '/' + 'source-tweet/' + _dir + '.json'))
        news.append(data_source['text'])
        is_rumour.append(1)
        for _file in os.listdir(new_pth_0):
            data_reactions = json.load(open(new_pth_0 + '/' + _file))
            news.append(data_reactions['text'])
            is_rumour.append(1)

    return pd.DataFrame({"text": news, "is_rumour": is_rumour})


def get_data():
    path_to_dataset = '../datasets/pheme_rnr_dataset/'
    df = pd.DataFrame({"text": [], "is_rumour": []})
    with os.scandir(path_to_dataset) as it:
        first = 0
        for entry in it:
            if not entry.name.startswith('.') and entry.is_dir() and first == 0:
                first = 1
                print(entry.name)
                _first_path = path_to_dataset + entry.name + '/rumours'
                _second_path = path_to_dataset + entry.name + '/non-rumours'
                local_df = create_dataframe(first_path=_first_path,
                                           second_path=_second_path)
                df = df.append(local_df)
    return df
