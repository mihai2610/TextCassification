import json
import os
import pandas as pd


def create_dataframe(path):
    fields = ['text', 'label']
    df = pd.read_csv(path, usecols=fields) #nrows=3000
    return df


def get_data():
    return create_dataframe("dataset_2/train.csv")
