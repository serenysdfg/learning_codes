# -*- coding: utf-8 -*-
# @createTime : 2019/3/27 22:43
# @Author : sereny
import os
from surprise import SVD
from surprise import SVDpp
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise import KNNBasic
from surprise import BaselineOnly
from surprise import Reader
from surprise.model_selection import KFold
from surprise.model_selection import cross_validate
from surprise.model_selection import GridSearchCV

file_path = os.path.expanduser('item_user_rate_time.txt')
reader = Reader(line_format='user item rating timestamp', sep=',')
surprise_data = Dataset.load_from_file(file_path, reader=reader)

all_trainset = surprise_data.build_full_trainset()
algo = KNNBasic(k=40,min_k=3,sim_options={'user_based': True}) # sim_options={'name': 'cosine','user_based': True} cosine/msd/pearson/pearson_baseline
algo.fit(all_trainset)