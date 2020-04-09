'''
#usercf：物品被用户u，v同时购买  itemcf： m1和m2同时被用户购买
#主要
item_user[item].add(user) #物品添加使用的用户  item_user.setdefault(item, set())
双重循环：user_sim_matrix[u][v] += 1   N[u] += 1
 user_sim_matrix[u][v] = con_items_count / math.sqrt(N[u] * N[v]) # #用户相似度
 根据用户相似度排序，相似度高推荐的物品的sim高，全部用户的sim相加在排序'''

from collections import defaultdict
import math
from operator import itemgetter
import sys
from util.utils import load_file, save_file


class UserCF(object):

    def __init__(self):
        self.user_sim_matrix = self.user_similarity() # 计算用户协同矩阵



    def _init_train(self, origin_data):
        self.train = dict()
        for user, item, _ in origin_data:
            self.train.setdefault(user, set())
            self.train[user].add(item)

    def user_similarity(self):
        item_user = dict()
        for user, items in self.train.items():
            for item in items:
                item_user.setdefault(item, set())
                item_user[item].add(user) #物品添加使用的用户

        # 建立用户协同过滤矩阵 
        user_sim_matrix = dict()
        N = defaultdict(int)  # 记录用户购买商品次数
        for item, users in item_user.items():
            for u in users:
                N[u] += 1
                for v in users:
                    if u == v:
                        continue
                    user_sim_matrix.setdefault(u, defaultdict(int))
                    user_sim_matrix[u][v] += 1      #usercf：物品被用户u，v同时购买  itemcf： m1和m2同时被用户购买

        # 计算相关度
        for u, related_users in user_sim_matrix.items():
            for v, con_items_count in related_users.items():
                user_sim_matrix[u][v] = con_items_count / math.sqrt(N[u] * N[v]) #用户相似度

        return user_sim_matrix

    def recommend(self, user, N, K):
        related_items = self.train.get(user, set)
        recommmens = dict()
        for v, sim in sorted(self.user_sim_matrix.get(user, dict).items(), #user_sim_matrix,物品之间的相关度
                             key=itemgetter(1), reverse=True)[:K]:  #前k个相似用户的相似性相加？？
            for item in self.train[v]:
                if item in related_items:
                    continue
                recommmens.setdefault(item, 0.)
                recommmens[item] += sim

        return dict(sorted(recommmens.items(), key=itemgetter(1), reverse=True)[: N])

    def recommend_users(self, users, N, K):
        recommends = dict()
        for user in users:
            user_recommends = list(self.recommend(user, N, K).keys())
            recommends[user] = user_recommends

        return recommends
