
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 13:09:26 2017

@author: lanlandetian
"""

import math
import operator


def ItemSimilarity(train):
    #calculate co-rated users between items
    #构建用户-物品表
    C =dict()
    N = dict()
    for u,items in train.items():
        for i in items:
            N.setdefault(i,0)
            N[i] += 1
            C.setdefault(i,{})
            for j in items:
                if i == j:
                    continue
                C[i].setdefault(j,0)
                C[i][j] += 1

    #calculate finial similarity matrix W
    W = C.copy()
    for i,related_items in C.items():
        for j,cij in related_items.items():
            W[i][j] = cij / math.sqrt(N[i] * N[j])
    return W

    
def Recommend(user_id,train, W,K = 3):
    rank = dict()
    ru = train[user_id]
    for i,pi in ru.items():
        for j,wij in sorted(W[i].items(), \
                           key = operator.itemgetter(1), reverse = True)[0:K]:
            if j in ru:
                continue
            rank.setdefault(j,0)
            rank[j] += pi * wij
    return rank
    
                           
def Recommendation(users, train, W, K = 3):
    result = dict()
    for user in users:
        rank = Recommend(user,train,W,K)
        R = sorted(rank.items(), key = operator.itemgetter(1), \
                   reverse = True)
        result[user] = R
    return result
'''2'''
'''
#1根据比例设置useritem数据 ：trainSet.setdefault(user, {})  2：trainSet[user][cookbook] = rating
#2 cookbook_popular  每个菜谱出现数量
#3 同时出现计算相似  m1和m2同时被用户购买
                    self.cookbook_sim_matrix.setdefault(m1, {}) #cookbook_sim_matrix：m1 ,m2同时出现的次数
                    self.cookbook_sim_matrix[m1].setdefault(m2, 0)
                    
                    self.cookbook_sim_matrix[m1][m2] = count / math.sqrt(self.cookbook_popular[m1] * self.cookbook_popular[m2])# 余弦相似度
#4 # 排名的依据——>推荐菜谱与该已看菜谱的相似度(累计)*用户对已看菜谱的评分
#5 针对目标用户U，对其看过的每部菜谱找到K部相似的菜谱，并推荐其N部菜谱
'''
################################################################################################################################
import random
import math
from operator import itemgetter
class ItemBasedCF:
    # 初始化参数
    def __init__(self):
        # 找到相似的20个菜谱，为目标用户推荐10个菜谱
        # K值：找到和已经看过菜谱最相似的20个菜谱
        self.n_sim_cookbook = 20
        # N值: 将其中前10名推荐给用户
        self.n_rec_cookbook = 10

        # 将数据集划分为训练集和测试集
        self.trainSet = {}
        self.testSet = {}

        # 用户相似度矩阵
        self.cookbook_sim_matrix = {}
        self.cookbook_popular = {}
        self.cookbook_count = 0


    # 读文件得到“用户-菜谱”数据(基于比例划分数据)
    def get_dataset(self, filename, pivot=0.75):
        trainset_len = 0
        testset_len = 0
        for line in filename.readlines():  
            user, cookbook, rating, timestamp = line.split(',')
            if(random.random()<pivot):
                self.trainSet.setdefault(user, {}) #user后跟字典
                self.trainSet[user][cookbook] = rating
                trainset_len += 1
            else:
                self.testSet.setdefault(user, {})
                self.testSet[user][cookbook] = rating
                testset_len += 1


    # 计算菜谱之间的相似度
    def calc_cookbook_sim(self):
        for user, cookbooks in self.trainSet.items():
            for cookbook in cookbooks:
                if cookbook not in self.cookbook_popular:
                    self.cookbook_popular[cookbook] = 0
                self.cookbook_popular[cookbook] += 1

        # self.cookbook_count = len(self.cookbook_popular) #cookbook_popular每个菜谱出现数量    cookbook_count：菜谱个数
        # print("Total cookbook number = %d" % self.cookbook_count)

        for user, cookbooks in self.trainSet.items():
            for m1 in cookbooks:
                for m2 in cookbooks:
                    if m1 == m2:
                        continue
                    self.cookbook_sim_matrix.setdefault(m1, {}) #cookbook_sim_matrix：m1 ,m2同时出现的次数
                    self.cookbook_sim_matrix[m1].setdefault(m2, 0)
                    # 朴素计数
                    #weight = 1
                    # 根据用户活跃度进行加权(item-IUF)
                    weight = 1/math.log2(1+len(cookbooks))
                    self.cookbook_sim_matrix[m1][m2] += weight
        print("Build co-rated users matrix success!")

        # 计算菜谱之间的相似性
        print("Calculating cookbook similarity matrix ...")
        for m1, related_cookbooks in self.cookbook_sim_matrix.items():
            mx = 0 # wix中最大的值
            for m2, count in related_cookbooks.items():
                # 注意0向量的处理，即某菜谱的用户数为0
                if self.cookbook_popular[m1] == 0 or self.cookbook_popular[m2] == 0:
                    self.cookbook_sim_matrix[m1][m2] = 0
                else:
                    self.cookbook_sim_matrix[m1][m2] = count / math.sqrt(self.cookbook_popular[m1] * self.cookbook_popular[m2])# 余弦相似度
                # 更新最大值
                mx = max(self.cookbook_sim_matrix[m1][m2],mx)
            # 进行相似度归一化(Item-Norm)
            for m2, count in related_cookbooks.items():
                self.cookbook_sim_matrix[m1][m2] /= mx
        print('Calculate cookbook similarity matrix success!')


    # 针对目标用户U，对其看过的每部菜谱找到K部相似的菜谱，并推荐其N部菜谱
    def recommend(self, user):
        K = self.n_sim_cookbook
        N = self.n_rec_cookbook
        rank = {}
        watched_cookbooks = self.trainSet[user]

        for cookbook, rating in watched_cookbooks.items():
            # 得到与看过菜谱最相似的K个菜谱
            for related_cookbook, w in sorted(self.cookbook_sim_matrix[cookbook].items(), key=itemgetter(1), reverse=True)[:K]:
                # 去掉已经看过的菜谱
                if related_cookbook in watched_cookbooks:
                    continue
                rank.setdefault(related_cookbook, 0)
                # 排名的依据——>推荐菜谱与该已看菜谱的相似度(累计)*用户对已看菜谱的评分
                rank[related_cookbook] += w * float(rating)
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[:N]

    # 产生推荐并通过准确率、召回率、覆盖率和F-Measure指数进行评估
    def evaluate(self):
        print('Evaluating start ...')
        N = self.n_rec_cookbook
        # 准确率和召回率
        hit = 0
        rec_count = 0
        test_count = 0
        # 覆盖率
        all_rec_cookbooks = set()

        for i, user in enumerate(self.trainSet):
            test_moives = self.testSet.get(user, {})
            rec_cookbooks = self.recommend(user)
            for cookbook, w in rec_cookbooks:
                if cookbook in test_moives:
                    hit += 1
                all_rec_cookbooks.add(cookbook)
            rec_count += N
            test_count += len(test_moives)

        precision = hit / (1.0 * rec_count)
        recall = hit / (1.0 * test_count)
        coverage = len(all_rec_cookbooks) / (1.0 * self.cookbook_count)
        # F1测量
        alpha = 1
        fmeasure = ((alpha * alpha + 1) * precision * recall) / (alpha * alpha * (precision + recall))
        print('precisioin=%.4f\trecall=%.4f\ncoverage=%.4f\tF-Measure=%.4f\n' % (precision, recall, coverage, fmeasure))


if __name__ == '__main__':
    rating_file = 'ml-latest-small/ratings.csv'
    itemCF = ItemBasedCF() # 初始化
    itemCF.get_dataset(rating_file) # 划分数据集
    itemCF.calc_cookbook_sim() # 计算菜谱相似度
