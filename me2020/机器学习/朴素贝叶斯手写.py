import csv
import string
import numpy as np
import math
# 目标函数：给定一篇文章(d)，计算属于各个分类(c) 的概率，以概率最大的分类作为最终结果。p（c|d）选最大
# 公式：https://zhuanlan.zhihu.com/p/43301674
# https://zhuanlan.zhihu.com/p/26262151参靠公式
'''文本
# ham	For fear of fainting with the of all that housework you just did? Quick have a cuppa			
# spam	Thanks for your subscription to Ringtone UK your mobile will be charged å£5/month Please confirm by replying YES or NO. If you reply NO you will not be charged			
# ham	Yup... Ok i go home look at the timings then i msg Ì_ again... Xuhui going to learn on 2nd may too but her lesson is at 8am			
# ham	Oops, I'll let you know when my roommate's done			
# ham	I see the letter B on my car			
# ham	Anything lor... U decide...

'''
'''
步骤：
log每个类别的比例 log_priors ：某个分类的概率
每个词语计数/句子长度+总长度的list    /每个label中各个词语的比例  ：计算某一个特征在某一个分类中出现的次数概率获得特征概率

预测：输入特征
属于a类别的概率=log_priors[a]+log_likelihoods每个特征的概率*输入的这个特征 之和
选最大的类别'''



def load_data(filename, train_ratio):
    with open(filename, "rb") as f:
        csv_reader = csv.reader(f)
        csv_reader.next()  # header
        dataset = [(line[0], line[1]) for line in csv_reader]

    np.random.shuffle(dataset)
    train_size = int(len(dataset) * train_ratio)
    return dataset[:train_size], dataset[train_size:]


def train(train_set):
    total_doc_cnt = len(train_set)

    label_doc_cnt = {}#label字典
    bigdoc_words = {}#文字字典

    for label, doc in train_set:
        if label not in label_doc_cnt:
            # init
            label_doc_cnt[label] = 0
            bigdoc_words[label] = []

        label_doc_cnt[label] += 1 #label的数量
        bigdoc_words[label].extend([
            w.strip(string.punctuation) for w in doc.split()])#某个label的文本集合bigdoc_words

    vocabulary = set()
    for words in bigdoc_words.values():
        vocabulary |= set(words) #vocabulary所有文本单词

    V = len(vocabulary)
    log_priors = {label: math.log(1.0 * cnt / total_doc_cnt) for label, cnt in label_doc_cnt.items()}#log每个类别的比例 log_priors

    log_likelihoods = dict()
    for label, words in bigdoc_words.items():
        word_cnt = len(words) + V
        log_likelihoods[label] = [math.log(1.0 * (1 + words.count(w)) / word_cnt) for w in vocabulary]#每个词语计数/句子长度+总长度的list    /每个label中各个词语的比例

    return log_priors, log_likelihoods, vocabulary

def predict(log_priors, log_likelihoods, vocabulary, input_text, expect_label=None):
    words = {w.strip(string.punctuation) for w in input_text.split()}

    prob_max = 0
    label_max = None

    probs = {}  # tmp for log
    for label, likelihood in log_likelihoods.items():
        prob = log_priors[label] + sum([p for w, p in zip(vocabulary, likelihood) if w in words])#属于a类别的概率=log_priors[a]+log_likelihoods每个词语的概率*输入的这个词语 之和
        probs[label] = prob

        if not prob_max or prob > prob_max:
            prob_max = prob
            label_max = label

    if expect_label and expect_label != label_max:
        print '---'
        print 'expect: %s, got: %s' % (expect_label, label_max)
        print probs
        print input_text

    return label_max


def main():
    filename = 'input/spam.csv'
    train_ratio = 0.75
    train_data, test_data = load_data(filename, train_ratio)

    print('data loaded. train: {}, test: {}').format(
        len(train_data), len(test_data))

    # train the model
    log_priors, log_likelihoods, vocabulary = train(train_data)
    print 'model trained. log_priors: {}, V(vocabulary word count): {}'.format(log_priors, len(vocabulary))

    pos_true = 0
    pos_false = 0
    neg_false = 0
    neg_true = 0

    for label, text in test_data:
        got = predict(log_priors, log_likelihoods, vocabulary, text, label)
        if label != got:
            if label == 'spam':
                pos_false += 1
            else:
                neg_false += 1
        else:
            if label == 'spam':
                pos_true += 1
            else:
                neg_true += 1

    print 'positive(spam) true: %s, false: %s' % (pos_true, pos_false)
    print 'negative true: %s, false: %s' % (neg_true, neg_false)
    print 'Precision: %.2f%%, Recall: %.2f%%' % (
        100.0 * pos_true / (pos_true + pos_false),
        100.0 * pos_true / (pos_true + neg_false),
        )


if __name__ == '__main__':
    main()