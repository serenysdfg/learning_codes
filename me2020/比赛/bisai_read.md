# ***\*模型\****

他们使用两层 Stacking Model，第一层采用逻辑回归、随机森林、XGBoost 算法，第二层又采用 XGBoost 算法把第一层的结果融合

# ***\*特征工程\****

方法：直接百度某个特征处理

### ***\*图\****

![img](D:/ruanjiandata/Typora_mdpic/wps44.jpg) 

 

# <img src="D:/ruanjiandata/Typora_mdpic/wps45.jpg" alt="img" style="zoom:200%;" /> 

<img src="D:/ruanjiandata/Typora_mdpic/wps46.jpg" alt="img" style="zoom:200%;" /> 

## ***\*730特征\****

### ***\*分类和序数特征\****

![img](D:/ruanjiandata/Typora_mdpic/wps47.jpg) 

### ***\*时间与坐标特征\****

o 时间特征的常见类型

o 周期性特征：如 day of week

o 自从（或距离）某个事件的时间

o 行不相关事件：如距离某个共同时刻的时间

o 行相关事件：如距离下一个假日的时间（每行对应的假日时间不同）

o 两个日期之间的差值

o 坐标特征的常见类型

o 来自数据中的有趣的地点

o 坐标聚类的中心

o 聚合数据（如某个地点周边的垃圾桶数量）

o 对于树模型，可以对坐标适当旋转作为一个新的特征，提高准确率

 

 

 

 

 

# ***\*特征工程\****

## **·** ***\*数据处理\****

### ***\*1数据清洗-做完才好看缺失\****

先统计单特征

df=df_trainuser[1:100].copy()

print(df['start_work_date'].describe())

print(df['start_work_date'].value_counts())

print(df['start_work_date'].value_counts().sort_index())

\#画一下图异常值

 

\# 单列字段清洗-去空格

df['商品名称'] = df['商品名称'].map(lambda s : s.strip())

df['A']=df['A'].map(str.strip)  # 去除两边空格

df['A']=df['A'].map(str.lstrip)  # 去除左边空格

df['A']=df['A'].map(str.rstrip)  # 去除右边空格

df['A']=df['A'].map(str.upper)  # 转大写

df['A']=df['A'].map(str.lower)  # 转小写

df['A']=df['A'].map(str.title)  # 首字母大写

\# 字段切分，并创建新特征

df.loc[:,"区域"] = df['仓库'].map(lambda s:s[0:2])

\# 转换某特征(列)的数据格式

df['行号'] = df['行号'].astype(float)  

\# 转化时间格式

df['time']=pd.to_datetime(df['time'])

 

 

### ***\*缺失处理：先处理缺失再value_count\****

https://www.zhihu.com/question/26639110

\# 缺失值判断(在原数据对象以T/F替换)

df.isnull()

df['A'].isnull()

\# 缺失值计数方法

\# 方法一

df['A'].isnull().value_counts()

\>>> True   68629

  False    1

  Name: A, dtype: int64

\# 方法二

df['A'].isnull().sum()

\>>> 68629

df.isnull().sum()

\>>> 仓库      0

  货号      0

  条码      2

  规格    62290

\# 默认axi=0，how=‘any’，按行，任意一行有NaN就整列丢弃

df.dropna()

df.dropna(axis=1)

\# 一行中全部为NaN的，才丢弃

df.driopna(how='all')

\# 保留至少3个非空值的行：一行中有3个值是非空的就保留

df.dropna(thresh=3)

\# 缺失值填充

df.fillna(0)

![img](D:/ruanjiandata/Typora_mdpic/wps48.jpg) 

#### ***\*缺失值较少的特征处理\*******\*10%-值填充\****

![img](D:/ruanjiandata/Typora_mdpic/wps49.jpg)![img](file:///C:\Users\sereny\AppData\Local\Temp\ksohtml16892\wps50.jpg) 

 

#### ***\*值替换\****

\# 将df的A列中 -999 全部替换成空值

df["A"].replace(-999, np.nan)

\#-999和1000 均替换成空值

obj.replace([-999,1000],  np.nan)

\# -999替换成空值，1000替换成0

obj.replace([-999,1000],  [np.nan, 0])

\# 同上，写法不同，更清晰

obj.replace({-999:np.nan, 1000:0})

#### ***\*重复值\****

\# 返回布尔向量、矩阵

df['A'].duplicated()

df.duplicated()

\# 保留k1列中的唯一值的行，默认保留第一行

df.drop_duplicated(["k1"])

\# 保留 k1和k2 组合的唯一值的行，take_last=True 保留最后一行

df.drop_duplicated(["k1","k2"], take_last=True)

 

 

### ***\*数值特征\****

#### ***\*#log变换\*******\*，长尾分布，单位不同\****

df['require_nums_count_log'] = np.log((1+ df['require_nums_count']))

income_log_mean = np.round(np.mean(df['require_nums_count_log']), 2)

 

### ***\*类别特征\****

Onehot

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

one_hot_feature = ['desire_jd_salary_id', 'cur_salary_id', 'cur_degree_id', 'live_city_id','city']

for feature in one_hot_feature:

  try:

​    df[feature] = LabelEncoder().fit_transform(df[feature].apply(int))

  except:

​    df[feature] = LabelEncoder().fit_transform(df[feature])

特征工程中通常要处理类别特征，如学历、性别、城市等，经常的做法是转换成dummy变量。会有LabelEncoder、OneHotEncoder、factorize、get_dummies4种方法。 

#### ***\*有级别的分类：学历等\****

df['cur_degree_id']=df['cur_degree_id'].fillna('0') #有缺失

degreeLevelDict={'博士':4,'中技':4,'EMBA':4,'MBA':4,'硕士':4,'本科':3,'大专':2,'中专':2,'高中':1,'初中':1,'其他':1,'0':0} 

df['cur_degree_id']=df['cur_degree_id'].map(degreeLevelDict)

 

user_fea_map=['cur_degree_id']

for fea in user_fea_map:

  df[fea+'_count'] = df[fea].map(df[fea].value_counts())

df['cur_degree_id_count']

 

\#年龄分段

fcc_survey_df['Age_bin_round'] = np.array(np.floor( np.array(fcc_survey_df['Age']) / 10.))

fcc_survey_df[['ID.x', 'Age', 'Age_bin_round']].iloc[1071:1076]

 

 

#### ***\*分桶特征-年龄\****

\#年龄分桶18-22，23-26，27-30，30-35，36-30

df=df_trainuser[1:100].copy()

 

bin_ranges = [18, 22, 26,30, 35, 80]

bin_names = [1, 2, 3, 4, 5]

df['age_bin'] = pd.cut(np.array( df['age']),  bins=bin_ranges,labels=bin_names)

df[['age','age_bin']]

![img](D:/ruanjiandata/Typora_mdpic/wps51.jpg) 

分桶大小必须足够小，分桶大小必须足够大，样本均匀

### **·** ***\*特征选择\****

### **·** ***\*维度压缩\****

 

 

# ***\*常用工具\****

### ***\*1\*******\*read_csv\****

%time userAll = pd.read_csv(***\*'omp_train_user.csv'\****, usecols=[***\*'user_id'\****,***\*'item_id'\****,***\*'behavior_type'\****,***\*'time'\****]) **#4.5 s**

### ***\*2大数据存储\****

1

***\*import\**** pickle
userAll.to_pickle(open(***\*'userAll.pkl'\****, ***\*'wb'\****)) **#存储****
**pkl_file = open(***\*'userAll.pkl'\****, ***\*'rb'\****)**#使用****
**userAll = pickle.load(pkl_file)

2

%time userSub.to_csv(***\*'userSub.csv'\****)

 

 

### ***\*3进度条\*******\*tqdm\****

range(1000)一共的长度

***\*import\**** time
***\*from\**** tqdm ***\*import\**** *
***\*for\**** i ***\*in\**** tqdm(range(1000)):
  time.sleep(.01)   **#进度条每0.1s前进一次，总时间为1000\*0.1=100s**

# ***\*1穿衣\*******\*搭配\*******\*1.13\****

# ***\*1.9\****

## ***\*淘宝穿衣搭配比赛（5天）\****

度量分词的重要性：TFIDF

词语的相似度：***\*余弦相似度\****

 

***\*各种模型的变化，添加函数增加泛化能力，p范数等等\****

 

***\*下载文件\****

svn checkout https://github.com/xubo245/SparkLearning/trunk/docs

将“tree/master”改成“trunk”

 

## ***\*方法总结\****

用相似性，tfidf，选择最相似的同类别

## ***\*1中文分词（标题分词）\****

### [基于 Gensim 的 Word2Vec 实践](https://zhuanlan.zhihu.com/p/24961011)

### [Gensim入门教程](http://www.cnblogs.com/iloveai/p/gensim_tutorial.html)

 

### ***\*1分词工具：\****https://segmentfault.com/a/1190000003971257

#### ***\*1结巴分词\****

找出文档的相似度

幸好***\*gensim\****提供了这样的工具，具体的处理思路如下***\*，对于中文文本的比较，\****先需要做分词处理，***\*根据分词的结果生成一个字典\****，然后再根据字典把***\*原文档转化成向量\****。然后去***\*训练相似度\****。把对应的文档构建一个索引，原文描述如下：

 

\# 生成字典和向量语料

dictionary = corpora.Dictionary(texts)  **#构建字典**

corpus = [dictionary.doc2bow(text) ***\*for\**** text ***\*in\**** texts] #变成向量

 

#### ***\*1特点\****

1，支持三种分词模式：

  a,精确模式，试图将句子最精确地切开，适合文本分析； 
  b,全模式，把句子中所有的可以成词的词语都扫描出来, 速度非常快，但是不能解决歧义； 
  c,搜索引擎模式，在精确模式的基础上，对长词再次切分，提高召回率，适合用于搜索引擎分词。

***\*import\**** jieba
seg_list = jieba.cut(***\*"我来到北京清华大学"\****, cut_all=***\*True\****)
print (***\*"\**** ***\*"\****.join(seg_list))  **# 全模式****
**seg_list = jieba.cut(***\*"他来到了网易杭研大厦"\****)  **# 默认是精确模式****
**seg_list = jieba.cut_for_search(***\*"小明硕士毕业于中国科学院计算所，后在日本京都大学深造"\****)  **# 搜索引擎模式**

#### ***\*2词性标注\****

***\*import\**** jieba.posseg ***\*as\**** pseg

words = pseg.cut(line)
***\*for\**** w ***\*in\**** words:
  print ( f, str(w))

#### ***\*2\*******\*.1分词后存储\****

***\*import\**** jieba.posseg ***\*as\**** pseg
filename = ***\*'others/mycorpus1.txt'\*******\*
\****fn = open(***\*'others/mycorpus.txt'\****, ***\*"r"\****,encoding=***\*'UTF-8'\****)
f = open(filename, ***\*"w+"\****,encoding=***\*'UTF-8'\****)
***\*for\**** line ***\*in\**** fn.readlines():
  words = jieba.cut(line)
  ***\*for\**** w ***\*in\**** words:
    f.write(str(w))
    f.write(***\*'  '\****)
    **# print >> f, str(w)****
**f.close()
fn.close()

 

#### ***\*2.2\*******\*获取文本相似度\****

**#coding=utf-8****
**datapath=***\*'others/mycorpus1.txt'\*******\*
\****querypath=***\*'others/11.txt'\*******\*
\****storepath=***\*'others/2.txt'\*******\*
\*******\*import\**** logging
***\*from\**** gensim ***\*import\**** corpora, models, similarities

***\*def\**** similarity(datapath, querypath, storepath):
  logging.basicConfig(format=***\*'%(asctime)s : %(levelname)s : %(message)s'\****, level=logging.INFO)

  ***\*class\**** MyCorpus(object):
    ***\*def\**** __iter__(self):
      ***\*for\**** line ***\*in\**** open(datapath, ***\*'r'\****, encoding=***\*'UTF-8'\****):
        ***\*yield\**** line.split()


  Corp = MyCorpus()
  dictionary = corpora.Dictionary(Corp)
  corpus = [dictionary.doc2bow(text) ***\*for\**** text ***\*in\**** Corp]

  tfidf = models.TfidfModel(corpus)
  corpus_tfidf = tfidf[corpus]

  query = open(querypath, ***\*'r'\****, encoding=***\*'UTF-8'\****).readline()
  vec_bow = dictionary.doc2bow(query.split())
  vec_tfidf = tfidf[vec_bow]

  index = similarities.MatrixSimilarity(corpus_tfidf)
  sims = index[vec_tfidf]

  similarity = list(sims)

  sim_file = open(storepath, ***\*'w'\****)
  ***\*for\**** i ***\*in\**** similarity:
    sim_file.write(str(i)+***\*'\*******\*\n\*******\*'\****)
  sim_file.close()

 

#### ***\*3 Word2vec\****

word2vec可以在百万数量级的词典和上亿的数据集上进行高效地训练；其次，该工具得到的训练结果——词向量（word embedding），可以很好地度量词与词之间的相似性。随着深度学习（Deep Learning）在自然语言处理中应用的普及，很多人误以为word2vec是一种深度学习算法。其实word2vec算法的背后是一个浅层神经网络。另外需要强调的一点是，word2vec是一个计算word vector的开源工具

 

 

word2vec模型其实就是简单化的神经网络，不是以前的one-hot

 word2vec中常见的模型有：CBOW（ContinuousBag Of Words Model），Skip-gram（Continuous Skip-gram Model），两者的模型图如下所示：由此可见，前者是由上下文推当前词，后者是由当前词推上下文；

 

## ***\*2代码学习\****

##### ***\*1交集\****

***\*def\**** m_intersection(a,b):**
**  ***\*return\**** list(set(a).intersection(set(b)))

##### ***\*2tfidf\****


***\*def\**** tfidf_items(items,infos):
  **"""****
**  **获取指定商品集的tfidf****
**  ***\*:param\**** **items:****
**  ***\*:param\**** **infos:****
**  ***\*:return\******:****
**  **"""****
**  corpus = []
  ***\*for\**** i ***\*in\**** items:
    corpus.append(infos[i]) #corpus是商品信息向量集合
  tf_idf = tfidf(corpus)
  items_info = {}
  ***\*for\**** i ***\*in\**** range(len(tf_idf)):
    items_info[items[i]] = tf_idf[i]
  ***\*return\**** items_info

items_info = tfidf_items(items,infos)  **# 获取指定商品集的tfidf  #1732**

 

##### ***\*3tfidf\****

***\*import\**** ***\*jieba\*******\*
\*******\*import\**** ***\*jieba.posseg\**** ***\*as\**** ***\*pseg\*******\*
\*******\*import\**** ***\*os\*******\*
\*******\*import\**** ***\*sys\*******\*
\*******\*from\**** ***\*sklearn\**** ***\*import\**** ***\*feature_extraction\*******\*
\*******\*from\**** ***\*sklearn.feature_extraction.text\**** ***\*import\**** ***\*TfidfTransformer\*******\*
\*******\*from\**** ***\*sklearn.feature_extraction.text\**** ***\*import\**** ***\*CountVectorizer\*******\*
\*******\*vectorizer=CountVectorizer()\*******\**#该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频\**\******\**
\**\******\*transformer=TfidfTransformer()\*******\**#该类会统计每个词语的tf-idf权值\**\******\**
\**\******\*tfidf=transformer.fit_transform(vectorizer.fit_transform(corpus))\*******\*
\*******\*
\*******\*word = vectorizer.get_feature_names()\****  ***\**# 获取词袋模型中的所有词语\**\******\**
\**\******\*weight = tfidf.toarray()\****  ***\**# 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重\**\******\**
\**\******\*for\**** ***\*i\**** ***\*in\**** ***\*range\*******\*(\*******\*len\*******\*(weight)):\****  ***\**# 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重\**\******\**
\**\***  ***\*print\*******\*
\****  ***\*u"-------这里输出第"\*******\*, i,\**** ***\*u"类文本的词语tf-idf权重------"\*******\*
\****  ***\*for\**** ***\*j\**** ***\*in\**** ***\*range\*******\*(\*******\*len\*******\*(word)):\*******\*
\****    ***\*print\*******\*
\****    ***\*word[j], weight[i][j]\****

 

 

count_vec.fit_transform的结果是一个巨大的矩阵。我们可以看到上表中有大量的0，因此sklearn在内部实现上使用了稀疏矩阵。

***\*def\**** tfidf(corpus):
  **"""****
**  **获取语料库中每一个句子的tf-idf值****
**  ***\*:param\**** **corpus: 语料库****
**  **"""****
**  vectorizer = CountVectorizer()
  x = vectorizer.fit_transform(corpus)
  a = x.toarray()
  **# print vectorizer.get_feature_names()****
**  transformer = TfidfTransformer()
  tfidf = transformer.fit_transform(a)
  ***\*return\**** tfidf.toarray()



##### ***\*4字典\****

items_info.values()

##### ***\*5\**** ***\*collections\****

是Python内建的一个集合模块，提供了许多有用的集合类

1 namedtuple是一个函数，它用来创建一个自定义的tuple对象，并且规定了tuple元素的个数，并可以用属性而不是索引来引用tuple的某个元素。

2 、deque除了实现list的append()和pop()外，还支持appendleft()和popleft()，这样就可以非常高效地往头部添加或删除元素。

3、使用dict时，如果引用的Key不存在，就会抛出KeyError。如果希望key不存在时，返回一个默认值，就可以用defaultdict：

4、使用dict时，Key是无序的。在对dict做迭代时，我们无法确定Key的顺序

5、Counter是一个简单的计数器，例如，统计字符出现的个数：

# ***\*2商场定位1.15\****

## ***\*方法总结gen_sample.py\****

目标：利用xgboost建立模型

 

一个一个商城来，wifi类型变成向量，，每个经纬度变成wifi的向量类型

**模型：##自变量：经纬度,wifi类型（向量表示）#y商店标签训练**

LabelEncoder用于商店标签的获取和还原，变成数字

模型得到每个商店的概率之后predSorted = (-pred).argsort()用于获取最大概率的商店

 

结果存储：res[[***\*'row_id'\****, ***\*'shop_id'\****]].to_csv(***\*'../data/sub.csv'\****, index=***\*False\****)

 

## ***\*1代码\****

### ***\*1 defaultdict\****

***\*from\**** collections ***\*import\**** defaultdict**
****#构造规则****
**wifi_to_shops = defaultdict(***\*lambda\**** : defaultdict(***\*lambda\**** :0))**#没有的都是0****
*****\*for\**** line ***\*in\**** user_shop_hehavior.values:  **#1138015****
**  wifi = sorted([wifi.split(***\*'|'\****) ***\*for\**** wifi ***\*in\**** line[5].split(***\*';'\****)],key=***\*lambda\**** x:int(x[1]),reverse=***\*True\****)[0] **#从大到小wifi强度,只要最强的****
**  wifi_to_shops[wifi[0]][line[1]] = wifi_to_shops[wifi[0]][line[1]] + 1**#wifi最强识别码+店铺所在位置，，+1  #75879wifi_to_shops wifi_to_shops['b_52574361']['s_2303143']=6**

 

### ***\*2写到csv\****

result = pd.DataFrame({***\*'row_id'\****:evalution.row_id,***\*'shop_id'\****:preds})
result.fillna(***\*'s_666'\****).to_csv(***\*'wifi_baseline.csv'\****,index=***\*None\****) **#随便填的 这里还能提高不少**

### ***\*4遍历iterrows\*******\* \****

[python里使用**iterrows()**对dataframe进行遍历 ](http://www.baidu.com/link?url=buwim_ALFwTIDNAA0EZIjCwc4uJ-fC5KQTeVgWjSejMbTruj-ceXqhXPCeapAFN8AZtbyGGYFjnBXw-zf0DT710UYhcuu6pAVPbgEVQq-Ba&wd=&eqid=e6048e1b0000a0e5000000065a5cb6e6)

otu = pd.read_csv("otu.txt",sep="\t")

for index,row in otu.iterrows():

 print index

 print row

### ***\*4reset_index\*******\*：\****

pandas contact 之后，一定要记得用reset_index去处理index,

sub_train = pd.concat([sub_train.reset_index(), pd.DataFrame(train_set)], axis=1) **#reset_index改变index**

### ***\*5\**** ***\*LabelEncoder\****

**用途：，变成数字后可以用于模型训练xgboost**

 

简单来说 LabelEncoder 是对不连续的数字或者文本进行编号，字符串变成整形，一样的字符串就是一样的数字

LabelEncoder可以将标签分配一个0—n_classes-1之间的编码 
将各种标签分配一个可数的连续编号

\>> > ***\*from\**** sklearn ***\*import\**** preprocessing
\>> > le = preprocessing.LabelEncoder()
\>> > le.fit([1, 2, 2, 6])
LabelEncoder()
\>> > le.classes_
array([1, 2, 6])
\>> > le.transform([1, 1, 2, 6])  **# Transform Categories Into Integers****
**array([0, 0, 1, 2], dtype=int64)
\>> > le.inverse_transform([0, 0, 1, 2])  **# Transform Integers Into Categories****
**array([1, 1, 2, 6])

 

\#OneHotEncoder 用于将表示分类的数据扩维：

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder()

ohe.fit([[1],[2],[3],[4]])

ohe.transform([2],[3],[1],[4]).toarray()

输出：[ [0,1,0,0] , [0,0,1,0] , [1,0,0,0] ,[0,0,0,1] ] 

### ***\*6to_pickle\****

pandas的dataframe永久存储；通过pickle模块的反序列化操作，我们能够从文件中创建上一次程序保存的对象

output = open('train_samples_top10.pkl', 'wb')

pickle.dump(train_samples, output)

output.close()

 

from pandas.core.frame import DataFrame

train_samples1=DataFrame(train_samples)

train_samples1.to_pickle(open('train1_samples_top10.pkl', 'wb'))

 


　基本接口：

pickle.dump(obj, file, [,protocol])将对象obj保存到文件file中去。　

pickle.load(file)：从file中读取一个字符串，并将它重构为原来的python对象。
file:类文件对象，有read()和readline()接口

　A Simple Code
**#使用pickle模块将数据对象保存到文件****
****
*****\*import\**** pickle

data1 = {***\*'a'\****: [1, 2.0, 3, 4+6j],
     ***\*'b'\****: (***\*'string'\****, ***\*u'Unicode string'\****),
     ***\*'c'\****: ***\*None\****}
selfref_list = [1, 2, 3]
selfref_list.append(selfref_list)

output = open(***\*'data.pkl'\****, ***\*'wb'\****)
**# Pickle dictionary using protocol 0.****
**pickle.dump(data1, output)
**# Pickle the list using the highest protocol available.****
**pickle.dump(selfref_list, output, -1)

output.close()


**#使用pickle模块从文件中重构python对象****
****
*****\*import\**** pprint, pickle
pkl_file = open(***\*'data.pkl'\****, ***\*'rb'\****)

data1 = pickle.load(pkl_file)
pprint.pprint(data1)
data2 = pickle.load(pkl_file)
pprint.pprint(data2)

pkl_file.close()

### ***\*7 DictVectorizer\****

· fit()：训练算法，设置内部参数。

· transform()：数据转换。

· fit_transform()：合并fit和transform两个方法。

 

***\*from\**** sklearn.feature_extraction ***\*import\**** DictVectorizer  
measurements = [  
  {***\*'city'\****: ***\*'Dubai'\****, ***\*'temperature'\****: 33.},  
   {***\*'city'\****: ***\*'London'\****, ***\*'temperature'\****: 12.},  
   {***\*'city'\****: ***\*'San Fransisco'\****, ***\*'temperature'\****: 18.},  
 ]  
vec = DictVectorizer()  
print(vec.fit_transform(measurements).toarray())***\*
\*******\*array([[  1.,  0.,  0.,  33.],\**** ***\*
\****    ***\*[  0.,  1.,  0.,  12.],\**** ***\*
\****    ***\*[  0.,  0.,  1.,  18.]])\**** ***\*
\*******\*"""\****  ***\*
\****print(vec.get_feature_names())  ***\*输出['city=Dubai', 'city=London', 'city=San Fransisco', 'temperature']\**** 

 

### ***\*8\**** ***\*iloc和loc\****

\1. loc——通过行标签索引行数据

•  data = [[1,2,3],[4,5,6]]  
•  index = [0,1]  
•  columns=[***\*'a'\****,***\*'b'\****,***\*'c'\****]  
•  df = pd.DataFrame(data=data, index=index, columns=columns)  
•  print df.loc[1] 

\2. iloc—通过行号获取行数据； 想要获取哪一行就输入该行数字，通过行标签索引会报错 
print df.iloc[0:]  **#同样通过行号可以索引多行****
**print df.iloc[:,[1]]  **#索引列数据****
**3 ix——结合前两种的混合索引：通过行号索引+通过行标签索引
print df.ix[1]
print df.ix[***\*'e'\****]

### ***\*9\*******\*argsort\****

按照大小顺序排序

x = np.array([3, 1, 2])
 np.argsort(x)
array([1, 2, 0])#最小的序号是1，最大

 

### ***\*10apply在pandas\****

apply 是 pandas 库的一个很重要的函数，多和 groupby 函数一起用，也可以直接用于 DataFrame 和 Series 对象。主要用于数据聚合运算，可以很方便的对分组进行现有的运算和自定义的运算。

 

float_df = fandango_films[float_columns]
**# 对选择的列计算总分，在lambda中的x是一个Series，代表了某一列****
**count = float_df.apply(***\*lambda\**** x: np.sum(x))
如果要在行上使用apply()方法，只要指定参数axis = 1即可
**# 计算每部电影的平均分****
**means = float_df.apply(***\*lambda\**** x: np.mean(x), axis = 1)

### ***\*11\****[***\*pandas基础之按行取数（\*******\**DataFrame\**\******\*）\****](http://www.baidu.com/link?url=Z9vUrspWqHw_KqnhamSNKCRfrGayX65d89n6TcWnw_wJbBLWSp_LTpuSA3qTeZEvOHLVH2gTGk0rr4m-KB0yW2HHo1M5lKFHEGUBk4K61w_&wd=&eqid=d13fbcf70000ef6b000000065a5dc55c)

Pd[1:4]

### ***\*12\*******\*Python 多进程 multiprocessing.Pool类详解\****

它与 threading.Thread类似，可以利用multiprocessing.Process对象来创建一个进程。该进程可以允许放在Python程序内部编写的函数中。该Process对象与Thread对象的用法相同，拥有is_alive()、join([timeout])、run()、start()、terminate()等方法。属性有：authkey、daemon（要通过start()设置）、exitcode(进程在运行时为None、如果为–N，表示被信号N结束）、name、pid。此外multiprocessing包中也有Lock/Event/Semaphore/Condition类，用来同步进程，其用法也与threading包中的同名类一样。multiprocessing的很大一部份与threading使用同一套API，只不过换到了多进程的情境。

## ***\*2xgboost\****

 [XGBoost](https://github.com/dmlc/xgboost)是近年来很受追捧的机器学习算法

***\*读入数据的方式\****有多种，比如 load_svmlight_file()以及DMatrix().这两种的效果是一样的。

***\*模型也有多种使用方法\****，比如xgboost.XGBClassifier().fit()；  xgboost.train(),

***\*其中load_svmlight_file()和 xgboost.XGBClassifier().fit()是xgboost的Scikit-Learn Wrapper 接口，另外两种是Python调用的普通函数\****

[***\*http://blog.csdn.net/vitodi/article/details/60141301\****](http://blog.csdn.net/vitodi/article/details/60141301)

xgbtrain = xgb.DMatrix(sub_train[feature], sub_train[***\*'label'\****])
xgbtest = xgb.DMatrix(sub_test[feature])
watchlist = [(xgbtrain, ***\*'train'\****), (xgbtrain, ***\*'test'\****)]
***\*model = xgb.train(params, xgbtrain, num_rounds, watchlist,\**** ***\*early_stopping_rounds\*******\*=\*******\*15\*******\*,\**** ***\*verbose_eval\*******\*=\*******\*False\*******\*)\****
preds = model.predict(xgbtest) **#4880**

 

 

 

***\*import\**** numpy ***\*as\**** np
***\*import\**** xgboost ***\*as\**** xgb
***\*from\**** sklearn.metrics ***\*import\**** accuracy_score
***\*import\**** numpy ***\*as\**** np
***\*import\**** matplotlib.pyplot ***\*as\**** plt
***\*from\**** sklearn ***\*import\**** datasets,svm
***\*from\**** sklearn.datasets ***\*import\**** load_svmlight_file
iris = datasets.load_iris()

train_X = iris.data[:, :2]  **# we only take the first two features.**train_Y = iris.target
test_X = iris.data[:, :2]  **# we only take the first two features.**test_Y = iris.target

**### first xgboost****
**xgm = xgb.XGBClassifier()   

xgm.fit(train_X, train_Y)

y_pred = xgm.predict(test_X)
predictions = [round(value) ***\*for\**** value ***\*in\**** y_pred]
**# evaluate predictions****
**accuracy = accuracy_score(test_Y, predictions)
print(xgm) print(***\*"XGBoost Accuracy: %.2f%%"\**** % (accuracy * 100.0))

**### second xgboost****
****
**xg_train = xgb.DMatrix( train_X, label=train_Y)
xg_test = xgb.DMatrix(test_X, label=test_Y)
**# setup parameters for xgboost****
**param = {}
**# use softmax multi-class classification**
watchlist = [ (xg_train,***\*'train'\****), (xg_test, ***\*'test'\****) ]
num_round = 100
bst = xgb.train(param, xg_train, num_round, watchlist );
**# get prediction****
**pred = bst.predict( xg_test );
print (***\*'predicting, classification error=%f'\**** % (sum( int(pred[i]) != test_Y[i] ***\*for\**** i ***\*in\**** range(len(test_Y))) / float(len(test_Y)) ))

**# do the same thing again, but output probabilities****
**param[***\*'objective'\****] = ***\*'multi:softprob'\*******\*
\****bst = xgb.train(param, xg_train, num_round, watchlist );
**# Note: this convention has been changed since xgboost-unity****
****# get prediction, this is in 1D array, need reshape to (ndata, nclass)****
**yprob = bst.predict( xg_test ).reshape( test_Y.shape[0], 3 )
ylabel = np.argmax(yprob, axis=1)
print (***\*'predicting, classification error=%f'\**** % (sum( int(ylabel[i]) != test_Y[i] ***\*for\**** i ***\*in\**** range(len(test_Y))) / float(len(test_Y)) ))

**## compare two data input method.****
****
**X_train, y_train=load_svmlight_file(***\*'/Users/AureDi/Desktop/heart_scale'\****) **# loading datasets in the svmlight/libsvm format.****
**xgm = xgb.XGBClassifier(max_depth=3)

### ***\*2.2参数怎么调整不知道\****

#### ***\*2.2.1GBM参数\****


选择一个相对来说***\*稍微高一点的learning rate\****。一般默认的值是0.1，不过针对不同的问题，0.05到0.2之间都可以

\1. 决定***\*当前learning rate下最优的决定树数量\****。它的值应该在40-70之间。记得选择一个你的电脑还能快速运行的值，因为之后这些树会用来做很多测试和调参。

\2. 接着***\*调节树参数\****来调整learning rate和树的数量。我们可以选择不同的参数来定义一个决定树，后面会有这方面的例子

**3.** ***\*降低learning rate\****，同时会增加相应的决定树数量使得模型更加稳健

为了决定boosting参数，我们得先设定一些参数的初始值，可以像下面这样：

**1.** ***\*min_ samples_ split=500:\**** 这个值应该在总样本数的0.5-1%之间，由于我们研究的是不均等分类问题，我们可以取这个区间里一个比较小的数，500。

**2.** ***\*min_ samples_ leaf=50:\**** 可以凭感觉选一个合适的数，只要不会造成过度拟合。同样因为不均等分类的原因，这里我们选择一个比较小的值。

**3.** ***\*max_ depth=8:\**** 根据观察数和自变量数，这个值应该在5-8之间。这里我们的数据有87000行，49列，所以我们先选深度为8。

**4.** ***\*max_ features=’sqrt’:\**** 经验上一般都选择平方根。

**5.** ***\*subsample=0.8:\**** 开始的时候一般就用0.8

#### ***\*2.2.2参数调优的一般方法\****

http://blog.csdn.net/han_xiaoyang/article/details/52665396

http://blog.sina.com.cn/s/blog_4766fd440102wqdy.html



 

尽管有两种booster可供选择，我这里只介绍***\*tree booster\****，因为它的表现远远胜过***\*linear booster\****，

我们会使用和GBM中相似的方法。需要进行如下步骤：

\1. 选择较高的***\*学习速率(learning rate)\****。一般情况下，学习速率的值为0.1。但是，对于不同的问题，理想的学习速率有时候会在0.05到0.3之间波动。选择***\*对应于此学习速率的理想决策树数量\****。XGBoost有一个很有用的函数“cv”，这个函数可以在每一次迭代中使用交叉验证，并返回理想的决策树数量。

\2. 对于给定的学习速率和决策树数量，进行***\*决策树特定参数调优\****(max_depth, min_child_weight, gamma, subsample, colsample_bytree)。在确定一棵树的过程中，我们可以选择不同的参数，待会儿我会举例说明。

\3. xgboost的***\*正则化参数\****的调优。(lambda, alpha)。这些参数可以降低模型的复杂度，从而提高模型的表现。

\4. 降低学习速率，确定理想参数。

咱们一起详细地一步步进行这些操作。

第一步：确定学习速率和tree_based 参数调优的估计器数目

为了确定boosting参数，我们要先给其它参数一个初始值。咱们先按如下方法取值：

1、max_depth = 5 :这个参数的取值最好在3-10之间。我选的起始值为5，但是你也可以选择其它的值。起始值在4-6之间都是不错的选择。

2、min_child_weight = 1:在这里选了一个比较小的值，因为这是一个极不平衡的分类问题。因此，某些叶子节点下的值会比较小。

3、gamma = 0: 起始值也可以选其它比较小的值，在0.1到0.2之间就可以。这个参数后继也是要调整的。

4、subsample, colsample_bytree = 0.8: 这个是最常见的初始值了。典型值的范围在0.5-0.9之间。

5、scale_pos_weight = 1: 这个值是因为类别十分不平衡。 
注意哦，上面这些参数的值只是一个初始的估计值，后继需要调优。这里把学习速率就设成默认的0.1。然后用xgboost中的cv函数来确定最佳的决策树数量。前文中的函数可以完成这个工作。

 

## ***\*3\**** ***\*LightGBM\****

比XGBOOST更快--LightGBM介绍

# ***\*常用python\****

 

pandas基本数据管理

***\*import\**** pandas ***\*as\**** pd
***\*import\**** numpy ***\*as\**** np
***\*from\**** pandas ***\*import\**** Series, DataFrame
pd.read_table()
**# 假设新建一个数据框为df****
**df.head(), df.tail(), df.shape(), df.dtypes()，df.info
df[***\*'var'\****] = values **# 新建变量****
**np.where(), np.logical_and, np.less, np.greater **# 变量重编码****
**df.index, df.columns, df.index.map, df.columns.map, df.index.rename, df.index.reanme **# 变量重命名****
**df.isnull, df.notnull, df.dropna, df,fillna **#  缺失值处理****
**pd.to_datetime**# 日期值****
**df.astype **# 数据类型转换****
**df.sorte_index df.sort.values **# 排序****
**pd.merge, pd.concat, pd.appedn **# 合并数据集****
**df.ix[], df[], df.loc[] **# 数据取子集****
**df.sample **# 抽样**

 

 

 

# ***\*3离线赛\****

F值？

## ***\*聚类\****

### ***\*kmeans聚类\****

高分化

### ***\*两\*******\*步\*******\*聚类\****

两步聚类算法是在SPSS Modeler中使用的一种聚类算法

https://www.cnblogs.com/tiaozistudy/p/twostep_cluster_algorithm.html

## ***\*1方法\****

使用前一天的数据，然后攻击

 

2

观察数据，很多重复。。

step3：取user与item子集上的交集

 

## ***\*2代码\****

train_user=train_user.drop([***\*'user_geohash'\****],axis=1)

train_user.to_csv(***\*'output/sample_submission.csv'\****,index=***\*False\****)

train_user[***\*'item_id'\****].count()

##### ***\*3\****

\# 筛选出12月18号一天的数据
***\*import\**** re
regex=re.compile(***\*r'^2014-12-18+ \d+$'\****)  #regex=re.compile(r'^2014-12-18+ \d+$')
***\*def\**** date(column):
  ***\*if\**** re.match(regex,column[***\*'time'\****]):
    date,hour=column[***\*'time'\****].split(***\*' '\****)
    ***\*return\**** date
  ***\*else\****:
    ***\*return\**** ***\*'null'\*******\*
\****train_user[***\*'time'\****]=train_user.apply(date,axis=0)  #有
train_user=train_user[(train_user[***\*'time'\****] ==***\*'2014-12-18'\****)]

 

behavior_type1=train_user[***\*'behavior_type'\****].value_counts()

 

##### ***\*4重复行\****

userAll.duplicated().sum()**#检查有无重复行,重复的行总数11505107****
**itemSet = itemSub[[***\*'item_id'\****]].drop_duplicates()**#去除重复的行**

##### ***\*5取交集merge\****

userSub = pd.merge(userAll,itemSet,on = 'item_id',how = 'inner')

#####  ***\*6根据时间》》？？？\*******\*date_parser\**** 

data=pd.read_csv(***\*'c:/data.csv'\****,parse_dates=***\*True\****,keep_date_col = ***\*True\****) 
***\*or\*******\*
\****data=pd.read_csv(***\*'c:/data.csv'\****,parse_dates=[0]) 

***\*date_parser\**** : function, default None

用于解析日期的函数，默认使用dateutil.parser.parser来做转换。Pandas尝试使用三种不同的方式解析，如果遇到问题则使用下一种方式。

1.使用一个或者多个arrays（由parse_dates指定）作为参数；

2.连接指定多列字符串作为一个列作为参数；

3.每行调用一次date_parser函数来解析一个或者多个字符串（由parse_dates指定）作为参数。

 

parse_dates：可以是布尔型、int、ints或列名组成的list、dict，默认为False。如果为True，解析index。如果为int或列名，尝试解析所指定的列。如果是一个多列组成list，尝试把这些列组合起来当做时间来解析。（敲厉害！！）

 

##### ***\*7按照指定列排序\****

df.sort_values(by=***\*"sales"\**** , ascending=***\*False\****)

##### ***\*7\*******\*指定列为索引\****

df1.set_index(['c','d'])

仍然想把转换的列留下来，那么可以使用参数drop=False

df1.set_index([***\*'c'\****,***\*'d'\****],drop=***\*False\****)
如果想要把索引放回到列中，可以使用reset_index()。
df1.set_index([***\*'c'\****,***\*'d'\****]).reset_index()

userSubOneHot.reset_index()

 

***\*改变列的顺序\****

mid = df[***\*'Mid'\****]
df.drop(labels=[***\*'Mid'\****], axis=1,inplace = ***\*True\****)
df.insert(0, ***\*'Mid'\****, mid)

 

##### ***\*pandas使用get_dummies进行one-hot编码\****

pd.get_dummies(df[***\*a\*******\*'\****],prefix = ***\*'type'\****) #饿到type_1,type_2

2

***\*import\**** pandas ***\*as\**** pd

df = pd.DataFrame([
  [***\*'green'\****, ***\*'M'\****, 10.1, ***\*'class1'\****],
  [***\*'red'\****, ***\*'L'\****, 13.5, ***\*'class2'\****],
  [***\*'blue'\****, ***\*'XL'\****, 15.3, ***\*'class1'\****]])

df.columns = [***\*'color'\****, ***\*'size'\****, ***\*'prize'\****, ***\*'class label'\****]

size_mapping = {
  ***\*'XL'\****: 3,
  ***\*'L'\****: 2,
  ***\*'M'\****: 1}
df[***\*'size'\****] = df[***\*'size'\****].map(size_mapping)

class_mapping = {label: idx ***\*for\**** idx, label ***\*in\**** enumerate(set(df[***\*'class label'\****]))}
df[***\*'class label'\****] = df[***\*'class label'\****].map(class_mapping)  

 

pd.get_dummies(userSub[***\*'behavior_type'\****],prefix = ***\*'type'\****)

#### ***\*统计数据集\****

test_x = dataDay_load.ix[***\*'2014-12-17'\****,:]**#17号特征数据集，最为测试输入数据集****
**test_y = dataDay_load.ix[***\*'2014-12-18'\****,[***\*'user_id'\****,***\*'item_id'\****,***\*'type_4'\****]]**#18号购买行为作为测试标签数据集****
**testSet = pd.merge(test_x,test_y, on = [***\*'user_id'\****,***\*'item_id'\****],suffixes=(***\*'_x'\****,***\*'_y'\****), how = ***\*'left'\****).fillna(0.0)**#构成测试数据集****
**testSet[***\*'labels'\****] = testSet.type_4_y.map(***\*lambda\**** x: 1.0 ***\*if\**** x > 0.0 ***\*else\**** 0.0 ) #标签
testSet.to_csv(***\*'testSet.csv'\****)

 

 

## ***\*3学习模型\****

 

了解还要h

 

# ***\*医药数模中药别\****

 

 

## ***\*文本处理聚类\****

### ***\*1步骤\****

![img](D:/ruanjiandata/Typora_mdpic/wps52.jpg) ![img](file:///C:\Users\sereny\AppData\Local\Temp\ksohtml16892\wps53.png)

***\*文本文档—\****

***\*文本去燥\*******\*（html标签等无关数据）---\****

***\*文本预处理：中文分词+去除停用词\****

***\*文本建模表示：\****向量空间模型VSM(把文本文档的内容转化成为了特征空间中的矢量)/布尔模型，概率模型，语言模型

***\*(\*******\*特征降维\*******\*)：\****特征集合大，vsm矩阵稀疏：

特征降维方法可分为两种：一种是全局降维和局部降维；另一种是特征选择和特征抽取。

***\*特征选择（权重计算：\****在衡量文本间的相似度之前，对组成文本向量的特征项权重的计算是必须

征权重计算方法有：布尔权重、词频权重***\*、\*******\*TF-IDF\**** ***\*方法\****等。

***\*（tfidf）选取大于一个特定阈值的特征作为整个文档集的特征项\****

***\*文本相似度计算：LDA主题模型\****

***\*聚类\****

### ***\*2解释\****  

***\*VSM：\*******\*用文本集合的特征词表示文本，，组成向量\****

###  向量空间模型VSM 比较vs    第二步，TF-IDF***\* \****  ***\*http://blog.csdn.net/eastmount/article/details/49898133\****

 

***\*特征选择：\*******\*文本特征提取\****

特征提取算法：一般是构造评价函数，如信息增益，期望交叉上熵，文本证据权，互信息和IFTDF(常用)

 

特征降维方法之一——特征选择方法。

文本聚类中的无监督特征选择方法，它们是：单词贡献度、单词熵、单词权、文档频数。

 

3构建模型

计算“百度百科-故宫”和“互动百科-故宫”的消息盒相似度代码如下。基本步骤：
1.分别统计两个文档的关键词，读取txt文件，CountKey()函数统计
2.两篇文章的关键词合并成一个集合MergeKey()函数，相同的合并，不同的添加
3.计算每篇文章对于这个集合的词的词频 TF-IDF算法计算权重，此处仅词频
4.生成两篇文章各自的词频向量
5.计算两个向量的余弦相似度，值越大表示越相似   

## ***\*其他\****

### ***\*计算文本相似度\*******\*（121总结）\****

TFIDF：先提取不同文章文本为一个列表，一条一条，然后提取整个所有关键字，然后弄向量，然后利用纯数字计算相似度

 整个文本聚类过程可以先后分为两步：0分词1、计算文本集合各个文档中TD－IDF值，2，根据计算的结果，对文件集合用k-means聚类方法进行迭代聚类。计算相似度

***\*基本步骤包括：\****
    2.使用jieba结巴分词对文本进行中文分词，同时插入字典关于关键词；
    3.scikit-learn对文本内容进行tfidf计算并构造N*M矩阵(N个文档 M个特征词)；（TF-IDF计算权重越大表示该词条对这个文本的重要性越大）
    4.再使用K-means进行文本聚类(省略特征词过来降维过程);
    5.最后对聚类的结果进行简单的文本处理，按类簇归类，也可以计算P/R/F特征值；

### ***\*数据处理\****

#### ***\*分词后"夏"的词语提取，去除其他无关数据\*******\*（还没懂）\****

我要，夏，秋等，，然后进行one-hot,其他不要

#### ***\*选择药物\*******\*：\*******\*考虑距离和时间\****

 

***\*1\*******\*很多分类重复的数据\*******\*：\*******\*先分词\*******\*，\*******\*再将分类无量纲化处理\****，比如9类编程1-9，穿下秋冬混合的就用春夏都表示1，二维

2聚类：分类无量纲分的太多会导致效果不理想，删除占比较小的功效，类别，根据查询资料和其他聚类结果

 

 

#### ***\*填充数据\*******\*，\*******\*有很多无\*******\*，\*******\*同上的\****

1用excel处理countif将中文替换成数字（别名中含有美丽的都替换成1）

2、spss重新编码成不同变量，无-1同 -2.。筛选出来

 

分词

### ***\*LCTCLAS的分词系统\****

但是运行的时候还是会有错误，init的过程无法正常进行，输出了“Init ICTCLAS failed!”

于是查了下，发现Data文件夹下会以“（当天日期）.err”的一个错误文件

![img](D:/ruanjiandata/Typora_mdpic/wps54.png)
发现错误是：2017-06-01 11:36:41] D:/ICTCLAS2016\Data\NLPIR.user Not valid license or your license expired! Please feel free to contact pipy_zhang@msn.com! 

百度了下，发现需要下载新的授权文件

下载地址：https://github.com/NLPIR-team/NLPIR/tree/master/

找到[License](https://github.com/NLPIR-team/NLPIR/tree/master/License)，然后是[license for a month](https://github.com/NLPIR-team/NLPIR/tree/master/License/license for a month)，接下来是[license for a month](https://github.com/NLPIR-team/NLPIR/tree/master/License/license for a month)

然后把里面的[NLPIR.user](https://github.com/NLPIR-team/NLPIR/blob/master/License/license for a month/NLPIR-ICTCLAS分词系统授权/NLPIR.user)下载下载，放在data的文件夹里面，就可以正常运行了

 

哈工大信息检索研究中心 ***\*<\*******\*中文停用词》\****

 

 

## ***\*代码\**** 

#### ***\*sys.argv[]的用法简明解释\****

　sys.argv[]说白了就是一个从程序外部获取参数的桥梁

Sys.argv[ ]其实就是一个列表，里边的项为用户输入的参数，关键就是要明白这参数是从程序外部输入的，而非代码本身的什么地方，要想看到它的效果就应该将程序保存了，从外部来运行程序并给出参数。https://www.cnblogs.com/aland-1415/p/6613449.html

 

#### ***\*strip\****

s.strip(rm)**#  删除s字符串中开头、结尾处，位于rm删除序列的字符****
**s.lstrip(rm) **#删除s字符串中开头处，****
**s.rstrip(rm)**#删除s字符串中结尾处**

###### ***\*存储文件\****

 

sFilePath = ***\*'./tfidffile'\*******\*
\*******\*if not\**** os.path.exists(sFilePath):
  os.mkdir(sFilePath)

 

  **# 这里将每份文档词语的TF-IDF写入tfidffile文件夹中保存****
**  ***\*for\**** i ***\*in\**** range(len(weight)) :
　　　　 print ***\*"Writing tf-idf in the"\****,i,***\*" file\**** ***\*nt"\****,sFilePath+***\*'/'\****+string.zfill(i,5)+***\*'.txt'\****,***\*"--------"\*******\*
\****    f = open(sFilePath+***\*'/'\****+string.zfill(i,5)+***\*'.txt'\****,***\*'w+'\****)
    ***\*for\**** j ***\*in\**** range(len(word)) :
      f.write(word[j]+***\*"   "\****+str(weight[i][j])+***\*"\*******\*\n\*******\*"\****)
    f.close()

 

### ***\*读取文件小trick\****

###### ***\*Utf文件需要说明utf\*******\*-\*******\*8\****

Gbk不用

inputs = open(***\*'diaosi.txt'\****, ***\*'r'\****)  **# 加载要处理的文件的路径****
**outputs = open(***\*'diaosi1.txt'\****, ***\*'w'\****)  **# 加载处理后的文件路径**

***\*with\**** open(***\*'diaosi1.txt'\****, ***\*'r'\****) ***\*as\**** fr:  **#**

干嘛干嘛

###### ***\*1文件读取方式\**** 

w   以写方式打开，必要时创建文件 
a    以追加模式打开 (从 EOF 开始, 必要时创建新文件) 
r    以读写模式打开，如果文件不存在将***\*抛出异常\**** 
r+   以读写模式打开 
w+  以读写模式打开 (参见 w ) 
a+   以读写模式打开 (参见 a ) 
rb   以二进制读模式打开 
wb   以二进制写模式打开 (参见 w ) 
ab   以二进制追加模式打开 (参见 a ) 
rb+  以二进制读写模式打开 (参见 r+ ) 
wb+  以二进制读写模式打开 (参见 w+ ) 
ab+  以二进制读写模式打开 (参见 a+ )

2读取的是gbk要decode

3 line = eachLine.strip().decode(***\*'utf-8'\****, ***\*'ignore'\****)  **#去除每行首尾可能出现的空格，并转为**

######  ***\*4readline和writeline\****

 print  line.readline() 和 .readlines()之间的差异是后者一次读取整个文件，象 .read()一样。.readlines()自动将文件内容分析成一个行的列表，该列表可以由 Python 的 for... in ... 结构进行处理。另一方面，.readline()每次只读取一行，通常比 .readlines()慢得多。仅当没有足够内存可以一次读取整个文件时，才应该使用.readline()

 

writeline()是输出后换行，下次写会在下一行写。write()是输出后光标在行末不会换行，下次写会接着这行写

### ***\*python读取excel\****

读取csv

userAll = pd.read_csv(***\*'=user.csv'\****, usecols = [***\*'user_id'\****,***\*'item_id'\****]) **#4.5 s**

读取excel分词

df=pd.read_excel(***\*'zhongyao.xlsx'\****,***\*'DRUG_ZB'\****)
col_data= list(df.iloc[ :,8])

**#jieba.load_userdict("addwords.txt")****
*****\*def\**** seg_sentence(sentence):
  sentence_seged = jieba.cut(sentence.strip())
  stopwords =[line.strip() ***\*for\**** line ***\*in\**** open(***\*'stopwords.txt'\****, ***\*'r'\****,encoding=***\*'UTF-8'\****).readlines()]  **# 这里加载停用词的路径****
**  outstr = ***\*''\*******\*
\****  ***\*for\**** word ***\*in\**** sentence_seged:
    ***\*if\**** word ***\*not in\**** stopwords:
      ***\*if\**** word != ***\*'\*******\*\t\*******\*'\****:
        outstr += word
        outstr += ***\*" "\*******\*
\****  ***\*return\**** outstr

outputs = open(***\*'out.txt'\****, ***\*'w'\****)  **# 加载处理后的文件路径****
*****\*for\**** line ***\*in\**** col_data:
  line_seg = seg_sentence(line)
  outputs.write(line_seg+***\*'\*******\*\n\*******\*'\****)
outputs.close()

 

### ***\*split一个个词语读取\****

f=open(***\*'stopwords.txt'\****,***\*'r'\****,encoding=***\*'utf-8'\****)
f1=open(***\*'stopwords3.txt'\****,***\*'w+'\****,encoding=***\*'utf-8'\****)  **#必须是utf-8****
*****\*for\**** line ***\*in\**** f:
  words=line.split(***\*" "\****)
  ***\*for\**** word ***\*in\**** words:
   **#  print("  ".join(word))****
**    f1.write(word+***\*'\*******\*\n\*******\*'\****)
f.close()
f1.close()文本去除符号处理 **
**line = eachLine.strip().decode(***\*'utf-8'\****, ***\*'ignore'\****)    **#去除每行首尾可能出现的空格，并转为Unicode进行处理****
** line1 = re.sub(***\*"[0-9\s+\.\!\/_,$%^\*()?;；:-【】+\*******\*\"\'\*******\*]+|[+——！，;:。？、~@#￥%……&\*（）]+"\****.decode(***\*"utf8"\****), ***\*""\****.decode(***\*"utf8"\****),line)

### ***\*结巴分词\*******\*，\*******\*导入词典和去除停用词\*******\*（\*******\*得是一行行的\*******\*）\****

#### ***\*1、结巴分词简单例程\****


1、对TXT文本进行分词：

***\*import\**** jieba  **# 导入jieba****
*****\*with\**** open(***\*'text.txt'\****, ***\*'r'\****)***\*as\**** f:  **# 打开所需分词文本text.txt****
**  ***\*for\**** line ***\*in\**** f:
    seg = jieba.cut(line.strip(), cut_all=***\*False\****)  **# jieba分词****
**    print ( ***\*'/'\****.join(seg) ) **# 其中join可以将分词结果变为列表格式****
**2、统计词频：Counter
***\*from\**** collections ***\*import\**** Counter
total = []
***\*with\**** open(***\*'text.txt'\****, ***\*'r'\****)***\*as\**** f:
  ***\*for\**** line ***\*in\**** f:
    seg_list = jieba.lcut(line.strip(), cut_all=***\*False\****)  **# jieba.lcut 可以直接输出列表。****
**    ***\*for\**** word ***\*in\**** seg_list:
      total.append(word)
c = Counter(total)  **# 这里一定要顶格写，否则就进入到上面的for循环里面，出错。****
*****\*for\**** item ***\*in\**** c.most_common(5):  **# 数字5表示提取前5个高频词，可根据需要更改。****
**  print( item[0], item[1])


3、添加自定义词典
词典为TXT文本，其格式：一个词一行，每行三部分：词语、词频（可省）、词性（可省）。其中每部分用空格分隔。其代码如下：
jieba.load_userdict(***\*'userdict.txt'\****)  **# 核心jieba.load_userdict('name.txt')****
** 4、动态调整词典
add_word(word, freq=***\*None\****, tag=***\*None\****)和del_word(word)可以在程序中动态修改词典（加词，或删词）。

suggest_freq(segment, tune=***\*True\****) 可以调节每个单个词语的词频，使其能 / 不能被分出来。
例如：text.txt文本的第四行中“中将”被词典分为一个词，但其实“中”和“将”是分开的。但是词典里面也没有“中”和“将”，这时需要动态调整代码：
jieba.suggest_freq((***\*'中'\****, ***\*'将'\****), tune=***\*True\****)  **# True表示希望分出来，False表示不希望分出来。**

 

5停用词词典，做成字典

我之前是把停用词在程序里存入一个列表，然后分每个词时都循环一遍列表，这样特别浪费时间。后来把停用词做成字典就很快了

.**decode(**"**utf-8**", "**ignore**") 忽略其中有异常的编码,

stopwords={}
fstop = open(***\*'stopwords1.txt'\****, ***\*'r'\****,encoding=***\*'utf-8'\****)
***\*for\**** eachWord ***\*in\**** fstop:
  stopwords[eachWord.strip().decode(***\*'utf-8'\****, ***\*'ignore'\****)] = eachWord.strip().decode(***\*'utf-8'\****, ***\*'ignore'\****)
fstop.close()

 

· stopwords = {}.fromkeys(['的', '包括', '等', '是']) 

#### ***\*2\*******\*、代码示例\****

***\*from\**** collections ***\*import\**** Counter
***\*import\**** jieba

jieba.load_userdict(***\*"\*******\*addwords\*******\*.txt"\****)



**# 对句子进行分词****
*****\*def\**** seg_sentence(sentence):
  sentence_seged = jieba.cut(sentence.strip())
  stopwords =[line.strip() ***\*for\**** line ***\*in\**** open(***\*'stopwords.txt'\****, ***\*'r'\****,encoding=***\*'UTF-8'\****).readlines()]  **# 这里加载停用词的路径****
**  outstr = ***\*''\*******\*
\****  ***\*for\**** word ***\*in\**** sentence_seged:
    ***\*if\**** word ***\*not in\**** stopwords:
      ***\*if\**** word != ***\*'\*******\*\t\*******\*'\****:
        outstr += word
        outstr += ***\*" "\*******\*
\****  ***\*return\**** outstr


inputs = open(***\*'diaosi.txt'\****, ***\*'r'\****)  **# 加载要处理的文件的路径****
**outputs = open(***\*'diaosi11.txt'\****, ***\*'w'\****)  **# 加载处理后的文件路径****
*****\*for\**** line ***\*in\**** inputs:
  line_seg = seg_sentence(line)  **# 这里的返回值是字符串****
**  outputs.write(line_seg)
outputs.close()
inputs.close()


**# WordCount** **统计****
*****\*with\**** open(***\*'diaosi11.txt'\****, ***\*'r'\****) ***\*as\**** fr:  **# 读入已经去除停用词的文件,词语+数目****
**  data = jieba.cut(fr.read())
data = dict(Counter(data))
***\*with\**** open(***\*'diaosi22.txt'\****, ***\*'w'\****) ***\*as\**** fw:  **# 读入存储wordcount的文件路径****
**  ***\*for\**** k, v ***\*in\**** data.items():
    fw.write(***\*'%s,%d\*******\*\n\*******\*'\**** % (k, v))

 

### ***\*Wordcloud生成\****


**#生成词云****
**  **# coding = utf-8****
**  ***\*from\**** os ***\*import\**** path
  ***\*from\**** wordcloud ***\*import\**** WordCloud, ImageColorGenerator
  ***\*import\**** matplotlib.pyplot ***\*as\**** plt
  ***\*from\**** scipy.misc ***\*import\**** imread

  print  (path.dirname(***\*'diaosi2.txt'\****))
  d = path.dirname(***\*'F:/Projects/python/TEST/tianchi1_9/shumo118/'\****)   **#F:/Projects/python/TEST/tianchi1_9/shumo118/****
**  text = open(path.join(d, ***\*'diaosi2.txt'\****)).read()
  bg_pic = imread(path.join(d, ***\*'22.jpg'\****))

  **# 生成词云****
**  wordcloud = WordCloud(mask=bg_pic, background_color=***\*'white'\****, scale=1.5, font_path=***\*r'C:\Windows\Fonts\FZYTK.TTF'\****).generate(text)
  image_colors = ImageColorGenerator(bg_pic)
  **# 显示词云图片****
****
**  plt.imshow(wordcloud.recolor(color_func=image_colors))
  plt.axis(***\*'off'\****)
  plt.show()

  **# 保存图片****
**  wordcloud.to_file(path.join(d, ***\*'result.jpg'\****))‬

 

 

wordcloud.WordCloud(font_path=***\*None\****, width=400, height=200, margin=2, ranks_only=***\*None\****, prefer_horizontal=0.9,mask=***\*None\****, scale=1, color_func=***\*None\****, max_words=200, min_font_size=4, stopwords=***\*None\****, random_state=***\*None\****,background_color=***\*'black'\****, max_font_size=***\*None\****, font_step=1, mode=***\*'RGB'\****, relative_scaling=0.5, regexp=***\*None\****, collocations=***\*True\****,colormap=***\*None\****, normalize_plurals=***\*True\****)
font_path : string //字体路径，需要展现什么字体就把该字体路径+后缀名写上，如：font_path = ***\*'黑体.ttf'\*******\*
\****width : int (default=400) //输出的画布宽度，默认为400像素
height : int (default=200) //输出的画布高度，默认为200像素
prefer_horizontal : float (default=0.90) //词语水平方向排版出现的频率，默认 0.9 （所以词语垂直方向排版出现频率为 0.1 ）
mask : nd-array ***\*or None\**** (default=***\*None\****) //如果参数为空，则使用二维遮罩绘制词云。如果 mask 非空，设置的宽高值将被忽略，遮罩形状被 mask 取代。除全白（**#FFFFFF）的部分将不会绘制，其余部分会用于绘制词云。****
****## 如：bg_pic = imread('读取一张图片.png')，背景图片的画布一定要设置为白色（#FFFFFF），然后显示的形状为不是白色的其他颜色。可以用ps工具将自己要显示的形状复制到一个纯白色的画布上再保存，就ok了。****
**scale : float (default=1) //按照比例进行放大画布，如设置为1.5，则长和宽都是原来画布的1.5倍。
min_font_size : int (default=4) //显示的最小的字体大小
font_step : int (default=1) //字体步长，如果步长大于1，会加快运算但是可能导致结果出现较大的误差。
max_words : number (default=200) //要显示的词的最大个数
stopwords : set of strings ***\*or None\**** //设置需要屏蔽的词，如果为空，则使用内置的STOPWORDS
background_color : color value (default=”black”) //背景颜色，如background_color=***\*'white'\****,背景颜色为白色。
max_font_size : int ***\*or None\**** (default=***\*None\****) //显示的最大的字体大小
mode : string (default=”RGB”) //当参数为“RGBA”并且background_color不为空时，背景为透明。
relative_scaling : float (default=.5) //词频和字体大小的关联性
color_func : callable, default=***\*None\**** //生成新颜色的函数，如果为空，则使用 self.color_func
regexp : string ***\*or None\**** (optional) //使用正则表达式分隔输入的文本
collocations : bool, default=***\*True\**** //是否包括两个词的搭配
colormap : string ***\*or\**** matplotlib colormap, default=”viridis” //给每个单词随机分配颜色，若指定color_func，则忽略该方法。

fit_words(frequencies)  //根据词频生成词云
generate(text)  //根据文本生成词云
generate_from_frequencies(frequencies[, ...])  //根据词频生成词云
generate_from_text(text)   //根据文本生成词云
process_text(text)  //将长文本分词并去除屏蔽词（此处指英语，中文分词还是需要自己用别的库先行实现，使用上面的 fit_words(frequencies) ）
recolor([random_state, color_func, colormap])  //对现有输出重新着色。重新上色会比重新生成整个词云快很多。
to_array()  //转化为 numpy array
to_file(filename)  //输出到文件

 

### ***\*关键词提取\****

### [Python TF-IDF计算100份文档关键词权重](http://www.cnblogs.com/chenbjin/p/3851165.html)

目前***\*关键词抽取的算法\****的论文越来越复杂，结合的特征也越来越多，因此速度也当然越来越慢。如果是实际中使用还是建议一些简单的算法，如果要写论文拼效果则可以将各种***\*LDA、TextRank或者word2vec等深度学习模型加进去\****。特征选择的越多，当然效果也就越好了。

 

关键词获取可以通过两种方式来获取： 
   1、在使用jieba分词对文本进行处理之后，可以通过***\*统计词频\****来获取关键词：jieba.analyse***\*.extract_tags\****(news, topK=10)，获取词频在前10的作为关键词。 
   2、使用TF-IDF权重来进行关键词获取，首先需要对文本构建词频矩阵，其次才能使用向量求TF-IDF值。

 

http://blog.csdn.net/suibianshen2012/article/details/68927060

关键词抽取从方法来说大致有两种：

· 第一种是关键词分配，就是有一个给定的关键词库，然后新来一篇文档，从词库里面找出几个词语作为这篇文档的关键词；

· 第二种是关键词抽取，就是新来一篇文档，从文档中抽取一些词语作为这篇文档的关键词；

目前大多数领域无关的关键词抽取算法（领域无关算法的意思就是无论什么主题或者领域的文本都可以抽取关键词的算法）和它对应的库都是基于后者的。从逻辑上说，后者比前着在实际使用中更有意义

从算法的角度来看，关键词抽取算法主要有两类：

· 有监督学习算法，将关键词抽取过程视为二分类问题，先抽取出候选词，然后对于每个候选词划定标签，要么是关键词，要么不是关键词，然后训练关键词抽取分类器。当新来一篇文档时，抽取出所有的候选词，然后利用训练好的关键词抽取分类器，对各个候选词进行分类，最终将标签为关键词的候选词作为关键词；

· 无监督学习算法，先抽取出候选词，然后对各个候选词进行打分，然后输出topK个分值最高的候选词作为关键词***\*。根据打分的策略不同，有不同的算法，例如TF-IDF，TextRank\****等算法；

jieba分词系统中实现了两种关键词抽取算法，分别是基于***\*TF-IDF关键词抽取算法\****和基于***\*TextRank关键词抽取算法\****，两类算法均是无监督学习的算法，下面将会通过实例讲解介绍如何使用jieba分词的关键词抽取接口以及通过源码讲解其实现的原理。

###### ***\**jieba使用自定义停用词集合\**\******\** \**\******\*jieba\*******\*.\*******\*set_stop_words(\*******\*"stop_words.txt"\*******\*)\****

 

 

1jieba.analyse.extracr和tfidf效果类似

2textrank

3词频统计

###### ***\*代码extract_tags\****

***\*import\**** jieba.analyse

**#获取关键词**  **
**tags = jieba.analyse.extract_tags(text, topK=3)  

\#keywords = jieba.analys***\*e.extract_tags\****(content, topK=20) # 基于TF-IDF算法进行关键词抽取

print ***\*u"关键词:"\****  print ***\*" "\****.join(tags) 

***\*for\**** item ***\*in\**** keywords:
  **# 分别为关键词和相应的权重****
**  print item[0], item[1]

***\*
\******#** content**：待提取关键词的文本****
****#** topK**：返回关键词的数量，重要性从高到低排序****
****#** withWeight**：是否同时返回每个关键词的权重****
****#** allowPOS=()**：词性过滤，为空表示不过滤，若提供则仅返回符合词性要求的关键词****
****
**tag= jieba.analyse.extract_tags(content,topK=20,withWeight=***\*True\****,allowPOS=(***\*'ns'\****, ***\*'n'\****, ***\*'vn'\****, ***\*'v'\****))

###### ***\*1\**** ***\*tfidf最原始\****

一般都是读取到一个列表中然后算，得到一个tfidf矩阵

　scikit-learn包进行TF-IDF分词权重计算主要用到了两个类：CountVectorizer和TfidfTransformer。其中

　　CountVectorizer是通过fit_transform函数将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在第i个文本下的词频。即各个词语出现的次数，通过get_feature_names()可看到所有文本的关键字，通过toarray()可看到词频矩阵的结果。简例如下：

 

 HashingVectorizer，规定feature个数

**1.** ***\*from\**** sklearn.feature_extraction.text ***\*import\**** HashingVectorizer 

\2. vectorizer = HashingVectorizer(stop_words = 'english',non_negative = True,  n_features = 10000)

\3. fea_train = vectorizer.fit_transform(newsgroup_train.data) 

 

 

**# -\*-coding:utf-8-\*-****
*****\*import\**** jieba.analyse
***\*from\**** sklearn ***\*import\**** feature_extraction
***\*from\**** sklearn.feature_extraction.text ***\*import\**** TfidfTransformer
***\*from\**** sklearn.feature_extraction.text ***\*import\**** CountVectorizer

***\*"""\*******\*
\****    ***\*TF-IDF权重：\*******\*
\****      ***\*1、CountVectorizer 构建词频矩阵  2、TfidfTransformer 构建tfidf权值计算\*******\*
\****      ***\*3、文本的关键字4、对应的tfidf矩阵\*******\*
\*******\*"""\****

corpus = [
...   ***\*'This is the first document.'\****,
...   ***\*'This is the second second document.'\****,
... ]

  tfidf =transformer.fit_transform(vectorizer.fit_transform(corpus))


  \# 01、构建词频矩阵，将文本中的词语转换成词频矩阵
  vectorizer = CountVectorizer()
  X = vectorizer.fit_transform(corpus) **# a[i][j]:表示j词在第i个文本中的词频**
  X.toarray()  **#词频矩阵的结果**  print X  **# 词频矩阵****
**  **# 02、构建TFIDF权值****
**  transformer = TfidfTransformer()
  tfidf = transformer.fit_transform(X)  **# 计算tfidf值** tfidf.toarray()  


  **# 03、获取词袋模型中的关键词****
**  word = vectorizer.get_feature_names()
  weight = tfidf.toarray()# tfidf矩阵

###### ***\*2.1 jieba基于进行关键词抽取tfidf\****

**text = "线程是程序执"**

**keywords =jieba.analyse.extract_tags (text)**  

###### ***\*2.2\**** ***\*基于TextRank算法进行关键词抽取\****

**text = "线程是程序执"**

**keywords =jieba.analyse.textrank(text)**

### ***\*聚类\****

#### ***\*K-means聚类的迭代过程\****

· 1。随机选取k个文件生成k个聚类cluster,k个文件分别对应这k个聚类的聚类中心Mean(cluster) = k ;对应的操作为从W[i][j]中0～i的范围内选k行（每一行代表一个样本），分别生成k个聚类，并使得聚类的中心mean为该行。

· 2.对W[i][j]的每一行，分别计算它们与k个聚类中心的距离（通过欧氏距离）distance(i,k)。

· 3.对W[i][j]的每一行，分别计算它们最近的一个聚类中心的n(i) = ki。

· 4.判断W[i][j]的每一行所代表的样本是否属于聚类，若所有样本最近的n(i)聚类就是它们的目前所属的聚类则结束迭代，否则进行下一步。

· 5.根据n(i) ，将样本i加入到聚类k中，重新计算计算每个聚类中心（去聚类中各个样本的平均值），调到第2步。

#### ***\*主函数KMeans：\****

sklearn.cluster.KMeans(n_clusters=8,

   init='k-means++', 

  n_init=10, 

  max_iter=300, 

  tol=0.0001, 

  precompute_distances='auto', 

  verbose=0, 

  random_state=None, 

  copy_x=True, 

  n_jobs=1, 

  algorithm='auto'

  )

参数的意义：

· n_clusters:簇的个数，即你想聚成几类

· init: 初始簇中心的获取方法

· n_init: 获取初始簇中心的更迭次数，为了弥补初始质心的影响，算法默认会初始10个质心，实现算法，然后返回最好的结果。

· max_iter: 最大迭代次数（因为kmeans算法的实现需要迭代）

· tol: 容忍度，即kmeans运行准则收敛的条件

· precompute_distances：是否需要提前计算距离，这个参数会在空间和时间之间做权衡，如果是True 会把整个距离矩阵都放到内存中，auto 会默认在数据样本大于featurs*samples 的数量大于12e6 的时候False,False 时核心实现的方法是利用Cpython 来实现的

· verbose: 冗长模式（不太懂是啥意思，反正一般不去改默认值）

· random_state: 随机生成簇中心的状态条件。

· copy_x: 对是否修改数据的一个标记，如果True，即复制了就不会修改数据。bool 在scikit-learn 很多接口中都会有这个参数的，就是是否对输入数据继续copy 操作，以便不修改用户的输入数据。这个要理解Python 的内存机制才会比较清楚。

· n_jobs: 并行设置

· algorithm: kmeans的实现算法，有：’auto’, ‘full’, ‘elkan’, 其中 ‘full’表示用EM方式实现

虽然有很多参数，但是都已经给出了默认值。所以我们一般不需要去传入这些参数,参数的。可以根据实际需要来调用。

#### ***\*例子\*******\*KMeans分析的一些类如何调取与什么意义。\****

***\*import\**** numpy ***\*as\**** np
***\*from\**** sklearn.cluster ***\*import\**** KMeans
data = np.random.rand(100, 3) **#生成一个随机数据，样本大小为100, 特征数为3****
****
****#假如我要构造一个聚类数为3的聚类器****
**estimator = KMeans(n_clusters=3)**#构造聚类器****
**estimator.fit(tfidf_matrix)**#聚类****
**label_pred = estimator.labels_ **#获取聚类标签**

**clusters = km.labels_.tolist()****
**centroids = estimator.cluster_centers_ **#获取聚类中心****
**inertia = estimator.inertia_ **# 获取聚类准则的总和**

estimator初始化Kmeans聚类；estimator.fit聚类内容拟合； 
estimator.label_聚类标签，这是一种方式，还有一种是predict；estimator.cluster_centers_聚类中心均值向量矩阵 
estimator.inertia_代表聚类中心均值向量的总和

#### ***\*查看聚类结果\****

from __future__ import print_function

print("Top terms per cluster:")

\# 按离质心的距离排列聚类中心，由近到远

order_centroids = km.cluster_centers_.argsort()[:, ::-1] 

 

for i in range(num_clusters):

  print("Cluster %d words:" % i, end='')

  for ind in order_centroids[i, :6]: # 每个聚类选 6 个词

​    print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0], end=',')

  print("Cluster %d titles:" % i, end='')

  for title in frame.ix[i]['title'].values.tolist():

​    print(' %s,' % title, end='')

##### ***\*可视化文档聚类\****

http://python.jobbole.com/85481/

# ***\*1130kaggle竞赛\****

 

在一个不熟悉的城镇饥饿，并根据您的个人喜好，在恰当的时刻提供餐厅推荐。该建议附带信用卡提供商的附加折扣，适用于当地的拐角处！

目前，Elo是巴西最大的支付品牌之一，与商家建立了合作伙伴关系，以便为持卡人提供促销或折扣。但这些促销活动是否适用于消费者或商家？客户是否喜欢他们的体验？商家看到重复业务吗？个性化是关键。

Elo建立了机器学习模型，以了解客户生命周期中最重要的方面和偏好，从食品到购物。但到目前为止，它们都没有专门针对个人或个人资料量身定制。这是你进来的地方。

在本次竞赛中，Kagglers将通过揭示客户忠诚度中的信号，开发算法来识别并为个人提供最相关的机会。您的意见将改善客户的生活，帮助Elo减少不需要的活动，为客户创造合适的体验。

 

### ***\*Root Mean Squared Error (RMSE)\****

Submissions are scored on the root mean squared error. RMSE is defined as:

![img](D:/ruanjiandata/Typora_mdpic/wps55.jpg) 

![img](D:/ruanjiandata/Typora_mdpic/wps56.jpg) 

 

## ***\*201905腾讯赛\****

## ***\*s数据预处理\****

https://juejin.im/post/5b610970e51d4534b93f5408

**·** ***\*数据清洗\****：缺失值，异常值，一致性；

**·** ***\*特征编码\****：one-hot 和 label coding；

**·** ***\*特征分箱\****：等频，等距，聚类等；

**·** ***\*衍生变量\****：可解释性强，适合模型输入；

**·** ***\*特征选择\****：方差选择，卡方选择，正则化等；

### ***\*缺失：\*******\*df[df[\*******\*'Fare'\*******\*].isnull()]\****

4、通常***\*遇到缺值\****的情况，我们会有几种常见的处理方式

 

  如果缺值的样本占总数比例极高，我们可能就直接舍弃了，作为特征加入的话，可能反倒带入noise，影响最后的结果了

  如果缺值的***\*样本适中，\****而该属性非连续值特征属性(比如说类目属性)，那就把NaN作为一个新类别，加到类别特征中

  如果缺值的样本适中，而该属性为连续值特征属性，有时候我们会考虑给定一个step(比如这里的age，我们可以考虑每隔2/3岁为一个步长)，然后把它离散化，之后把NaN作为一个type加到属性类目中。

  有些情况下，缺失的值个数并不是特别多，那我们也可以试着根据已有的值，拟合一下数据，补充上。

​	

 

 

 

1直接删除（比例太高）  2相似特征替换（平均值，众数）

df.loc[(df['Pclass']==3)&(df['Age']>60)&(df['Sex']=='male')].Fare.mean() （用这几位乘客的***\*Fare平均值\****来填补。）

def fare_fill(data):

  fare = data.Fare[data.Pclass == 3].mean()

  print('fare fillvalues:', fare)

  data.Fare.replace(np.NaN, fare, inplace=True)

return data

![img](D:/ruanjiandata/Typora_mdpic/wps57.jpg) 

 

3大量缺失70%，或者忽略cabin特征

df['CabinCat'] = pd.Categorical.from_array(df.Cabin.fillna('0').apply(lambda x: x[0])).codes

pandas的 Categorical.from_array()用法。代码含义是用“0”替换Cabin缺失值，并将非缺失Cabin特征值提取出第一个值以进行分类，比如A114就是A，C345就是C，

***\*查看分类后分许情况：查看缺失数据和不缺失的区别\****

fig, ax = plt.subplots(figsize=(10,5)) 

sns.countplot(x='CabinCat', hue='Survived',data=df) 

plt.show()

 

2、年龄20%缺失

Age有20%缺失值，缺失值较多，大量删除会减少样本信息，由于它与Cabin不同，这里将利用其它特征进行预测填补Age，也就是拟合未知Age特征值，会在后续进行处理。

### ***\*归一化\****

如果大家了解逻辑回归与梯度下降的话，会知道***\*，\*******\*各属性值之间scale差距太大\*******\*，\*******\*将对收敛速度造成几万点伤害值！甚至不收敛，\****先用scikit-learn里面的***\*preprocessing\*******\*模\****块对这俩货做一个scaling，所谓scaling，其实就是将一些变化幅度较大的特征化到[-1,1]之内。

 

### ***\*利用GBDT模型构造新特征的方法\****

http://breezedeus.github.io/2014/11/19/breezedeus-feature-mining-gbdt.html

先用已有特征训练GBDT模型，然后利用GBDT模型学习到的树来构造新特征，最后把这些新特征加入原有特征一起训练模型

 

对了，已经有人利用这种方法赢得了Kaggle一个CTR预估比赛的冠军，代码可见https://github.com/guestwalk/kaggle-2014-criteo，里面有这种方法的具体实现

### ***\*特殊数据特征\****

#### ***\*数据一致性分析\*******\*，\*******\*不完全相信数据，\****

df.loc[df['surname']=='abbott',['Name','Sex','Age','SibSp','Parch']] 

首先寻找到了船上姓 abbott 的所有人，看孩子情况不符合逻辑换顺序

Kaggle泰坦尼克：家庭群体特征：一群人生还的概率应该是存在共性

#### ***\*衍生变量（自己生成特征\****

· ***\*Title\****：从Name中提取Title信息，因为同为男性，Mr.和 Master.的生还率是不一样的；

· ***\*TitleCat\****：映射并量化Title信息，虽然这个特征可能会与Sex有共线性，但是我们先衍生出来，后进行筛选；

· ***\*FamilySize\****：可视化分析部分看到SibSp和Parch分布相似，固将SibSp和Parch特征进行组合；

· ***\*NameLength\****：从Name特征衍生出Name的长度，因为有的国家名字越短代表越显贵；

· ***\*CabinCat\****：Cabin的分组信息； 高级衍生变量

 

![img](D:/ruanjiandata/Typora_mdpic/wps58.jpg) 

 

# ***\*重要：多值特征处理\****

 

后期上分得靠人群和时段特征了

那么，一种思路来了，比如一个用户喜欢两个球队，这个field的特征可能是[1,1,0,0,0,0,0.....0]，那么我们使用两次embedding lookup，再取个平均不就好了嘛。

嗯，这的确也许可能是一种思路吧，在tensorflow中，其实有一个函数能够实现我们上述的思路，那就是tf.nn.embedding_lookup_sparse。别着急，我们一步一步来实现多值离散特征的embedding处理过程。

# ***\*CTR预估 | 一文搞懂DeepFM的理论实践\****

\1. CTR 预估

CTR 预估 ***\*数据特点\****：

### ***\*数据处理:\*******\*输入中包含类别型和连续型数据。\*******\*类别型数据需要 one-hot, 连续型数据可以先离散化再 one-hot\*******\*，也可以直接保留原值\****

维度非常高、数据非常稀疏、特征按照 Field 分组
CTR 预估 重点 在于 学习组合特征。注意，组合特征包括二阶、三阶甚至更高阶的，阶数越高越复杂，越不容易学习。Google 的论文研究得出结论：高阶和低阶的组合特征都非常重要，同时学习到这两种组合特征的性能要比只考虑其中一种的性能要好

### ***\* \*******\*特征组合：\*******\*在\**** ***\*DeepFM\**** ***\*提出之前\*******\*，\****

已有 LR，FM，FFM，FNN，PNN（以及三种变体：IPNN,OPNN,PNN*）,Wide&Deep 模型，这些模型在 CTR 或者是推荐系统中被广泛使用。

LR 最大的缺点就是无法组合特征，依赖于人工的特征组合

FM 通过隐向量 latent vector 做内积来表示组合特征，从理论上解决了低阶和高阶组合特征提取的问题。但是实际应用中受限于计算复杂度，一般也就只考虑到 2 阶交叉特征

相继提出了使用 CNN 或 RNN 来做 CTR 预估的模型。但是，***\*CNN 模型的缺点是：偏向于学习相邻特征的组合特征。RNN 模型的缺点是：比较适用于有序列 (时序) 关系的数据\****

FNN 的提出，应该算是一次非常不错的尝试：先使用预先训练好的 FM，得到隐向量，然后作为 DNN 的输入来训练模型。缺点在于：受限于 FM 预训练的效果。

随后提出了 PNN，PNN 为了捕获高阶组合特征，在embedding layer和first hidden layer之间增加了一个product layer。根据 product layer 使用内积、外积、混合分别衍生出IPNN, OPNN, PNN*三种类型。

无论是 FNN 还是 PNN，他们都有一个绕不过去的缺点：***\*对于低阶的组合特征，学习到的比较少。\****而前面我们说过，低阶特征对于 CTR 也是非常重要的。

Google 意识到了这个问题，为了同时学习低阶和高阶组合特征，提出了 ***\*Wide&Deep 模型\****。它混合了一个 ***\*线性模型（Wide part）\****和 ***\*Deep 模型 (Deep part)\****。这两部分模型需要不同的输入，而 ***\*Wide part\**** 部分的输入，依旧 ***\*依赖人工特征工程\****



DEEPFM不需要训练人工特征，能同时学习低阶和高阶特征

![img](D:/ruanjiandata/Typora_mdpic/wps59.jpg) 

具体：https://xueqiu.com/9217191040/110099109

论文：DeepFM: A Factorization-Machine based Neural Network for CTR Prediction

比赛思路：

知乎live

# ***\*https://blog.csdn.net/ruibin_cao/article/details/89543489\****

# ***\*竞赛\****

## ***\*20190807文本分类-微博用户画像2016预测年龄性别地域\****

![img](D:/ruanjiandata/Typora_mdpic/wps60.jpg)![img](file:///C:\Users\sereny\AppData\Local\Temp\ksohtml16892\wps61.jpg)![img](file:///C:\Users\sereny\AppData\Local\Temp\ksohtml16892\wps62.jpg)![img](file:///C:\Users\sereny\AppData\Local\Temp\ksohtml16892\wps63.jpg)![img](file:///C:\Users\sereny\AppData\Local\Temp\ksohtml16892\wps64.jpg)![img](file:///C:\Users\sereny\AppData\Local\Temp\ksohtml16892\wps65.jpg) 

![img](D:/ruanjiandata/Typora_mdpic/wps66.jpg)![img](file:///C:\Users\sereny\AppData\Local\Temp\ksohtml16892\wps67.jpg)![img](file:///C:\Users\sereny\AppData\Local\Temp\ksohtml16892\wps68.jpg)![img](file:///C:\Users\sereny\AppData\Local\Temp\ksohtml16892\wps69.jpg) 

 

## ***\*0808搜狗用户画像\****

 

## ***\*用电\****

时间电话特征--一堆

计数比例特征

df['nunique_CITY_ORG_NO'] = df.CUST_NO.map(jobinfo.groupby('CUST_NO').CITY_ORG_NO.nunique())
df['ratio_CITY_ORG_NO'] =  df['counts_of_jobinfo'] / df['nunique_CITY_ORG_NO']

## ***\*阿里\****

![img](D:/ruanjiandata/Typora_mdpic/wps70.jpg) 

### ***\*Ctr\****

![img](D:/ruanjiandata/Typora_mdpic/wps71.jpg) 

## ***\*Jdata用户购买预测\****



行为（浏览或收藏）商品数/行为（浏览或收藏）商品种类/行为（浏览或收藏）天数/收藏商品数/收藏商品种类/有收藏行为的天数
地理信息：
用户下单过的地点数/用户订单数最大的地点编号
参数信息：
用户所购买商品price/para1/para2/para3的最大值最小值平均值中位数
用户花费：
用户的总花费
用户购买集中度：
用户购买集中度=购买的商品次数/购买的商品种类
用户商品忠诚度：
用户购买同一sku的最大次数
用户购买转化率：
用户购买转化率=用户购买的商品种类/用户有行为（浏览或收藏）的商品种类
日期特征：
购买的最小的day/最大的day/平均的day
近3个月/5个月 月首购买日期的最大、最小、平均、中位数
特征时间窗口：7天/14天/1月/3个月/5个月
品类维度：总体/(101,30)目标品类
最终特征维度：347
其他：未使用离散化/one-hot
————————————————
版权声明：本文为CSDN博主「长离未离1986」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/Francis1019/article/details/87899972

https://blog.csdn.net/yasin0/article/details/84404493

 

数据清洗,比如说:

· 去掉只有购买记录的用户(没有可用的历史浏览等记录来预测用户将来的购买意向)

· 去掉浏览量很大而购买量很少的用户(惰性用户或爬虫用户)

· 去掉最后5(7)天没有记录(交互)的商品和用户

### ***\*用户特征\****

user_id(用户id)

age(年龄)

sex(性别)

user_lv_cd(用户级别)

browse_num(浏览数)

addcart_num(加购数)

delcart_num(删购数)

click_num(点击数)

favor_num(收藏数)

buy_num(购买数)

buy_browse_ratio(购买浏览转化率)

buy_addcart_ratio(购买加购转化率)

buy_click_ratio(购买点击转化率)

 有购买行为的天数和月数

————————————————

Item特征

 

sku_id(商品id)

attr1,attr2,attr3(特征1，2，3)

cate(品类)

brand(品牌)

browse_num(浏览数)

addcart_num(加购数)

delcart_num(删购数)

click_num(点击数)

favor_num(收藏数)

buy_num(购买数)

buy_browse_ratio(购买浏览转化率)

buy_addcart_ratio(购买加购转化率)

buy_click_ratio(购买点击转化率)

buy_favor_ratio(购买收藏转化率)

comment_num(评论数),

has_bad_comment(是否有差评),

 

### ***\*时间特征\****

用户近期行为特征：

在上面针对用户进行累积特征提取的基础上，分别提取用户近一个月、近三天的特征，然后提取一个月内用户除去最近三天的行为占据一个月的行为的比重

***\*行为\****

用户对同类别下各种商品的行为:

用户对各个类别的各项行为操作统计

用户对各个类别操作行为统计占对所有类别操作行为统计的比重

 

***\*累积商品特征:\****

分时间段

针对商品的不同行为的购买转化率

均值、类别特征

 

分时间段下各个商品类别的购买转化率、均值

## ***\*18年购买\****

![img](D:/ruanjiandata/Typora_mdpic/wps72.jpg) 

![img](D:/ruanjiandata/Typora_mdpic/wps73.jpg) 

 

## ***\*2019.10.21\*******\*总结2018 CCF大数据竞赛-面向电信行业存量用户的智能套餐个性化匹配模型`\****

\#数据预处理

temp_test['3_total_fee']=temp_test['3_total_fee'].replace("\\N",-1)

temp_test['2_total_fee']=temp_test['2_total_fee'].astype(np.float64)

特征去除

​	df = df.drop(['former_complaint_num'], axis=1)

日期：

  sale['data_date'] = pd.to_datetime(sale['data_date'], format='%Y%m%d')

  sale['own_week'] = sale['data_date'].map(lambda x: (datetime.datetime(2018, 3, 16)-x).days//7)

isin：

  part = daily[daily['goods_id'].isin(sub['goods_id'].unique())]

其他

  df['goods_price'] = df['goods_price'].map(lambda x: x.replace(',', '') if type(x) == np.str else x)

  df['goods_price'] = pd.to_numeric(df['goods_price'])

​	

时间

  droped = daily.drop_duplicates(subset='goods_id')

  droped['open_date'] = droped.apply(lambda x: x['data_date'] - datetime.timedelta(x['onsale_days']), axis=1)

 

\#使用 pivot统计

  grouped_daily = daily.groupby(['goods_id', 'own_week'])['goods_click'].sum().reset_index()

  pivot_daily = grouped_daily.pivot(index='goods_id', columns='own_week', values='goods_click')

  new_columns = {}

  for i in list(pivot_daily.columns):

​    new_columns[i] = 'goods_click_' + str(i)

  pivot_daily.rename(columns=new_columns, inplace=True)

  pivot_daily.fillna(0, inplace=True)

  sub = pd.merge(sub, pivot_daily, on='goods_id', how='left')

 

统计

  raw_price = df.groupby('sku_id')['orginal_shop_price'].mean().reset_index()

  sub = pd.merge(sub, raw_price, on='sku_id', how='left')

--销量预测作用？？

 

  sub['smooth'] = sub.apply(lambda x: kalman_smooth(x), axis=1)

  for i in range(15):

​    sub['sales_smo_'+str(i)] = sub.apply(lambda x: x['smooth'][i], axis=1)

  print('------------kalman smooth-----------------')

 

  sub['sales_8'] = sub['sales_8'] * 1.1

  sub['sales_9'] = sub['sales_9'] * 1.2

def kalman_smooth(x):

 

  series = [x['sales_0'], x['sales_1'], x['sales_2'], x['sales_3'], x['sales_4'],

​           x['sales_5'], x['sales_6'], x['sales_7'], x['sales_8'], x['sales_9'], x['sales_10'], x['sales_11'],

​           x['sales_12'], x['sales_13'], x['sales_14']]

  kf = KalmanFilter(n_dim_obs=1, n_dim_state=1, initial_state_mean=series[0])

  state_means, state_covariance = kf.smooth(series)

  return state_means.ravel().tolist()

特征

pay_num_times_train=temp_train['pay_num']/temp_train['pay_times']

test.insert(1,column='pay_num_times',value=pay_num_times_test)

给权重

total_fee_weight_train=0.8*temp_train['1_total_fee']+0.5*temp_train['2_total_fee']\

​                 +0.2*temp_train['3_total_fee']+temp_train['4_total_fee']

选择不同列中的最小最大均值等

fee_min = list()

​    for row in range(df.shape[0]):

 

​      \#月度最低出账金额

​      fee_min_item = min(df.at[row, 'fee_1_month'],

​                df.at[row, 'fee_2_month'],

​                df.at[row, 'fee_3_month'],

​                df.at[row, 'fee_4_month'])

​		fee_min.append(fee_min_item)

df['fee_min'] = fee_min

 

 

 

\# lgb 参数

params={

  "learning_rate":0.2,

  "lambda_l1":0.1,

  "lambda_l2":0.2,

  "max_depth":6,  #6  本次修改了

  "objective":"multiclass",

  "num_class":11,

  "verbose":-1,

}

 

 

\# 自定义F1评价函数

def f1_score_vali(preds, data_vali):

  labels = data_vali.get_label()

  preds = np.argmax(preds.reshape(11, -1), axis=0)

  score_vali = f1_score(y_true=labels, y_pred=preds, average='weighted')  #改了下f1_score的计算方式

  return 'f1_score', score_vali, True

 

xx_score = []

cv_pred = []

 

\#先对模型进行调参

\#lgb模型,k折交叉验证，分类问题使用分层抽样

skf = StratifiedKFold(n_splits=n_splits,random_state=seed,shuffle=True)

import time

now=time.time()

for index,(train_index,test_index) in enumerate(skf.split(X,y)):

  X_train,X_valid,y_train,y_valid = X[train_index],X[test_index],y[train_index],y[test_index]

  train_data = lgb.Dataset(X_train, label=y_train)

  validation_data = lgb.Dataset(X_valid, label=y_valid)

  clf=lgb.train(params,train_data,num_boost_round=10000,valid_sets=[validation_data],early_stopping_rounds=200,feval=f1_score_vali,verbose_eval=1)

  plt.figure(figsize=(12,6))

  lgb.plot_importance(clf, max_num_features=40)

  plt.title("Featurertances")

  plt.show()

  feature_importance=pd.DataFrame({

​     'column': train_col,

​     'importance': clf.feature_importance(),

   }).to_csv('feature_importance_leaves57.csv',index=False)

  xx_pred = clf.predict(X_valid,num_iteration=clf.best_iteration)

  xx_pred = [np.argmax(x) for x in xx_pred]

  xx_score.append(f1_score(y_valid,xx_pred,average='macro'))

  y_test = clf.predict(X_test,num_iteration=clf.best_iteration)

  y_test = [np.argmax(x) for x in y_test]  #输出概率最大的那个

  if index == 0:

​    cv_pred = np.array(y_test).reshape(-1, 1)

  else:

​    cv_pred = np.hstack((cv_pred, np.array(y_test).reshape(-1, 1)))

​		

 

\# 其实这里已经对8折的数据做了一次投票，最后输出投票后的结果

submit = []

for line in cv_pred:

  submit.append(np.argmax(np.bincount(line)))

 

可以将无效值强制转换为NaN，如下所示：

df[['fee_1_month']]= pd.to_numeric(df.fee_1_month,, errors='coerce')

 

 

 

\#简单的投票融合

def vote(sublist=[]):

  result = pd.read_csv(sublist[0]).sort_values('user_id')

  print(result.columns)

  \# print(result['current_service'])

  label2current_service = dict(

​    zip(range(0, len(set(result['current_service']))), sorted(list(set(result['current_service'])))))

  current_service2label = dict(

​    zip(sorted(list(set(result['current_service']))), range(0, len(set(result['current_service'])))))

 

  for i in sublist:

​    temp = pd.read_csv(i).sort_values('user_id')

​    result[i] = temp.current_service.map(current_service2label)

  temp_df = result[sublist]

  \# 投票

  submit = []

  for line in temp_df.values:

​    submit.append(np.argmax(np.bincount(line)))

 

  result['current_service'] = submit

  result['current_service'] = result['current_service'].map(label2current_service)

  result.to_csv('result.csv',index=False)

  return result[['user_id', 'current_service']]

 

 

data = vote(sublist=["submit_75.csv","submit_92.csv","submit_88.csv"])

 

 

 

  def one_hot(df,cate_feats=[]):

​    """

​    对离散特征数据进行one-hot编码

​    :param df:

​    :param cate_feats:

​    :return:

​    """

​    for col in cate_feats:

​      one_hot = pd.get_dummies(df[col], prefix=col)

​      df = pd.concat([df, one_hot], axis=1)

​    df.drop(columns=cate_feats, inplace=True)

 

​    return df

 

  def Max_Min(self,df,const_feats=[]):

​    """

​    对连续特征数据进行归一化操作

​    :param df:

​    :param const_feats:

​    :return:

​    """

​    for col in const_feats:

​      scaler=MinMaxScaler()

​      df[col]=scaler.fit_transform(np.array(df[col].values.tolist()).reshape(-1,1))  #都是这样写的

​    return df

、xgb训练

  def __init__(self, mode):

​    self.params = {

​            'silent': 1,  #

​            'colsample_bytree': 0.8, #

​            'eval_metric': 'mlogloss',

​            'eta': 0.05,

​            'learning_rate': 0.1,

​            'njob': 8,

​            'min_child_weight': 1,

​            \# 'subsample': 0.8,

​            'seed': 0,  #0

​            'objective': 'multi:softmax',

​            'max_depth': 6,#原来是6

​            'gamma': 0.0,

​            'booster': 'gbtree',

​            'num_class': 11

​    }

 

 

  def train_model(self, X_train, y_train, X_valid, y_valid, X_test, result=None):

 

​    self.dtrain = xgb.DMatrix(X_train.drop(['user_id'], axis=1), y_train, missing=-99999.99)

​    if self.mode:

​      self.dvalid = xgb.DMatrix(X_valid.drop(['user_id'], axis=1), y_valid, missing=-99999.99)

​      watchlist = [(self.dtrain, 'train'), (self.dvalid, 'valid')]

​    else:

​      self.dtest = xgb.DMatrix(X_test.drop(['user_id'], axis=1), missing=-99999.99)

​      watchlist = [(self.dtrain, 'train')]

 

​    self.model = xgb.train(self.params,

​                self.dtrain,

​                evals=watchlist,

​                num_boost_round=1000,#1000

​                \# feval='f1_score',

​                early_stopping_rounds=100)

​    \#模型的保存

 

​    if self.mode:

​      self.valid_model(X_valid, y_valid)

​    else:

​      result = self.predict_model(X_test)

 

​    importance = self.model.get_fscore()

​    importance = sorted(importance.items(), key=operator.itemgetter(1))

​    pd.DataFrame(importance,columns=['feature','score']).to_csv('feature_importance.csv',index=False)

​    print(pd.DataFrame(importance, columns=['feature', 'score']))

​    return result

 

## ***\*阿里移动推荐算法\****

规则：#对于商品子集里的商品，18号加购物车且没买的，生成提交文件

 

## 乘用车ccf

```python
map映射获取
train_sales.drop_duplicates('province').set_index('province')['adcode']
data['bodyType'] = train_sales['province'].map(train_sales.drop_duplicates('province').set_index('province')['adcode'])
```

## 11.27海上风场SCADA数据缺失

数据缺失弥补 https://zhuanlan.zhihu.com/p/66410871 





## 0227硬盘数据

### 学习

```python
#2根据某一列排序 sort_values
test = test.sort_values(['serial_number','dt']，ascending=True)

#3去重
test = test.drop_duplicates().reset_index(drop=True)
#时间 #20180203 int变成字符串再转换成datetime 再用dt.days
    df['dt'] = df['dt'].apply(lambda x:''.join(str(x)[0:4] +'-'+ str(x)[4:6]  +'-'+ str(x)[6:]))  
    df['dt'] = pd.to_datetime(df['dt'])    
    #差异日期
df['diff_day'] = (df['fault_time'] - df['dt']).dt.days
      

```

### 标签处理，多重标签

```python
###分类有两种情况，同样的数据包含结果1和6那要把同时包含两个的分为一类
#使用.astype(str) 和groupby  .apply(lambda x :'|'.join(x)).reset_index()
tag['tag'] = tag['tag'].astype(str)
tag = tag.groupby(['serial_number','fault_time','model'])['tag'].apply(lambda x :'|'.join(x)).reset_index()
#重新构建映射 dict(zip( [nunique列举,1,2,3|5] ,range(数量)    )) 构建左边到右边的字典zip（列表，范围）
map_dict = dict(zip(tag['tag'].unique(), range(tag['tag'].nunique())))
tag['tag'] = tag['tag'].map(map_dict).fillna(-1).astype('int32') #再用map映射
```

### 大量数据

```python
#很多数据处理，读取，去重先，合并
train_2018_7 = joblib.load('./disk_sample_smart_log_2018_Q2/train_2018_7.jl.z')
serial_2017_7 = train_2017_7[['serial_number','dt']].sort_values('dt').drop_duplicates('serial_number')
serial = pd.concat((serial_2017_7,serial_2017_8),axis = 0)#上下合并
serial = serial.sort_values('dt').drop_duplicates('serial_number').reset_index(drop=True)
serial.columns = ['serial_number','dt_first']
train_x = train_x.merge(serial,how = 'left',on = 'serial_number')

gc.collect()
```

### 特征

```python
###硬盘的使用时常
train_x['days'] = (train_x['dt'] - train_x['dt_first']).dt.days

###当前时间与另一个model故障的时间差，
train_x['days_1'] = (train_x['dt'] - train_x['fault_time_1']).dt.days
train_x.loc[train_x.days_1 <= 0,'tag'] = None
train_x.loc[train_x.days_1 <= 0,'days_1'] = None

###同一硬盘第一次使用到开始故障的时间
train_x['days_2'] = (train_x['fault_time_1'] - train_x['dt_first']).dt.days
train_x.loc[train_x.fault_time_1 >= train_x.dt,'days_2'] = None

train_x['serial_number'] = train_x['serial_number'].apply(lambda x:int(x.split('_')[1]))
```

### 分类

```python
clf = LGBMClassifier(   
    learning_rate=0.001,
    n_estimators=10000,#预测改为100
    num_leaves=127,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=2019,
    is_unbalenced = 'True',#不平衡  预测的时候  去掉这句
    metric=None
)

clf.fit(
    train_x, train_y,
    eval_set=[(val_x, val_y)],
    eval_metric='auc',
    early_stopping_rounds=50,
    verbose=100
)

#预测
print(train_df.shape,test_x.shape)
clf.fit(
    train_df, labels,
    eval_set=[(train_df, labels)],
    eval_metric='auc',
    early_stopping_rounds=10,
    verbose=10
)

sub['p'] = clf.predict_proba(test_x)[:,1]
sub['label'] = sub['p'].rank()
sub['label']= (sub['label']>=sub.shape[0] * 0.996).astype(int)
submit = sub.loc[sub.label == 1]
###这里我取故障率最大的一天进行提交，线上测了几个阈值，100个左右好像比较好。。。。
submit = submit.sort_values('p',ascending=False)
submit = submit.drop_duplicates(['serial_number','model'])
submit[['manufacturer','model','serial_number','dt']].to_csv("sub.csv",index=False,header = None)
submit.shape
```

本题是给定一段连续采集(天粒度)的硬盘状态监控数据以及故障标签数据，参赛者需要自己提出方案，按天粒度判断每块硬盘是否会在未来30日内发生故障。例如，可以将预测故障问题转化为传统的二分类问题，通过分类模型来判断哪些硬盘会坏；因为七月份的数据不知道label，所以线下验证的时候也必须模拟线上隔一个月用四月份训练，六月份验证。

#这里只是二分类，一些其他思路 可以试试多分类，label分为1，2....30天，超过30，也可也试试回归做。



```
file='movies.dat'
with open(file, encoding='ISO-8859-1') as fp:
    for line in fp:
        print(line)
```

## EDA

### 缺失

```python
# 按列计算缺失值的函数
def missing_values_table(df):
        #计算总的缺失值
        mis_val = df.isnull().sum()
        
        #计算缺失值的百分比
        mis_val_percent = 100 * df.isnull().sum() / len(df)
   
        #把结果制成表格
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        #对列重命名，第一列：Missing Values，第二列：% of Total Values
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        #根据百分比对表格进行降序排列
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        #打印总结信息：总的列数，有数据缺失的列数
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
       
        # 返回带有缺失值信息的dataframe
        return mis_val_table_ren_columns
```



### 去除异常值

```python
# 计算第一和第三 四分数
first_quartile = data['Site EUI (kBtu/ft²)'].describe()['25%']
third_quartile = data['Site EUI (kBtu/ft²)'].describe()['75%']

# 第四分位数范围
iqr = third_quartile -first_quartile

# 去除异常值

data = data [(data['Site EUI (kBtu/ft²)']>(first_quartile - 3*iqr)) & (data['Site EUI (kBtu/ft²)']<(third_quartile + 3*iqr))]
```

### 相关性

```python
# 找到所有相关性并排序
correlations_data = data.corr()['score'].sort_values()
print(correlations_data.head(15),'\n')# 打印最负相关性
print(correlations_data.tail(15))# 打印最正相关性
'''在下面的代码中，我们采用数值变量的对数和平方根变换，对两个选定的分类变量（建筑类型和自治市镇）进行one-hot编码，
计算所有特征与得分之间的相关性，并显示前15个最正和前15个最负相关。'''

# 选择数字列
numeric_subset = data.select_dtypes('number') #选择类别
# 使用数字列的平方根和对数创建列
for col in numeric_subset.columns:
    if col == 'score':next# 跳过the Energy Star Score 这一列
    else:
        numeric_subset['sqrt_ '+ col] = np.sqrt(numeric_subset[col])
        numeric_subset['log_' + col] = np.log(numeric_subset[col])

# 选择分类列  one hot 编码     
categorical_subset = data[['Borough','Largest Property Use Type']]
categorical_subset = pd.get_dummies(categorical_subset)
# 使用concat对两个数据帧进行拼接，确保使用axis = 1来执行列绑定
features = pd.concat([numeric_subset,categorical_subset],axis =1)

features = features.dropna(subset = ['score'])
correlations = features.corr()['score'].dropna().sort_values()
```

###  画图

两个变量之间的关系，我们使用散点图 

```python
figsize(8,8)
plt.hist(data['score'].dropna(),bins = 100,edgecolor = 'k') #单变量直方图
sns.kdeplot(subset['score'].dropna(),label = b_type) #密度图
'''可以在几个不同的变量之间建立Pairs Plot。 Pairs Plot是一次检查多个变量的好方法，因为它显示了对角线上的变量对和单个变量直方图之间的散点图。
使用seaborn PairGrid函数，我们可以将不同的图绘制到网格的三个方面:
1上三角显示散点图2对角线将显示直方图
3下三角形将显示两个变量之间的相关系数和两个变量的2-D核密度估计。'''
# 提取要绘制的列
data = features[['score','Site', 'Weather']]
data = data.replace({np.inf: np.nan,-np.inf:np.nan})# 把 inf 换成 nan
data=data.dropna()
# 重命名columns
data = data.rename(columns = {'Sit': 'Si','Weather':'we'})

grid = sns.PairGrid(data = plot_data,size=3)# 创建 pairgrid 对象
grid.map_upper(plt.scatter,color = 'red', alpha =0.6)# 上三角是散点图
grid.map_diag(plt.hist,color ='red',edgecolor = 'black')# 对角线是直方图
grid.map_lower(corr_func);# 下三角是相关系数和二维核密度图
grid.map_lower(sns.kdeplot,cmap = plt.cm.Reds)
plt.suptitle('Pairs Plot of Energy Data', size = 36, y = 1.02);         
```



###  消除共线特征 

```python
def remove_collinear_features(x, threshold):
    y = x['score'] # 不要删除得分
    x = x.drop(columns = ['score'])
  
    # 计算相关性矩阵
    corr_matrix = x.corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    # 迭代相关性矩阵并比较相关性
    for i in iters:
        for j in range(i):
            item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
            col = item.columns
            row = item.index
            val = abs(item.values)
            # 如果相关性超过阈值
            if val >= threshold:drop_cols.append(col.values[0])

    # 删除每对相关列中的一个
    drops = set(drop_cols)
    x = x.drop(columns = drops)
    x = x.drop(columns = ['Weather' , 'Water U'])

    x['score'] = y# 将得分添加回数据     
    return x


features = remove_collinear_features(features, 0.6)# 删除大于指定相关系数的共线特征
features  = features.dropna(axis=1, how = 'all')# 删除所有 na 值的列
```

### 训练前数据处理

```python
#归一化
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(X)#训练
X = scaler.transform(X)# 转换训练数据和测试数据
X_test = scaler.transform(X_test)

# Convert y to one-dimensional array (vector)
y = np.array(train_labels).reshape((-1, ))
y_test = np.array(test_labels).reshape((-1, ))
```

### 模型

```python
#参考：https://github.com/DeqianBai/Your-first-machine-learning-Project---End-to-End-in-Python
# 线性回归
lr = LinearRegression()
svm = SVR(C=1000,gamma =0.1)
random_forest = RandomForestRegressor(random_state = 60)
gradient_boosted = GradientBoostingRegressor(random_state=60)
knn = KNeighborsRegressor(n_neighbors=10)


model = GradientBoostingRegressor(random_state = 42)
random_cv = RandomizedSearchCV(estimator=model,
                               param_distributions=hyperparameter_grid,
                               cv=4, n_iter=25, 
                               scoring = 'neg_mean_absolute_error',
                               n_jobs = -1, verbose = 1, 
                               return_train_score = True,
                               random_state=42)
random_cv.fit(X, y)
random_cv.best_estimator_

```

## 0324天池-安泰杯跨境电商智能算法大赛分享

稀疏+冷启动

 怎样利用已成熟国家A的稠密用户数据和待成熟国家B的稀疏用户数据，训练出的正确模型对于国家B的用户有很大价值。 

 用户下一个可能交互商品，选手们可以提交预测的TOP30商品列表，排序越靠前命中得分越高。 

1构建两个模型：

- 历史交互商品模型

- **关联商品模型**

   商品相似性 ， 假设用户越近交互的两个商品相似性越高 

  

模型融合：历史商品模型排序第4的商品召回率仅有1.5，而关联模型排序第一位召回率为3.1。 

取历史模型前三个和关联模型后27个组成30



物品相似度：m1，m2在一个用户一起出现的次数和/（根号（a出现次数*b出现次数））

### CIKM



视频推荐最近没做完那个

# 0321-《零基础入门数据挖掘》

## 函数

```
 ## 定义了一个统计函数，方便后续信息统计 
def Sta_inf(data):
    print('_min',np.min(data))
    print('_max:',np.max(data))
    print('_mean',np.mean(data))
    print('_ptp',np.ptp(data))
    print('_std',np.std(data))
    print('_var',np.var(data))
```



## 特征

```python
#提取数值类
numerical_cols = Train_data.select_dtypes(exclude = 'object').columns
categorical_cols = Train_data.select_dtypes(include = 'object').columns
#提取特征列
fea_cols = [col for col in num_cols if col not in ['SaleID','creatDate']
            
## 绘制标签的统计图，查看标签分布
plt.hist(Y_data)
plt.show()
plt.close(
```

## 模型

### 五折交叉

```python
## xgb-Model 回归
#重要参数+模型定义
xgr = xgb.XGBRegressor(n_estimators=120, learning_rate=0.1, gamma=0, subsample=0.8, colsample_bytree=0.9, max_depth=7) #,objective='reg:squarederror'
scores_train = []
scores = []
## 5折交叉验证方式StratifiedKFold
sk=StratifiedKFold(n_splits=5,shuffle=True,random_state=0) for train_ind,val_ind in sk.split(X_data,Y_data):
    train_x=X_data.iloc[train_ind].values
    train_y=Y_data.iloc[train_ind]
    val_x=X_data.iloc[val_ind].values
    val_y=Y_data.iloc[val_ind]

    xgr.fit(train_x,train_y)
    pred_train_xgb=xgr.predict(train_x)
    pred_xgb=xgr.predict(val_x)

    score_train = mean_absolute_error(train_y,pred_train_xgb)
    scores_train.append(score_train)
    score = mean_absolute_error(val_y,pred_xgb)
    scores.append(score)
print('Train mae:',np.mean(score_train))
print('Val mae',np.mean(scores))
```

### 2定义分开

```python
def build_model_xgb(x_train,y_train):
    model = xgb.XGBRegressor(n_estimators=150, learning_rate=0.1, gamma=0, subsample=0.8, colsample_bytree=0.9, max_depth=7) #, objective ='reg:squarederror'
    model.fit(x_train, y_train)
    return model
def build_model_lgb(x_train,y_train):
    estimator = lgb.LGBMRegressor(num_leaves=127,n_estimators = 150)
    param_grid = { 'learning_rate': [0.01, 0.05, 0.1, 0.2], }
    gbm = GridSearchCV(estimator, param_grid)
    gbm.fit(x_train, y_train)
    return gbm

x_train,x_val,y_train,y_val = train_test_split(X_data,Y_data,test_size=0.3)
print('Predict lgb...')
model_lgb_pre = build_model_lgb(X_data,Y_data)
subA_lgb = model_lgb_pre.predict(X_test)
print('Sta of Predict lgb:')#看结果
Sta_inf(subA_lgb)
```

## 模型融合

```python
 #这里我们采取了简单的加权融合的方式
val_Weighted = (1-MAE_lgb/(MAE_xgb+MAE_lgb))*val_lgb+(1-MAE_xgb/(MAE_xgb+MAE_lgb))*val_xgb

val_Weighted[val_Weighted<0]=10 # 由于我们发现预测的最小值有负数，而真实情况下，price为负是不存在的，
print('MAE of val with Weighted ensemble:',mean_absolute_error(y_val,val_Weighted))

sub = pd.DataFrame()
sub['SaleID'] = X_test.SaleID
sub['price'] = sub_Weighted
sub.to_csv('./sub_Weighted.csv',index=False)
```

## 评估指标

### 分类

```
## Precision,Recall,F1-score
from sklearn import metrics
y_pred = [0, 1, 0, 0]
y_true = [0, 1, 0, 1]
print('Precision',metrics.precision_score(y_true, y_pred))
print('Recall',metrics.recall_score(y_true, y_pred))
print('F1-score:',metrics.f1_score(y_true, y_pred))

## AUC
import numpy as np
from sklearn.metrics import roc_auc_score
y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])
print('AUC socre:',roc_auc_score(y_true, y_scores))
```

### 回归评估

```python
# coding=utf-8
import numpy as np
from sklearn import metrics
# MAPE需要自己实现 def mape(y_true, y_pred):
 return np.mean(np.abs((y_pred - y_true) / y_true))
y_true = np.array([1.0, 5.0, 4.0, 3.0, 2.0, 5.0, -3.0])
y_pred = np.array([1.0, 4.5, 3.8, 3.2, 3.0, 4.8, -2.2])
# MSE
print('MSE:',metrics.mean_squared_error(y_true, y_pred))
# RMSE
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_true, y_pred)))
# MAE
print('MAE:',metrics.mean_absolute_error(y_true, y_pred))
# MAPE
print('MAPE:',mape(y_true, y_pred))

## R2-score
from sklearn.metrics import r2_score
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
print('R2-score:',r2_score(y_true, y_pred))
```



## 画图



```python
## 1) 总体分布概况（无界约翰逊分布等） import scipy.stats as st
y = Train_data['price']
plt.figure(1); plt.title('Johnson SU')
sns.distplot(y, kde=False, fit=st.johnsonsu)
plt.figure(2); plt.title('Normal')
sns.distplot(y, kde=False, fit=st.norm)
## 2) 查看skewness and kurtosis
sns.distplot(Train_data['price']);
print("Skewness: %f" % Train_data['price'].skew())
print("Kurtosis: %f" % Train_data['price'].kurt())

Train_data.skew(), Train_data.kurt()
sns.distplot(Train_data.skew(),color='blue',axlabel ='Skewness')
sns.distplot(Train_data.kurt(),color='blue',axlabel ='Kurtness')

## 3) 查看预测值的具体频数
plt.hist(Train_data['price'], orientation = 'vertical',histtype = 'bar', color ='red')
plt.show()


## 4) 数字特征相互之间的关系可视化
sns.set()
columns = ['price', 'v_12', 'v_8' , 'v_0', 'power', 'v_5', 'v_2', 'v_6', 'v_1', 'v_14']
sns.pairplot(Train_data[columns],size = 2 ,kind ='scatter',diag_kind='kde')
plt.show()
## 5) 多变量互相回归关系可视化
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8), (ax9, ax10)) = plt.subplots(nrows=5, ncols=2, 
# ['v_12', 'v_8' , 'v_0', 'power', 'v_5', 'v_2', 'v_6', 'v_1', 'v_14']
v_12_scatter_plot = pd.concat([Y_train,Train_data['v_12']],axis = 1)
sns.regplot(x='v_12',y = 'price', data = v_12_scatter_plot,scatter= True, fit_reg=True, ax=ax1)
v_8_scatter_plot = pd.concat([Y_train,Train_data['v_8']],axis = 1)
sns.regplot(x='v_8',y = 'price',data = v_8_scatter_plot,scatter= True, fit_reg=True, ax=ax2)
  
```

### 类别特征

```python
                                                                               ## 1) unique分布 for fea in categorical_features:
 print(Train_data[fea].nunique())
                                                                                  ## 2) 类别特征箱形图可视化
# 因为 name和 regionCode的类别太稀疏了，这里我们把不稀疏的几类画一下
categorical_features = ['model',
'brand',
'bodyType',
'fuelType',
'gearbox',
'notRepairedDamage'] 
for c in categorical_features:
 	Train_data[c] = Train_data[c].astype('category')
 	if Train_data[c].isnull().any():
 		Train_data[c] = Train_data[c].cat.add_categories(['MISSING'])
 		Train_data[c] = Train_data[c].fillna('MISSING') 
def boxplot(x, y, **kwargs):
 	sns.boxplot(x=x, y=y)
 	x=plt.xticks(rotation=90) 
f = pd.melt(Train_data, id_vars=['price'], value_vars=categorical_features)
g = sns.FacetGrid(f, col="variable", col_wrap=2, sharex=False, sharey=False, size=5) 
g = g.map(boxplot, "value", "price")
## 4) 类别特征的柱形图可视化 def bar_plot(x, y, **kwargs):
 sns.barplot(x=x, y=y)
 x=plt.xticks(rotation=90) f = pd.melt(Train_data, id_vars=['price'], value_vars=categorical_features) g = sns.FacetGrid(f, col="variable", col_wrap=2, sharex=False, sharey=False, size=5) g = g.map(bar_plot, "value", "price")
## 5) 类别特征的每个类别频数可视化(count_plot)
def count_plot(x, **kwargs):
 sns.countplot(x=x)
 x=plt.xticks(rotation=90) f = pd.melt(Train_data, value_vars=categorical_features) g = sns.FacetGrid(f, col="variable", col_wrap=2, sharex=False, sharey=False, size=5) g = g.map(count_plot, "value")
```

的

```
## 1) 相关性分析
price_numeric = Train_data[numeric_features]
correlation = price_numeric.corr()
print(correlation['price'].sort_values(ascending = False),'\n')

f , ax = plt.subplots(figsize = (7, 7))
plt.title('Correlation of Numeric Features with Price',y=1,size=16)
sns.heatmap(correlation,square = True, vmax=0.8)
```

## 其他看

![1584857817466](C:\Users\sereny\AppData\Roaming\Typora\typora-user-images\1584857817466.png)

![1584857834849](C:\Users\sereny\AppData\Roaming\Typora\typora-user-images\1584857834849.png)

![1584857862157](C:\Users\sereny\AppData\Roaming\Typora\typora-user-images\1584857862157.png)

