# 最近

提交

cd /home/pc/UserData/cqj/project/temp/11kagg/code

 kaggle competitions submit -c ashrae-energy-prediction -f 1130-21submission.csv -m "Message" 

 kill -s 9 24552

 watch -n 3 nvidia-smi









# 比赛代码整理

## 理论

### 理论3

0513特征工程
特征交叉：两个数值变量的加减乘除、FM和FFM模型。

时间序列分析：滞后特征（又称lag特征，当前时间点之前的信息）、滑动窗口统计特征（如回归问题中计算前n个值的均值，分类问题中前n个值中每个类别的分布）。

特征组合GBDT

思考了一批结合业务的特征，包括行为转移、周期、偏好等。规则。。。特征的提取重要。。。我们时间不多了。。。

上几个简单的特征，比如，在分隔日期前brand的7天销量，或者用户对该brand的总点击次数等。详细部分另谈

### 理论1



一、数据预处理
       由于所给数据“脏”数据比较多，所以首先需要做大量的预处理，包括：

1.处理类型错误的数据。如‘A25’列中数据应该为数值型却混入了一个‘1900/3/10 0:00’时间数据。

2.处理时间数据的异常。

3.处理明显的数值异常。

4.使用中位数填充缺失值

二、特征工程
       特征工程是决定一个比赛的关键，因此在特征上我们做了大量处理，包括：

1.对连续型特征进行离散化编码。由于数据集中的数据包含大量异常值，对连续的特征进行离散化编码可以一定程度上降低异常值带来的影响。离散化的方法包括：

  （1）对于正态性的特征，根据分位数离散化。

  （2）对于有大量偏离中心的数据（异常值），使用众数进行离散化编码，即等于众数，大于众数，小于众数。

  （3）对于时间数据根据时段编码，即分成早上、下午、晚间

  （4）根据数据的分布进行离散化，如对id的离散化。

2.构造组合特征。将基础数据通过加减乘除进行两两组合，从中选择重要性比较高或者有意义的特征加入模型训练。

3.离散特征对target（‘收率’）聚合求统计值。使用离散化后的特征（‘B14、B12’等）对收率进行聚合操作，然后求均值、最大值、最小值和数量。

4.rank排序特征。对连续的特征进行排序可以得到排序特征，排序特征对于异常数据具有较强的鲁棒性，使得模型更加稳定，降低过拟合的风险。

5.DBSCAN聚类特征。DBSCAN聚类的主要作用也是可以标识离群点。

6.业务相关特征。主要是仿照收率的定义，使用b14去除以其他原料的和。

三、特征选择
1.删除常量特征

2.删除缺失值多，变化小的特征

### 理论2

看出单模型方面xgboost的表现最佳，因此我们一直使用它作为基础模型更新
迭代不同的特征版本。
bagging采用5个xgboost（设置不同的随机种子）进行投票；stacking设置
LR、RF、ExtraTreesClassifier、xgboost作为基分类器，LR作为二级分类器，

3. 特征筛选
   (1) 删掉缺失值太多的特征
   (2) 删掉无意义特征
   (3) 删掉重复特征
    可以直接用drop_duplicates删掉的并不多，但在上千个特征的采集过程中，一般都会有些显性和隐性的重复，比如：＂子宫＂，＂输卵管＂这些关键字与＂血压＂强相关，原因是＂性别＂和＂血压＂强相关．所以这样的关键字提取很多也没什么用．
    此时可以分析特征间的相关性corr()．corr()只能检测线性相关，所以绝对值如果低并不说明无关，但如果值高，一定相关．那如果两个特征相关性为1，就可以去掉其中一个了？也不行！计算相关性时一般忽略掉空值，这个也要考虑在内．

(4) 单变量特征筛选
 Sklearn也提供了一些特征筛选的方法：sklearn.feature_selection.*，比如SelectKBest可以支持卡方Chi2．Sklearn对数据有一些要求，比如有的要求非空．
 另外用于检测单变量与目标相关性的还有互信息，皮尔森相关系数，最大信息系数等等．
 残差分析也是一种常用方法．简单地说就是：计算当X位于不同取值范围时，Y均值的变化．从而检查X是否对Y有影响，它能检测到一些非线性相关的情况．

(5) 基于模型的特征选择
 基于模型的特征选择，指用模型训练完成之后，通过模型输出的特征重要性feature_importances_，选取其前N个特征．
这个是训练之后才能得到的数据，都训练完了，时间都花掉了，才选出特征有什么用呢？这是个先有鸡还是先有蛋问题．
 首先，用这种方法选出的重要特征可以更好地解释模型和数据．而且它是多特征组合的结果，不只考虑了局部特征．
 另外，还可以用它筛掉一些干涉性特征，比如做10折交叉验证，其中9次都认为某特征不重要，其余那一次，很可能是干扰，也算一种统计吧．
 一个小技巧是，在提取特征的过程中，可以边提取边训练（设置参数，少量实例，少量特征以快速训练）至少能粗分出某个新特征重要与否，是否应该保留．

(6) 其它方法
 还有主成分分析PCA等方法．方差分析 ANOVA，信息值分析IV等等．
 上述都是从特征角度筛选，还有从实例角度筛选（不限于此题），比如分析广告和购买时，那些从来不买东西，从来不点广告的人，可能就需要另外处理，或者在回归前先做个分类，计算一些统计特征（有点跑题了



## ***\*0前面用import\****

```python
'''
import os
os.chdir("F:/project/python/3competition/ccf/lingjian/code")#导入数据
import sys
sys.path.append("F:/project/python/3competition/ccf/lingjian/code") #文件关联
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error as mse

# -*- coding: utf-8 -*-
pd.set_option('display.max_columns',None)
path = '../data/'
train = pd.read_csv(path + '/first_round_training_data.csv')
'''

import numpy as np
import pandas as pd
import gc
from tqdm import tqdm, tqdm_notebook
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
import datetime
import time
import lightgbm as lgb
import xgboost as xgb
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 500)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

%matplotlib inline
plt.style.use('seaborn')

#11.19
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
sns.set(font='SimHei')  # 解决Seaborn中文显示问题
sns.set(font_scale=1)

warnings.simplefilter('ignore')
matplotlib.rcParams['figure.dpi'] = 100
sns.set()
```

## 常用找

#### EDA常用

```python
df.loc[(df['id']==800)&(df.meter==0)]['mm'].value_counts().sort_index()
df['site_id'].value_counts().sort_index() #sort_values()
list(train_df)
#根据单一id看一下每个特征分布
#看数据
for t in [train_sales,train_search,train_user,evaluation_public,submit_example]:
    print(30*'#'+30*'#')
    print(t[0:10])
#变量定义
df['series_{}_{}_{}_{}_sum'.format( col,group,start,end)]
data.loc[data['用户年龄']==0,'用户年龄'] = None
#size
vid_tabid_group = part1_2.groupby(['vid','table_id']).size().reset_index()

df.loc[(df.Month==1)&(df.y==2018), 'label'] = sub['label'].values
```

#### 时间

ts = time.time()

time.time() - ts

```python
#时间为文件名
import time
now = time.strftime("%m%d-%H",time.localtime(time.time())) 
#print time.strftime("%b %d %Y %H:%M:%S", time.gmtime(t))
```

#### 删除特征

**try**:
  **del** data_df[f+'_cnts']
**except**:
  **pass**

#### ***\*分组后融合\****

```python
mo_pro_label=data.loc[(data.regYear==2017)].groupby(['model','province'])['label'].mean().rename('label_perModPro').reset_index()
data=pd.merge(data,mo_pro_label,'left',on=['model','province'])
tmp  =df.groupby(['stationID'])['deviceID'].nunique().reset_index(name='nuni_deviceID_of_stationID')
result  = result.merge(tmp, on=['stationID'], how='left')
```



#### ***\*Pandas数据处理：\****

```python
data = train.append(test).reset_index(drop=True) #drop=True取掉index
pd.set_option('display.max_columns',None)#不用省略号
print(train_monthly.isnull().sum())## 缺失值常用
```

 

#### ***\*用来存储的array\****

oof = np.zeros((X_train.shape[0],4))

### ***\*批量删除特征\****

```python
# for fea in ['label_shifted1', 'label_shifted2', 'label_shifted3']:
\#   del data[fea]
```



### ***\*定位：\****

```python
data_df.loc[(data.regMonth>=1)&(data.regYear==2018), ['id','label']]=
data_df.loc[(data.regMonth>=1)&(data.regYear==2018), ['id','label']].values
```



### ***\*变量命名\****

```python
df['model_adcode_mt_{}_{}'.format(col,i)]
df['label_lastmonth'+str(i)']
df.loc[(df.LE_province==i)&(df.model==j)&(df.month_no==24), ['label_lastmonth1']]= df.loc[(df.LE_province==i)&(df.model==j)&(df.month_no==24), ['label']].values
```



### ***\*分组查看\****

data.sort_values(['province','model'])[['label_increase','label_lastmonth','label','month_no','province','model']] #查看

 ***\*步骤\****

 

![img](file:///C:\Users\sereny\AppData\Local\Temp\ksohtml103712\wps1.jpg) 

## 1EDA

思考问题
数据应该怎么清洗和处理才是合理的？
根据数据的类型可以挖掘怎样的特征？
数据中的哪些特征会对标签的预测有帮助？
统计分析
	数值型
		统计量
			min, max, mean, medium, std:观察Label是否均衡
			相关系数:找到高相关的特征以及特征之间的冗余度

文本变量:词频(TF)，TF-IDF，文本长度等

### 整理每次用

```
根据不同的变量探索目标变量
```

```python

#1
print('df data:', df.info(),'\n',df.describe(),'\n',(df.count() / len(df)))
df.head()
#2
df=train.copy（

#其他value_counts
for fea in ['province','city']:
    print(fea+'_unique',len(df[fea].unique()))
    print(df[fea].value_counts().sort_index())
#数量绘图
    
df[df['floor_count'].isnull()].shape

df[df['deviceid']=='1672767a5a645e6cf795f9965238b774']
data = train.append(test).reset_index(drop=True)
label='target'
test=df[df[label].isnull()]  
train=df[df[label].notnull()] #tr_index = ~data['target'].isnull()


```

### 每次可能用

```python
#查看交集
temp_df = test_df[~test_df['id'].isin(train_df['id'])]
print('No intersection:', len(temp_df))


```



### 训练集和数据集区别？todo

### 1文件读取

#### 1取数据

```python
sep='$',encoding='utf-8') header
data_part1=pd.read_csv('8.txt',sep='$',encoding='utf-8')

ad_static=pd.re

ad_csv('../data/ad_static_feature.out',sep='\t',header=**None**) # (735911, 7)
ad_static_col=['adId','adCreatetime','adAccountid','itemId','itemType','adSize','adIndustryid']
ad_static.columns=ad_static_col

pandas.read_csv（file,sep="\t",header=标题行,names=列名/None，prefix，engine=c（更快）/python（更完善）
，nrows =100，iterator =False（True逐块处理文件））
```



#### 2压缩



```python
train_df = pd.read_csv('../data/train.csv')
train_df=reduce_mem_usage(train_df)
train_df.to_pickle('../data/train_df.pkl')
#或者
def read_to_pikle(filename):
    df=pd.read_csv('../data/'+filename+'.csv')
    df=reduce_mem_usage(df)
    df.to_pickle('../data/'+filename+'.pkl')
    return df
train_df=read_to_pikle('train')

```



### 1画图

```python
import matplotlib
import matplotlib.pyplot as plt
在Jupyter Notebook页面内显示绘图


#matplotlib.use('agg') #在PyCharm中不显示绘图
#设置坐标轴名称
plt.xlabel('x-label-English')
plt.ylabel('y-label-English')
# # 设置标题
plt.title('title',fontsize=20,verticalalignment='bottom') # 设置字体大小，垂直底部对齐
# 绘图并保存
plt.scatter(df['id'], df['Adpctr'], alpha=0.6)  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
#plt.savefig('./Adpctr.png')#保存图片
```

df['depth_mean'] =df.groupby(['DT_M'])['meter_reading'].transform('mean') #撒点太多看均值

### ***\*数据探索\****



### ***\*读取文件\*******\* \*******\*data_list\**** ***\*=\**** ***\*os\*******\*.\*******\*listdir(path\*******\*+\*******\*'/Metro_train/'\*******\*)\****

数据联合

features, test_features = **features**.align(**test_features**, join='inner', axis=1)

####  hdf

```python
df.to_hdf(path + '/a.h5', 'df', mode="w")
df = pd.read_hdf(path + '/df.h5')
```

####  pickle

```python
#pandas数据pickling比保存和读取csv文件要快2-3倍（lz测试不准，差不多这么多）。

train_df.to_pickle( 'middlewares/train_df')
train_df= pd.read_pickle( 'middlewares/train_df')

train_df.to_pickle(os.path.join(CWD, 'middlewares/train_df'))
train_df= pd.read_pickle(os.path.join(CWD, 'middlewares/train_df'))
#不过lz测试了一下，还是直接pickle比较快，比pd.read_pickle快2倍左右。
pickle.dump(ltu_df, open(os.path.join(CWD, 'middlewares/ltu_df.pkl'), 'wb'))
ltu_df = pickle.load(open(os.path.join(CWD, 'middlewares/ltu_df.pkl'), 'rb'))
```



 

### ***\*10.24-1\*******\*#查看训练测试集合的数据分布情况\****

作用：# 删除特征"V5","V9","V11","V17","V22","V28"，训练集和测试集分布不均

\#kdeplot核密度估计(kernel density estimation)是在概率论中用来估计未知的密度函数，属于非参数检验方法之一。通过核密度估计图可以比较直观的看出数据样本本身的分布特征

fea=para_feat1+attr_feat

para_feat1 = ['Parameter{0}'.format(i) for i in range(1, 11)]##???

attr_feat = ['Attribute{0}'.format(i) for i in range(1, 11)]

\# 查看特征分布

data_all=data.copy()

fig = plt.subplots(figsize=(30,20))

j = 1

for column in para_feat1:

  plt.subplot(5, 8, j)

  g = sns.kdeplot(data_all[column][(~data_all["label"].isnull())], color="Red", shade = True)

  g = sns.kdeplot(data_all[column][(data_all["label"].isnull())], ax =g, color="Blue", shade= True)

  g.set_xlabel(column)

  g.set_ylabel("Frequency")

  g = g.legend(["train","test"])

  j += 1

plt.show()

### ***\*2、\*******\*移除相关变量的阈值\*******\*-\*******\*热力图\****

\# 找出相关程度，绘制热力图

data_train1=train.copy()

plt.figure(figsize=(20, 16))  # 指定绘图对象宽度和高度

colnm = data_train1.columns.tolist()  # 列表头

mcorr = data_train1[colnm].corr(method="spearman")  # 相关系数矩阵，即给出了任意两个变量之间的相关系数

mask = np.zeros_like(mcorr, dtype=np.bool)  # 构造与mcorr同维数矩阵 为bool型

mask[np.triu_indices_from(mask)] = True  # 遮挡热力图上三角部分的mask

cmap = sns.diverging_palette(220, 10, as_cmap=True)  # 返回matp|lotlib colormap对象

g = sns.heatmap(mcorr, mask=mask, cmap=cmap, square=True, annot=True, fmt='0.2f')  # 热力图（看两两相似度）

plt.show()

 

\# 移除相关变量的阈值

threshold = 0.1

\# 绝对值相关矩阵

corr_matrix = data_train1.corr().abs() # 相关系数矩阵加绝怼值 33*33

\# 删除与target的相关系数小于threshold的特征

drop_col=corr_matrix[corr_matrix["target"]<threshold].index

data_all.drop(drop_col,axis=1,inplace=True)

### ***\*3最大最小归一化\****

\# 数据基本统计量，包含：

\# count：数量  mean：均值  std：标准差  min：最小值  25%：下四分位  50%：中位数  75%：上四分位  max：最大值

data_all=data.copy()

cols_numeric=list(data_all.columns)

def scale_minmax(col):

  return (col-col.min())/(col.max()-col.min())

scale_cols = para_feat1+attr_feat

data_all[scale_cols] = data_all[scale_cols].apply(scale_minmax,axis=0)

data_all[scale_cols].describe()

print(data_all[scale_cols].describe())

### ***\*4数据筛选\****

mcorr = **df_train**.corr()
\# 和target相关性较小的feature
drop_list_2 = [c **for** c **in** mcorr['target'].index **if** abs(mcorr['target'][c]) < 0.15]

 

 

### ***\*#画图\*******\*：\*******\*查看数值规律\*******\*bar/线\****

**时间序列代码：**https://mp.weixin.qq.com/s?__biz=MzU1Nzc1NjI0Nw==&mid=2247484533&idx=1&sn=8cbc7576d01fc404a57306116910aa38 **#Plotting data**

import matplotlib.pyplot as plt 

train['count'].plot(figsize=(15,8), title= 'Daily Ridership', fontsize=14)

plt.show()

plt.plot(train.index, train['count'], label='Train')

plt.legend(loc='best')

plt.title("Naive Forecast")

 

#### ***\*是否正太分布\****

import seaborn as sns

import matplotlib.pyplot as plt

\#%matplotlib inline #不输出

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体

plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

sns.set(font='SimHei')  # 解决Seaborn中文显示问题

j=1

fig = plt.subplots(figsize=(30,20))

for i in para_feat:

  plt.subplot(5, 8, j)

  data[i].hist()

  j += 1

plt.show()

#### ***\*Sns使用\****

https://www.jianshu.com/p/8bb06d3fd21b

plt.rc("font",family="SimHei",size="12")  #用于解决中文显示不了的问题

sns.set_style("whitegrid") 

#### ***\*线状\****

gp_month_mean = df.groupby(['province'], as_index=False)['popularity'].mean()

 

sns.lineplot(x="province", y="popularity", data=gp_month_mean).set_title("Monthly mean")

sns.barplot(x="province", y="popularity", data=gp_month_mean).set_title("Monthly mean")

sns.jointplot(x='popularity',y='label',data=data,height=8)

sns.boxplot(data=data['popularity'])

plt.show()

### ***\*散点\****

**# 数据没有lable**

data ***\*=\**** pd***\*.\****read_csv('/home/kesci/input/dataset7402/column_2C_weka.csv')

plt***\*.\****scatter(data['pelvic_radius'],data['degree_spondylolisthesis'])

plt***\*.\****xlabel('pelvic_radius')

plt***\*.\****ylabel('degree_spondylolisthesis')

plt***\*.\****show()

 

#### ***\*皮尔逊热图：\****

| colormap = plt.cm.viridis plt.figure(figsize=(30,30))        |
| ------------------------------------------------------------ |
| plt.title('Pearson Correlation of Features', y=1.05, size=15) |
| sns.heatmap(DF.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True) |

 

### ***\*数据保存\****

df.to_pickle('data/testA/totalExposureLog.pkl')

train_df=pd.read_pickle(path1)

## 2数据预处理

### 数据处理

```python
#数据处理
#合并
train = train.merge(train_stacking, 'left', 'user_id')
test = test.merge(test_stacking, 'left', 'user_id')


data['gender'] = data.gender.astype(int) #类型转换

origin_cate_feature=[]
origin_num_feature =[]
#特征处理
for i in origin_num_feature:
    data[i] = data[i].astype(float)
	
```



### 最大最小归一化

\# scaling function
**def** scale(**train_features**,**test_features**):

  scaler=MinMaxScaler()
  data=pd.concat([**train_features**,**test_features**])
  scaled_data=pd.DataFrame(scaler.fit_transform(data),columns=**test_features**.keys())



 

### 代码11.27--0818数据格式化处理方法

```python
# 重复数据的拼接操作

def merge_table(df):
    df['field_results'] = df['field_results'].astype(str)
    if df.shape[0] > 1:
        merge_df = " ".join(list(df['field_results']))
    else:
        merge_df = df['field_results'].values[0]
    return merge_df

# 数据简单处理

print('find_is_copy')
print(part_1_2.shape)
is_happen = part_1_2.groupby(['vid','table_id']).size().reset_index()

# 重塑index用来去重

is_happen['new_index'] = is_happen['vid'] + '_' + is_happen['table_id']
is_happen_new = is_happen[is_happen[0]>1]['new_index']

part_1_2['new_index'] = part_1_2['vid'] + '_' + part_1_2['table_id']

unique_part = part_1_2[part_1_2['new_index'].isin(list(is_happen_new))]
unique_part = unique_part.sort_values(['vid','table_id'])
no_unique_part = part_1_2[~part_1_2['new_index'].isin(list(is_happen_new))]
print('begin')
part_1_2_not_unique = unique_part.groupby(['vid','table_id']).apply(merge_table).reset_index()
part_1_2_not_unique.rename(columns={0:'field_results'},inplace=True)
print('xxx')
tmp = pd.concat([part_1_2_not_unique,no_unique_part[['vid','table_id','field_results']]])

# 行列转换

print('finish')
tmp = tmp.pivot(index='vid',values='field_results',columns='table_id')
tmp.to_csv('../tmp/tmp.csv')
print(tmp.shape)
print('totle time',time.time() - begin_time)
```



### ***\*Plt画图\****

df = data['Parameter1']

fig,axes = plt.subplots(1,2,figsize = (10,4))

ax1 = axes[0]

ax1.scatter(df.index,df.values)

 

ax2 = axes[1]

df.hist(bins = 20,alpha = 0.7,ax = ax2)

df.plot(kind = 'kde',secondary_y = True,ax = ax2)  #使用y轴作为副坐标轴

### ***\*时间\****

10.26.2019

![img](file:///C:\Users\sereny\AppData\Local\Temp\ksohtml103712\wps2.jpg)![img](file:///C:\Users\sereny\AppData\Local\Temp\ksohtml103712\wps3.jpg) 

train.rename(columns={'时间':'time','辐照度':'irradiance','风速':'speed','风向':'direction','温度':'temp','压强':'pressure','湿度':'humidity','实发辐照度':'real_irradiance','实际功率':'Power'},inplace=***\*True\****)

```python

for x in tqdm(train_df['request_timestamp'].values,total=len(train_df)):
  localtime=time.localtime(x)
  wday.append(localtime[6])
train_df['wday']=wday
train_df['period_id']=train_df['hour']*2+train_df['minute']//30
request_day=time.mktime(time.strptime(line[1], '%Y%m%d%H%M%S'))//(3600*24)


# 利用每列平均值填充缺失值from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(X[: , 1:3 ])
X[: ,1:3] = imputer.transform(X[:, 1:3])# 上面一步还可以这样做代替# from sklearn.impute import SimpleImputer# imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')# X[: ,1:3] = imputer.fit_transform(X[:, 1:3])

# 拆分数据集为训练集合和测试集合from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X , Y , test_size = 0.2, random_state = 0)
# 特征缩放from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
```



### ***\*异常值\****

(np.abs(series_1.values - x) <= scope * (feature_1_max - feature_1_min))

feature_index=**data**[~((**data**[feature] >= (Q1 - step)) & (**data**[feature] <= (Q3 + step)))].index

#### ***\*#异常处理（\*******\*设1.5倍四分位距之外的数据为异常值\****

1.5*（Q3-Q1）以外3的取值为异常，其中Q3和Q1为数据的较大四分位和较小四分位。在比赛中，我们设置的q*(Q3-Q1)，q的取值1.4、1.5、10或者其他值。

，对这些异常数据的不同处理方式会影响后续算法的表现。一般大家都会想到，使用平均值、前值、特殊值填充等策略，具体哪种好，可能需要具体场景下多试几次才知道

 

\# col, col2, col3 中 ，设1.5倍四分位距之外的数据为异常值，用上下四分位数的均值填充

for col in ['popularity', 'carCommentVolum', 'newsReplyVolum']:

  col_per=np.percentile(df[col],(25,75))#Q1 = np.percentile(data[feature],25)

  diff=1.5*(col_per[1] - col_per[0])

  col_per_in = (df[col] >= col_per[0] - diff) & (df[col] <= col_per[1] + diff)

data.loc[~col_per_in, col] = col_per.mean()

用前一个填充

 

### ***\*填充\****

app_train_test=app_train_test.fillna(method='ffill')

### ***\*脏数据\****

脏数据：两者相关互相最小，如果不相关则为脏数据，晚上有太阳辐射为脏数据

 

### ***\*数据转换对应\****

quality_map = {'Excellent': 0, 'Good': 1, 'Pass': 2, 'Fail': 3}

train['label'] = train['Quality_label'].map(quality_map)

### ***\*排序sort\****

gp_category_sum.sort_values('label')

### ***\*查看重复缺失\****

\##  重复值

print(train_monthly.duplicated().any())

print('**'*30)

\## 缺失值常用

print(train_monthly.isnull().sum())

### ***\*去重\****

op_df=op_df.drop_duplicates(['aid','day'],keep='last')

op_df=op_df.drop_duplicates('aid',keep='last')

 

 

### ***\*腾讯赛top1：\****

```python
df=pd.read_csv('data/testA/totalExposureLog.out', sep='\t',names=['id','request_timestamp','position','uid','aid','imp_ad_size','bid','pctr','quality_ecpm','totalEcpm']).sort_values(by='request_timestamp')
df[['id','request_timestamp','position','uid','aid','imp_ad_size']]=df[['id','request_timestamp','position','uid','aid','imp_ad_size']].astype(int)
df[['bid','pctr','quality_ecpm','totalEcpm']]=df[['bid','pctr','quality_ecpm','totalEcpm']].astype(float) 
df.to_pickle('data/testA/totalExposureLog.pkl') 
del df
gc.collect()

填充

df=df.fillna(-1)
for f in ['aid','create_timestamp','advertiser','good_id','good_type','ad_type_id']:
  items=[]
  for item in df[f].values:
    try:
      items.append(int(item))
    except:
      items.append(-1)
  df[f]=items
  df[f]=df[f].astype(int)
df['ad_size']=df['ad_size'].apply(lambda x:' '.join([str(int(float(y))) for y in str(x).split(',')]))   

过滤

#过滤出价和曝光过高的广告
train_df=train_df[train_df['imp']<=3000]
train_df=train_df[train_df['bid']<=1000]

 
```



### ***\*离散值\****

ad_logdata = ad_logdata[ad_logdata.Adbid<5e+6]

ad_static = ad_static[ad_static.adCreatetime != 0] #(726621, 7) #保留不等于0的
ad_static=ad_static[ad_static['adIndustryid'].apply(**lambda** x:','**not  in** x)  ]

 

## 3特征

### 特征处理

```python
df['sex'] = df['sex'].map( {'female': 0, 'male': 1} ).astype(int)
df["embarked"] = df['embarked'].dropna().map( {'S':0, 'C':1, 'Q':2} ).astype(int)
```


​    

### 2连续特征

#### #分箱分桶

```python
df['agebin'] = np.array(np.floor(np.array(df['Age']) / 10.))
#2函数
'''用map一个一个对应，2用粪桶，bin 和catebin  3用函数+aply'''
#apply
def get_refer_day(d):
    if d == 20:
        return 29
    elif d in [5,6,7,8,9]:
        return d + 2
tmp['day'] = tmp['day'].apply(get_refer_day)

for f in stat_columns:
    tmp.rename(columns={f: f+'_last'}, inplace=True) 
	
```



### 1类别特征onehot

```python
#Onehot
sex_df = pd.get_dummies(user["sex"], prefix="sex")
user = pd.concat([user, sex_df], axis=1) #行对其，左右

# 解析分类特征数据from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# 创建虚拟变量，对第一列数据进行onehot，也就是country这列
onehotencoder = OneHotEncoder(categories = X[:, 0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)
```



#### 1.1labelencoder

```python
from sklearn.preprocessing import LabelEncoder
for fea in ['primary_use']:
    df['primary_use']=df[fea].fillna('NA')  #重要
    df['le_'+fea]= LabelEncoder().fit_transform(df[fea])
```

### 3文本特征

```python
文本处理，去除停用词，采用weword2vec平均、求和、tf-idf求和的方

tensorfloe--gpu
有个字符串相似度检测的库，difflib。

单词的组合特征
去噪，特征筛选，调参，模型组合，结果分析几乎都没做，要做的我也没做，自然有语言处理也没做完

文本中携带的信息太多了，总结一下大家的做法：
(1) 先把文本为主的和数据为主的分开．
(2) 对文本为主的：分词，提取关键字，统计出现频率（TF/IDF，聚类等等）．
(3) 判断高频词是否在字段是存在，生成一些的的布尔特征．
(4) 人工检查高频文本，选出像＂脂肪肝＂，＂高血糖＂这样的关键字及字段写正则．
(5) 还有直接在Excel里用公式做的，手工改的…


```

#### 代码

```python
#分词
	def process_text(self):
		# 停用词
		with codecs.open(self.stopwords, encoding='utf-8', errors='ignore')as f:
		    stop =set()
		    for line in f: #载入停用词
		        stop.add(line.strip())        
		def de_stop(line):
		    line = line.strip().split()
		    res = []
		    for word in line:
		        if word not in stop:
		            res.append(word)
		    return ' '.join(res)
		# 去停用词的微博正文
		self.text['words'] = self.text.content.apply(lambda x: de_stop(x))
		temp = pd.DataFrame(self.text.groupby('uid')['words'].apply(lambda x: ' '.join(x))).reset_index()
		self.data_x = self.data_x.merge(temp[['uid','words']], how='left',on='uid')
		# source信息
		temp = pd.DataFrame(self.status.groupby('uid')['source'].apply(lambda x: ' '.join(x))).reset_index()
		self.data_x = self.data_x.merge(temp[['uid','source']], how='left',on='uid')
		# source分词
		def fenci(line):
		    line = line.strip().split()
		    line = ''.join(line)
		    seglist = jieba.cut(line)
		    line = ' '.join(seglist)
		    return line
		self.data_x['source_fenci'] = self.data_x.source.apply(lambda x:fenci(x))
		self.data_x['weibo_and_source'] = (self.data_x.words + self.data_x.source_fenci)
		
		


'''svd降维'''
svd=TruncatedSVD(n_components=500)
svd.fit(f_word[0])
f_word_pca=svd.transform(f_word[0]),svd.transform(f_word[1])

svd=TruncatedSVD(n_components=500)
svd.fit(f_letter[0])
f_letter_pca=svd.transform(f_letter[0]),svd.transform(f_letter[1])

'''词tfidf特征 ngram1'''
tfidf4=TfidfVectorizer(min_df=3,ngram_range=(1,1))
tfidf4.fit(xs[0])
f_word_n1=[tfidf4.transform(x) for x in xs]
print('f_word_n1',f_word_n1[0].shape)

'''字tfidf特征 ngram1'''
tfidf5=TfidfVectorizer(min_df=3,ngram_range=(1,1),analyzer='char')
tfidf5.fit(xs[0])
f_letter_n1=[tfidf5.transform(x) for x in xs]
print('f_letter_n1',f_letter_n1[0].shape)

'''source tfidf 特征'''
source=get(train_data,9),get(test_data,9)
tfidf6=TfidfVectorizer(min_df=3,ngram_range=(1,1))
tfidf6.fit(source[0])
f_source_tfidf=[tfidf6.transform(x) for x in source]
'''字tfidf特征'''
tfidf2=TfidfVectorizer(min_df=5,ngram_range=(1,2),analyzer='char')
tfidf2.fit(xs[0])
f_letter=[tfidf2.transform(x) for x in xs]
print('f_letter',f_letter[0].shape)

'''话题词特征'''
theme=get(train_data,6),get(test_data,6)
xs=theme
tfidf3=TfidfVectorizer(min_df=5,ngram_range=(1,2),analyzer='char')
tfidf3.fit(xs[0]+xs[1])
f_theme_word=[tfidf3.transform(x) for x in xs]

'''话题特征'''
nospace=[[item.replace(' ','') for item in t] for t in theme]
f_theme=get_f_cnter(nospace)

'''话题字特征'''
xs=nospace
tfidf3=TfidfVectorizer(min_df=5,ngram_range=(1,2),analyzer='char')
tfidf3.fit(xs[0]+xs[1])
f_theme_letter=[tfidf3.transform(x) for x in xs]


'''获得计数类文本特征'''
def get_f_cnter(xs,min_df=5):
    cnter=CountVectorizer(min_df=min_df)
    cnter.fit(xs[0]+xs[1])
    f_xs=[cnter.transform(x) for x in xs]
    return f_xs
	
'''信息来源'''
source=get(train_data,9),get(test_data,9)
f_source=get_f_cnter(source)

'''关键词表'''
xs=content
keywords=load_keywords()
cnter=CountVectorizer(binary=True,vocabulary=keywords)
cnter.fit(xs[0]+xs[1])
f_keywords=[cnter.transform(x) for x in xs]


##
sentence = "hello! wo?rd!."
cleanr = re.compile('<.*?>')
sentence = re.sub(cleanr, ' ', sentence)        #去除html标签
sentence = re.sub(r4,'',sentence)
print(sentence)


先筛掉一部分词汇，比如常用词，语气词，，tfidf降低维度
拿到了分词后的文件，在一般的NLP处理中，会需要去停用词。由于word2vec的算法依赖于上下文，而上下文有可能就是停词。因此对于word2vec，我们可以不用去停词。
现在我们可以直接读分词后的文件到内存。这里使用了word2vec提供的LineSentence类来读文件，然后套用word2vec的模型。
这里只是一个示例，因此省去了调参的步骤，实际使用的时候，你可能需要对我们上面提到一些参数进行调参。
tfidf要去掉停用词
```

#### doc2vec

```python
#doc2vec-训练--------------------------------------------
SentimentDocument = namedtuple('SentimentDocument', 'words tags')
class Doc_list(object):
    def __init__(self,f):
        self.f = f
    def __iter__(self):
        for i,line in enumerate(codecs.open(self.f,encoding='utf8')):
            words = line.split()
            tags = [int(words[0][2:])]
            words = words[1:]
            yield SentimentDocument(words,tags)
d2v = Doc2Vec(dm=0, size=300, negative=5, hs=0, min_count=3, window=30,sample=1e-5,workers=8,alpha=0.025,min_alpha=0.025)
doc_list = Doc_list('alldata-id.txt')
d2v.build_vocab(doc_list)

#-------------------train dbow doc2vec---------------------------------------------
df_lb = pd.read_csv(cfg.data_path + 'all_v2.csv',usecols=['Education','age','gender'],nrows=200000)
ys = {}
for lb in ['Education','age','gender']:
    ys[lb] = np.array(df_lb[lb])

for i in range(2):
    print(datetime.now(),'pass:',i + 1)
    run_cmd('shuf alldata-id.txt > alldata-id-shuf.txt')#乱序
    doc_list = Doc_list('alldata-id.txt')
    d2v.train(doc_list)
    X_d2v = np.array([d2v.docvecs[i] for i in range(200000)])
    for lb in ["Education",'age','gender']:
        scores = cross_val_score(LogisticRegression(C=3),X_d2v,ys[lb],cv=5)
        print('dbow',lb,scores,np.mean(scores))
d2v.save(cfg.data_path + 'dbow_d2v.model')
print(datetime.now(),'save done')
```

#### 文本处理职位

```
网上https://www.cnblogs.com/jkmiao/p/4874803.html
```



### ***\*2数值特征10.29看光伏预测\****

https://mp.weixin.qq.com/s/Yix0xVp2SiqaAcuS6Q049g 光伏发电量预测

#### ***\*PolynomialFeatures构造\*******\*特征和\*******\*sklearn\*******\*的\*******\*Imputer\*******\*+corr\****

数值特征
+-*/根据变量

```python
# app_train['C_4']=app_train['功率A']/app_train['风速']
# app_test['C_4']=app_test['功率A']/app_test['风速']

# app_train['C_5']=app_train['功率B']/app_train['风速']
# app_test['C_5']=app_test['功率B']/app_test['风速']
#
# app_train['C_6']=app_train['C_4']app_train['C_4']
# app_test['C_6']=app_test['C_4']app_test['C_4']
# app_train['C']=app_train['电流B']-app_train['电流C']
# app_test['C']=app_test['电流B']-app_test['电流C']

# app_train['C_3']=app_train['dis2peak']app_train['平均功率']
# app_test['C_3']=app_test['dis2peak']app_test['平均功率']

poly_features = app_train[['板温', '现场温度', '光照强度', '风速', '风向']]
poly_features_test = app_test[['板温', '现场温度', '光照强度', '风速', '风向']]
poly_target = app_train['发电量']

# imputer for handling missing values
from sklearn.preprocessing import Imputer
imputer = Imputer(strategy='median')
# Need to impute missing values
poly_features = imputer.fit_transform(poly_features) #训练集缺失值/（平均值、中位数或 众数 等）进行输入。
poly_features_test = imputer.transform(poly_features_test) #测试机

poly_transformer = PolynomialFeatures(degree=2)
poly_transformer.fit(poly_features)
# Transform the features
poly_features = poly_transformer.transform(poly_features)
poly_features_test = poly_transformer.transform(poly_features_test)
print('Polynomial Features shape: ', poly_features.shape)

poly_features = pd.DataFrame(poly_features,
               columns=poly_transformer.get_feature_names(['板温', '现场温度', '光照强度', '风速', '风向']))
 '板温', '现场温度', '光照强度', '风速', '风向', '板温^2', '板温 现场温度', '板温 光照强度', '板温 风速', '板温 风向', '现场温度^2', '现场温度 光照强度', '现场温度 风速', '现场温度 风向', '光照强度^2', '光照强度 风速', '光照强度 风向', '风速^2', '风速 风向', '风向^2',
# Add in the target
poly_features['TARGET'] = poly_target

# Find the correlations with the target
poly_corrs = poly_features.corr()['TARGET'].sort_values() #根据相关度选择

# Display most negative and most positive
# print(poly_corrs)
# Put test features into dataframe
poly_features_test = pd.DataFrame(poly_features_test,
                 columns=poly_transformer.get_feature_names(['板温', '现场温度', '光照强度', '风速', '风向']))
''''''

''''''
# Merge polynomial features into training dataframe
poly_features['ID'] = app_train['ID']
app_train_poly = app_train.merge(poly_features, on='ID', how='left')

# Merge polnomial features into testing dataframe
poly_features_test['ID'] = app_test['ID']
app_test_poly = app_test.merge(poly_features_test, on='ID', how='left')

# Align the dataframes
app_train_poly, app_test_poly = app_train_poly.align(app_test_poly, join='inner', axis=1)

app_train_poly['发电量'] = poly_target
app_train = app_train_poly
app_test = app_test_poly

 
```

 

### ***\*特征降维\****

\# pca = PCA(n_components = 0.95)
pca = PCA(n_components='mle', svd_solver='full')
pca.fit(df_train_x)
df_train_x = pca.transform(df_train_x)
df_test = pca.transform(df_test)

 

统计特征

#### ***\*transform\****

all_data['user_jd_city_nunique'] = all_data.groupby('user_id')['city'].transform('nunique').values

#### ***\*Lambda：\****

  all_data['desire_jd_type_id_len'] = all_data['desire_jd_type_id'].apply(

​    lambda x: len(x.split(',')) if isinstance(x, str) else np.nan)

#### ***\*分类统计提取特征groupby/agg\****

  \# 提取特征 2018-04-04 1天

  start_day = '2018-04-04 00:00:00'

  for gb_c in [['user_id'],  # 用户

​         ['cate'],  # 品类

​         ['shop_id'],  # 店铺

​         ['user_id', 'cate'],  # 用户-品类

​         ['user_id', 'shop_id'],  # 用户-店铺

​         ['cate', 'shop_id'],  # 品类-店铺

​         ['user_id', 'cate', 'shop_id']]:  # 用户-品类-店铺

​    print(gb_c)

​    

​    action_temp = jdata_data[(jdata_data['action_time'] >= start_day)

​                 & (jdata_data['action_time'] <= end_day)]

 

​    \# 特征函数

​    features_dict = {

​      'sku_id': [np.size, lambda x: len(set(x))],

​      'type': lambda x: len(set(x)),

​      'brand': lambda x: len(set(x)),

​      'shop_id': lambda x: len(set(x)),

​      'cate': lambda x: len(set(x)),

​      'action_time': [

​        get_first_hour_gap,  # first_hour_gap

​        get_last_hour_gap,  # last_hour_gap

​        get_act_days  # act_days

​      ]

 

​    } 

​    features_columns = [c +'_' + '_'.join(gb_c)

​              for c in ['sku_cnt', 'sku_nq', 'type_nq', 'brand_nq', 'shop_nq', 'cate_nq', 'first_hour_gap', 'last_hour_gap', 'act_days']]

​    f_temp = action_temp.groupby(gb_c).agg(features_dict).reset_index()

 

grouped = sales_train_subset[['shop_id','item_id','item_cnt_day']].groupby(['shop_id','item_id']).agg({'item_cnt_day':'sum'}).reset_index() 

grouped = grouped.rename(columns={'item_cnt_day' : 'item_cnt_month'}) 

 

### 3统计特征几种方法（有用

#### 理论

2.5 统计型

加减平均：商品价格高于平均价格多少，用户在某个品类下消费超过平均用户多少，用户连续登录天数超过平均多少...
分位线：商品属于售出商品价格的多少分位线处
次序型：排在第几位
比例类：电商中，好/中/差评比例，你已超过全国百分之…的同学

```
train['TransactionAmt_to_mean_card1'] = train['TransactionAmt'] / train.groupby(['card1'])['TransactionAmt'].transform('mean')
train['TransactionAmt_to_mean_card4'] = train['TransactionAmt'] / train.groupby(['card4'])['TransactionAmt'].transform('mean')
train['TransactionAmt_to_std_card1'] = train['TransactionAmt'] / train.groupby(['card1'])['TransactionAmt'].transform('std')
train['TransactionAmt_to_std_card4'] = train['TransactionAmt'] / train.groupby(['card4'])['TransactionAmt'].transform('std')
```



#### 例子

```python
df = df_.copy()

# base time
df['day']     = df['time'].apply(lambda x: int(x[8:10]))
df['week']    = pd.to_datetime(df['time']).dt.dayofweek + 1
df['weekend'] = (pd.to_datetime(df.time).dt.weekday >=5).astype(int)
df['minute']  = df['time'].apply(lambda x: int(x[14:15]+'0'))

# count,sum
result = df.groupby(['ID', 'week', ]).status.agg(['count', 'sum']).reset_index()

# nunique
tmp     = df.groupby(['stationID'])['deviceID'].nunique().reset_index(name='nuni_deviceID_of_stationID')
result  = result.merge(tmp, on=['stationID'], how='left')
tmp     = df.groupby(['stationID','hour'])['deviceID'].nunique().reset_index(name='nuni_deviceID_of_stationID_hour')
result  = result.merge(tmp, on=['stationID','hour'], how='left')
tmp     = df.groupby(['stationID','hour','minute'])['deviceID'].nunique().\
                                       reset_index(name='nuni_deviceID_of_stationID_hour_minute')
result  = result.merge(tmp, on=['stationID','hour','minute'], how='left')

# in,out加减
result['inNums']  = result['sum']
result['outNums'] = result['count'] - result['sum']

#
result['day_since_first'] = result['day'] - 1 
result.fillna(0, inplace=True)
del result['sum'],result['count']

return result
```



#### 均值

```python
b=df.groupby(['bt']).agg({'label':['sum','min','max','mean']}).reset_index()
b.columns=['bt','le_bdsum','le_bdmin','le_bdmax','le_bdmean']
df = pd.merge(df, b, on=['bt'], how='left')

\#某个月份这个模型/省份的销售量
adf=df.groupby(['Mon','pro'])['label'].sum().reset_index()
adf['le_mean'] =df.groupby(['Mon','pro'])['label'].transform('mean')

 

result= log.groupby([f,'day'],as_index=**False**)['request_cont'].sum()
result.columns=[f,'day',f+'_cnts']

data_df=data_df.merge(result,on=[f,'day'],how='left')
```





### 基本特征agg分组特征

**for** item **in** items:
  temp = train_data.groupby(item, as_index = **False**)['label'].agg({item+'_click':'sum', item+'_count':'count'})
  temp[item+'_ctr'] = temp[item+'_click']/(temp[item+'_count'])
  train_data = pd.merge(train_data, temp, on=item, how='left')

 

\# 构造基本特征
**for** col **in** ['aid','goods_id','account_id']:
  print(col)
  result = logs.groupby([col,'day'], as_index=**False**)['isExp'].agg({
    col+'_cnts'    : 'count',
    col+'_sums'    : 'sum',
    col+'_rate'    : 'mean'
    })
  result[col+'_negs'] = result[col+'_cnts'] - result[col+'_sums']
  data = data.merge(result, how='left', on=[col,'day'])

### 交叉组合特征构造

2.6 组合特征

1. 拼接型：简单的组合特征。例如挖掘用户对某种类型的喜爱，对用户和类型做拼接。正负权重，代表喜欢或不喜欢某种类型。

- user_id&&category: 10001&&女裙 10002&&男士牛仔
- user_id&&style: 10001&&蕾丝 10002&&全棉　

2. 模型特征组合：

- 用GBDT产出特征组合路径
- 组合特征和原始特征一起放进LR训练

转化为二分类看一下，，
#构造基本特征
def get_base_features(df_):

#### 2加减乘除：

数值加减，字符串加减构造新的labelencoder

```python
df['adcode_model'] = df['adcode'] + df['model']
df['adcode_model_mt'] = df['adcode_model'] * 100 + df['mt']
```



#### 1unique特征

```python
########################### Encode Meter
#################################################################################
# Building and site id
for enc_col in ['building_id', 'site_id']: #每个建筑物/天气类型有几个仪表分类，unique集合的labelcoder
    temp_df = train_df.groupby([enc_col])['meter'].agg(['unique'])
    temp_df['unique'] = temp_df['unique'].apply(lambda x: '_'.join(str(x))).astype(str)

    le = LabelEncoder()
    temp_df['unique'] = le.fit_transform(temp_df['unique']).astype(np.int8)
    temp_df = temp_df['unique'].to_dict()

    train_df[enc_col+'_uid_enc'] = train_df[enc_col].map(temp_df)
    test_df[enc_col+'_uid_enc'] = test_df[enc_col].map(temp_df)
    
    # Nunique
    temp_dict = train_df.groupby([enc_col])['meter'].agg(['nunique'])['nunique'].to_dict() #个数
    train_df[enc_col+'-m_nunique'] = train_df[enc_col].map(temp_dict).astype(np.int8)
    test_df[enc_col+'-m_nunique'] = test_df[enc_col].map(temp_dict).astype(np.int8)

del temp_df, temp_dict
```



### ***\*购买特征按照时间衰减\****

| actions['weights'] = actions['time'].map(lambda x: datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(x, '%Y-%m-%d %H:%M:%S')) |
| ------------------------------------------------------------ |
| #actions['weights'] = time.strptime(end_date, '%Y-%m-%d') - actions['datetime'] |
| actions['weights'] = actions['weights'].map(lambda x: math.exp(-x.days)) |
| print actions.head(10)                                       |
| actions['action_1'] = actions['action_1'] * actions['weights'] |
| actions['action_2'] = actions['action_2'] * actions['weights'] |

2

| actions = pd.concat([actions['sku_id'], df], axis=1)actions = actions.groupby(['sku_id'], as_index=False).sum() |
| ------------------------------------------------------------ |
| actions['product_action_1_ratio'] = actions['action_4'] / actions['action_1'] |
| actions['product_action_2_ratio'] = actions['action_4'] / actions['action_2'] |

### 时间特征

####  to_datetime 

```python
pd.to_datetime(test['startTime']).dt.dayofweek
```

![image-20191126192014493](D:/ruanjiandata/Typora_mdpic/image-20191126192014493.png)

#### 时间特征代码平移

```python
def get_shift_feature(df_, start, end, col, group): #   df,1,9,'label','adcode_model_mt'
    '''
    历史平移特征
    col  : label,popularity
    group: adcode_model_mt, model_mt
    '''
    df = df_.copy()
    add_feat = []
    for i in range(start, end+1):
        add_feat.append('shift_{}_{}_{}'.format(col,group,i))  #
        df['{}_{}'.format(col,i)] = df[group] + i    #  
        df_last = df[~df[col].isnull()].set_index('{}_{}'.format(col,i))  #adcode_model_mt1等作为index
        df['shift_{}_{}_{}'.format(col,group,i)] = df[group].map(df_last[col]) # adcode_model_mt  map到  df_last
        del df['{}_{}'.format(col,i)]
    return df, add_feat

def get_adjoin_feature(df_, start, end, col, group, space):  # (df, start, end, ,'label', 'adcode_model_mt', space=1)
    '''
    相邻N月的首尾统计
    space: 间隔
    Notes: shift统一为adcode_model_mt
    '''
    df = df_.copy()
    add_feat = []
    for i in range(start, end+1):   
        add_feat.append('adjoin_{}_{}_{}_{}_{}_sum'.format(col,group,i,i+space,space)) # 求和
        add_feat.append('adjoin_{}_{}_{}_{}_{}_mean'.format(col,group,i,i+space,space)) # 均值
        add_feat.append('adjoin_{}_{}_{}_{}_{}_diff'.format(col,group,i,i+space,space)) # 首尾差值
        add_feat.append('adjoin_{}_{}_{}_{}_{}_ratio'.format(col,group,i,i+space,space)) # 首尾比例
        df['adjoin_{}_{}_{}_{}_{}_sum'.format(col,group,i,i+space,space)] = 0
        for j in range(0, space+1):
            df['adjoin_{}_{}_{}_{}_{}_sum'.format(col,group,i,i+space,space)]   = df['adjoin_{}_{}_{}_{}_{}_sum'.format(col,group,i,i+space,space)] +\
                                                                                  df['shift_{}_{}_{}'.format(col,'adcode_model_mt',i+j)]
        df['adjoin_{}_{}_{}_{}_{}_mean'.format(col,group,i,i+space,space)]  = df['adjoin_{}_{}_{}_{}_{}_sum'.format(col,group,i,i+space,space)].values/(space+1)
        df['adjoin_{}_{}_{}_{}_{}_diff'.format(col,group,i,i+space,space)]  = df['shift_{}_{}_{}'.format(col,'adcode_model_mt',i)].values -\
                                                                              df['shift_{}_{}_{}'.format(col,'adcode_model_mt',i+space)]
        df['adjoin_{}_{}_{}_{}_{}_ratio'.format(col,group,i,i+space,space)] = df['shift_{}_{}_{}'.format(col,'adcode_model_mt',i)].values /\
                                                                              df['shift_{}_{}_{}'.format(col,'adcode_model_mt',i+space)]
    return df, add_feat

def get_series_feature(df_, start, end, col, group, types): #(df, start, end, ,'label', 'adcode_model_mt', ['sum','mean','min','max','std','ptp']) 
    '''
    连续N月的统计值
    Notes: shift统一为adcode_model_mt
    '''
    df = df_.copy()
    add_feat = []
    li = []
    df['series_{}_{}_{}_{}_sum'.format(col,group,start,end)] = 0
    for i in range(start,end+1):
        li.append('shift_{}_{}_{}'.format(col,'adcode_model_mt',i))
    df['series_{}_{}_{}_{}_sum'.format( col,group,start,end)] = df[li].apply(get_sum, axis=1)
    df['series_{}_{}_{}_{}_mean'.format(col,group,start,end)] = df[li].apply(get_mean, axis=1)
    df['series_{}_{}_{}_{}_min'.format( col,group,start,end)] = df[li].apply(get_min, axis=1)
    df['series_{}_{}_{}_{}_max'.format( col,group,start,end)] = df[li].apply(get_max, axis=1)
    df['series_{}_{}_{}_{}_std'.format( col,group,start,end)] = df[li].apply(get_std, axis=1)
    df['series_{}_{}_{}_{}_ptp'.format( col,group,start,end)] = df[li].apply(get_ptp, axis=1)
    for typ in types:
        add_feat.append('series_{}_{}_{}_{}_{}'.format(col,group,start,end,typ))
    
    return df, add_feat
```

#### 天，转换成星期，分成上午下午晚上

#### 常用代码

```python
#常用代码日期
test['week']    = pd.to_datetime(test['startTime']).dt.dayofweek + 1
test['weekend'] = (pd.to_datetime(test.startTime).dt.weekday >=5).astype(int)
test['day']     = test['startTime'].apply(lambda x: int(x[8:10]))
test = test.drop(['startTime','endTime'], axis=1)
data = pd.concat([data,test], axis=0, ignore_index=True)

stat_columns = ['inNums','outNums']


#按照时间分隔
df['adReqday_index']=pd.to_datetime(df['adReqday']) #变成了datetime,date是创建时间
df=Train.set_index(['adReqday_index'])
train=df.loc[:'2019-3-11']
val=df.loc['2019-3-12']
trainx=Train_enc[ Train['adReqday'] <=  '2019-03-11']

#字符转换成datetime
for df in [train_df, test_df, train_weather_df, test_weather_df]:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
for df in [train_df, test_df]:
    df['DT_M'] = df['timestamp'].dt.month.astype(np.int8)
    df['DT_W'] = df['timestamp'].dt.weekofyear.astype(np.int8)
    df['DT_D'] = df['timestamp'].dt.dayofyear.astype(np.int16)
    
    df['DT_hour'] = df['timestamp'].dt.hour.astype(np.int8)
    df['DT_day_week'] = df['timestamp'].dt.dayofweek.astype(np.int8)
    df['DT_day_month'] = df['timestamp'].dt.day.astype(np.int8)
   ##  0-6  0 , 6-12  1  ,13-18  2,19-24 3
for df in [train_df, test_df]:
    df['DT_w_hour'] = np.where((df['DT_hour']>5)&(df['DT_hour']<13),1,0)
    df['DT_w_hour'] = np.where((df['DT_hour']>12)&(df['DT_hour']<19),2,df['DT_w_hour'])
    df['DT_w_hour'] = np.where((df['DT_hour']>18),3,df['DT_w_hour'])
```

#### 代码2

```python
时间处理
df['create_order_time']  =2018/7/17 13:27:44
def get_preprocessing(df_):
    df = df_.copy()   
    df['hour']  = df['create_order_time'].apply(lambda x:int(x[11:13]))
    df['date']  = (df['month'].values - 7) * 31 + df['day']    
    del df['create_order_time']    
    return df

train = get_preprocessing(train)

常用代码
df_uni['datetime'] = pd.to_datetime(df_uni['nginxtime'] / 1000, unit='s') + timedelta(hours=8)
df_uni['hour'] = df_uni['datetime'].dt.hour

#查看日期
df_uni['datetime'] = pd.to_datetime(df_uni['nginxtime'] / 1000, unit='s') 
df_uni['datetime'].describe()


在很多情况下，我们的原始数据中的时间和日期并不是时间类型的，例如excel中可能是Unicode，csv中可能是Str。因此我们在进行时间切片之前首先要将非时间类型的时间数据转换为时间类型。
https://blog.csdn.net/hqr20627/article/details/79535533
```



### 时间窗口

11.27滑窗时序特征

```python
def get_rolling_feat(df_, range_list, target_col="label"):
    df = df_.copy()
    df['model_adcode'] = df['adcode'] + df['model']
    rolling_feat = []
    for i in range_list:
        df["rolling_mean_{}_{}".format(i, target_col)] = df.groupby("model_adcode").apply(lambda x: x[target_col].rolling(i).mean().shift(1)).reset_index()[target_col]
        rolling_feat.append("rolling_mean_{}_{}".format(i, target_col))
        df["rolling_median_{}_{}".format(i, target_col)] = df.groupby("model_adcode").apply(lambda x: x[target_col].rolling(i).median().shift(1)).reset_index()[target_col]
        rolling_feat.append("rolling_median_{}_{}".format(i, target_col))
        df["rolling_std_{}_{}".format(i, target_col)] = df.groupby("model_adcode").apply(lambda x: x[target_col].rolling(i).std().shift(1)).reset_index()[target_col]
        rolling_feat.append("rolling_std_{}_{}".format(i, target_col))
        df["rolling_min_{}_{}".format(i, target_col)] = df.groupby("model_adcode").apply(lambda x: x[target_col].rolling(i).min().shift(1)).reset_index()[target_col]
        rolling_feat.append("rolling_min_{}_{}".format(i, target_col))
        df["rolling_max_{}_{}".format(i, target_col)] = df.groupby("model_adcode").apply(lambda x: x[target_col].rolling(i).max().shift(1)).reset_index()[target_col]
        rolling_feat.append("rolling_max_{}_{}".format(i, target_col))
    return df, rolling_feat
```

for month_num in time_gap[gap_name]:
  ##### 满足近期的时间标签
  order['time_label'] = 0
  order.ix[(order['month_diff'] <= month_num) & (order['month_diff'] > 0), 'time_label'] = 1
  order_tmp = order.ix[order['time_label'] == 1, :]

 

```python
actions = get_accumulate_action_feat(train_start_date, train_end_date)
actions = None
for i in (1, 2, 3, 5, 7, 10, 15, 21, 30):
    start_days = datetime.strptime(train_end_date, '%Y-%m-%d')
    - timedelta(days=i)start_days = start_days.strftime('%Y-%m-%d')
    if actions is None:
        actions = get_action_feat(start_days, train_end_date)
     else:
        actions = pd.merge(actions, get_action_feat(start_days, train_end_date), how='left',on=['user_id', 'sku_id'])
actions = pd.merge(actions, user, how='left', on='user_id')
```

 

统计特征


"""
7号前一天，6号的统计特征
用户/商品/品牌/店铺/类别/城市 点击次数，购买次数，转化率，占前面所有天的占比

"""
**def** latest_day_feature(**org**):
  data = **org**[**org**['day'] ==6]
  col = ['user_id', 'item_id', 'item_brand_id', 'shop_id', 'item_category_list', 'item_city_id', 'query1', 'query','context_page_id','predict_category_property']
  train = **org**[**org**['day'] == 7][['instance_id'] + col]
  user = data.groupby('user_id', as_index=**False**)['is_trade'].agg({'user_buy': 'sum', 'user_cnt': 'count'})
  user['user_6day_cvr'] = (user['user_buy']) / (user['user_cnt'] + 3)
  train = pd.merge(train, user[['user_id', 'user_6day_cvr']], on='user_id', how='left')
  items = col[1:]
  **for** item **in** items:
    tmp=data.groupby(item,as_index=**False**)['is_trade'].agg({item+'_buy':'sum',item+'_cnt':'count'})
    tmp[item+'_6day_cvr'] = tmp[item+'_buy'] / tmp[item+'_cnt']
    train = pd.merge(train, tmp[[item, item+'_6day_cvr']], on=item, how='left')
    print(item)
  **for** i **in** range(len(items)):
    **for** j **in** range(i+1,len(items)):
      egg=[items[i],items[j]]
      tmp = data.groupby(egg, as_index=**False**)['is_trade'].agg({'_'.join(egg) + '_buy': 'sum', '_'.join(egg) + '_cnt': 'count'})
      tmp['_'.join(egg) + '_6day_cvr'] = tmp['_'.join(egg) + '_buy'] / tmp['_'.join(egg) + '_cnt']
      train = pd.merge(train, tmp[egg+['_'.join(egg) + '_6day_cvr']], on=egg, how='left')
      print(egg)
  train.drop(col, axis=1).to_csv('../data/6day_cvr_feature.csv',index=**False**)
  **return** train

### ***\*set_index使用\****

df_tmp = df.set_index('LE_com_model_province')
df_tmp
df['shift_model_adcode_mt_{}_{}'.format(col, i)] = df['model_adcode_mt'].map(df_last['label'])

根据数值筛选数字

df=df[df['month_no'].between(22,24)]

### ***\*文本特征\****

  tfidf_enc = TfidfVectorizer(ngram_range=(1, 2))

  tfidf_vec = tfidf_enc.fit_transform(tmp_cut)

  svd_tag = TruncatedSVD(n_components=10, n_iter=20, random_state=2019)

  tag_svd = svd_tag.fit_transform(tfidf_vec)

  tag_svd = pd.DataFrame(tag_svd)

  tag_svd.columns = [f'desc_svd_{i}' for i in range(10)]

  train_jd = pd.concat([train_jd, tag_svd], axis=1)

 



## 4模型

#### 理论

```python
xgb文档：http://www.huaxiaozhuan.com/%E5%B7%A5%E5%85%B7/xgboost/chapters/xgboost_usage.html
xgbrank：https://www.kaggle.com/chriscc/xgb-rank  https://github.com/foxtrotmike/xgbrank/blob/master/xgbr_example.py
CATBOOST：https://catboost.ai/docs/concepts/loss-functions-regression.html
XGB调参：https://www.cnblogs.com/mfryf/p/6293814.html
模型参数
https://www.jianshu.com/p/1100e333fcab
for f in cate_features:
    train[f]=train[f].astype('category')
	
模型融合

针对算法本身：
个体学习器ht来自不同的模型集合
个体学习器ht来自于同一个模型集合的不同超参数，例如学习率η不同
算法本身具有随机性，例如用不同的随机种子来得到不同的模型




模型
num_boost_round=gbm.best_iteration,（https://zhuanlan.zhihu.com/p/59998657）

5. 不做任何处理（模型自动编码）

XgBoost和Random Forest，不能直接处理categorical feature，必须先编码成为numerical feature。
lightgbm和CatBoost，可以直接处理categorical feature。
lightgbm： 需要先做label encoding。用特定算法（On Grouping for Maximum Homogeneity）找到optimal split，效果优于ONE。也可以选择采用one-hot encoding，。Features - LightGBM documentation
CatBoost： 不需要先做label encoding。可以选择采用one-hot encoding，target encoding (with regularization)。CatBoost — Transforming categorical features to numerical features — Yandex Technologies


trainoff_user, testoff_user= train_test_split(df_trainuser1,random_state=2019, test_size=0.25) #这样子同一个用户的不在一起

##增量训练
    # 重点来了，通过 init_model 和 keep_training_booster 两个参数实现增量训练
    gbm =lgb.train(params,
                    lgb_train,
                    num_boost_round=1000,
                    valid_sets=lgb_eval,
                    init_model=gbm,             # 如果gbm不为None，那么就是在上次的基础上接着训练
                    feature_name=x_cols,
                    early_stopping_rounds=10,
                    verbose_eval=False,
                    keep_training_booster=True) # 增量训练 


  xgboost比sklearn自带的GBDT快，lightgbm比xgboost快，catboost最慢，但它在小数据集中效果好．一般情况下，xgboost得分一般比lightgbm要高一点．
 也没有哪个好，哪个不好，针对不同的情况选择不同的模型吧．像这种＂大数据＂的，lightgbm相对快一些．
 —

 GradientBoostingRegressor

 rom sklearn import model_selection
from sklearn.ensemble import GradientBoostingRegressor

#### 将训练集分成5份，4份用来训练模型，1份用来预测，这样就可以用不同的训练集在一个模型中训练

clf = GradientBoostingRegressor()

#### 定义特征集和结果集

print(model_selection.cross_val_score(clf, X, y, cv=5))



print('\ny_pred:',np.exp(y_pred[:10]))
print('\ny_test',np.exp(y_test[:10]))

lgb官方文档
https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRanker.html

相关性分析：特征之间并不是完全独立的，这时候可以计算correlation coefficient来确定特征之间的两两关系。还可以计算每个特征和目标值（这里是logerror）之间的相关性，绝对值越大说明这个特征的作用越大。
模型权重分析：现在大多数模型在训练完成后都会给出获取特征权重的接口，这时的权重是直接反映了一个特征在这个模型中的作用。这是判断一个特征是否有用的重要方法。例如，原始特征有卧室数量和卫生间数量，我们自己创造了一个特征房间总数（卧室数量+卫生间数量）。这时可以用这种方法判断这个新特征是否有效。


调参这事很复杂，有很多经验、方法，实际的比赛中往往还是需要一些玄学。调参最常用的方法就是GridSearch和RandomSearch。GridSearch是给定每个待调参数的几个选择，然后排列组合出所有可能性（就像网格一样），做Cross Validation，然后挑选出最好的那组参数组合。RandomSerach很类似，只是不直接给定参数的有限个取值可能，而是给出一个参数分布，从这个分布中随机采样一定个数的取值。
调参的方法理解了，那具体调什么参数呢？Zillow Prize比赛里，主要用的模型是XGBoost和LightGBM。下面列出一些主要用到的参数，更多的还是直接看文档。
XGBoost:

booster: gblinear/gbtree 基分类器用线性分类器还是决策树
max_depth: 树最大深度
learning_rate：学习率
alpha：L1正则化系数
lambda：L2正则化系数
subsample: 训练时使用样本的比例

LightGBM：

num_leaves: 叶子节点的个数
max_depth：最大深度，控制分裂的深度
learning_rate: 学习率
objective: 损失函数（mse, huber loss, fair loss等）
min_data_in_leaf: 叶子节点必须包含的最少样本数
feature_fraction: 训练时使用feature的比例
bagging_fraction: 训练时使用样本的比例
```



### 不同树的参数比较

 https://blog.csdn.net/weiyongle1996/article/details/78446244 

### 2特征处理

```python
# 特征分类
num_feat = ['regYear'] + stat_feat
cate_feat = ['adcode','bodyType','model','regMonth']

# 类别特征处理
if m_type == 'lgb':
    for i in cate_feat:
        data_df[i] = data_df[i].astype('category') #lgb有类别
elif m_type == 'xgb':
    lbl = LabelEncoder()  
    for i in tqdm(cate_feat):
        data_df[i] = lbl.fit_transform(data_df[i].astype(str))
```
### 1特征选择

```python
remove_columns = ['timestamp','site_id','building_id','DT_M',TARGET]
features_columns = [col for col in list(train_df) if col not in remove_columns]


train[ycol] = train.Quality_label.map({'Excellent':0,'Good':1,'Pass':2,'Fail':3})
dropcols=['p1' 'a12', 'qlabel', ycol]

feature_names=list(filter(lambda x:x not in dropcols,train.columns))
```



### ***\*GBDT11.5.2019\****

```python
y_pred = gbdt.predict_proba(x_test)
for y in y_pred:
y[0] 表示样本label=0的概率 y[1]表示样本label=1的概率
  new_y_pred.append(1 if y[1] > 0.5 else 0)
mse = mean_squared_error(y_test, new_y_pred)
accuracy = metrics.accuracy_score(y_test.values, new_y_pred)
auc = metrics.roc_auc_score(y_test.values, new_y_pred)
```



### ***\*各种模型参数\****

```python
svr_ = SVR(kernel='linear', degree=3, coef0=0.0, tol=0.001,
      C=1.0, epsilon=0.1, shrinking=True, cache_size=20)

lgb_ = lgb.LGBMModel(boosting_type='gbdt', num_leaves=35,
           max_depth=20, max_bin=255, learning_rate=0.03, n_estimator=10,
           subsample_for_bin=2000, objective='regression', min_split_gain=0.0,
           min_child_weight=0.001, min_child_samples=20, subsample=1.0, verbose=0,
           subsample_freq=1, colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0,
           random_state=None, n_jobs=-1, silent=True)

RF_model = RandomForestRegressor(n_estimators=50, max_depth=25, min_samples_split=20,
                 min_samples_leaf=10, max_features='sqrt', oob_score=True,
                 random_state=10)

BR_model = BayesianRidge(alpha_1=1e-06, alpha_2=1e-06, compute_score=False,
             copy_X=True, fit_intercept=True, lambda_1=1e-06,
             lambda_2=1e-06, n_iter=300,
             normalize=False, tol=0.0000001, verbose=False)

linear_model = LinearRegression()
ls = Lasso(alpha=0.00375

 
```



### LGB

### ccf11.27

```python
def Lgb_Classifier(train_x, train_y,test_x,test_y):
    import lightgbm as lgb
    lgb_trn = lgb.Dataset(
                        data=train_x,
                        label=train_y,
                        free_raw_data=True)
    lgb_val = lgb.Dataset(
                        data=test_x,
                        label=test_y,
                        free_raw_data=True)
    
    params_lgb = {'boosting_type': 'gbdt','objective': 'multiclass',
                  'num_class': 4,  'metric': 'multi_error',
                  'num_leaves': 168, 'learning_rate': 0.01, 'num_threads':-1}
    fit_params_lgb = {'num_boost_round': 1000, 'verbose_eval':50,'early_stopping_rounds':50}
    
    lgb_reg = lgb.train(params=params_lgb, 
                        train_set=lgb_trn, 
                        **fit_params_lgb,
                        valid_sets=[lgb_trn, lgb_val])
    return lgb_reg


def model_cv(train,test,esti,feature_names):
    oof = np.zeros((train.shape[0],4))
    cv_model=[]
    kfolder = KFold(n_splits=nfold, shuffle=True, random_state=2019)
    for fold_id, (train_index, test_index) in enumerate(kfolder.split(train)):
        print(f'\nFold_{fold_id} Training ================================\n')
        reg = {'LGB':Lgb_Classifier}
        train_x, train_y = train.loc[train_index, feature_names], train.loc[train_index,ycol]
        test_x, test_y = train.loc[test_index, feature_names], train.loc[test_index, ycol]
        if esti=='LGB' :
            model= reg[esti](train_x, train_y,test_x,test_y)
            oof[test_index] = model.predict(test_x,num_iteration=model.best_iteration)
            cv_model.append(model)
            if len(test):
                test.loc[:,submit.columns[1:]] += model.predict(test[feature_names],num_iteration=model.best_iteration+model.best_iteration) / nfold
    return oof,test,cv_model
```



```python
#调参 https://zhuanlan.zhihu.com/p/39782438
#https://blog.csdn.net/sunyusunyu2011/article/details/81607187
params = {
    'task': 'train',
    'boosting_type': 'gbdt',  # GBDT算法为基础
    'objective': 'binary',  # 因为要完成预测用户是否买单行为，所以是binary，不买是0，购买是1
    'metric': 'auc',  # 评判指标
    'max_bin': 255,  # 大会有更准的效果,更慢的速度
    'learning_rate': 0.1,  # 学习率
    'num_leaves': 64,  # 大会更准,但可能过拟合
    'max_depth': -1,  # 小数据集下限制最大深度可防止过拟合,小于0表示无限制
    'feature_fraction': 0.8,  # 防止过拟合
    'bagging_freq': 5,  # 防止过拟合
    'bagging_fraction': 0.8,  # 防止过拟合
    'min_data_in_leaf': 21,  # 防止过拟合
    'min_sum_hessian_in_leaf': 3.0,  # 防止过拟合
    'header': True  # 数据集是否带表头
    'header' = myFeval #自定义评估函数
}
#参数解释 https://juejin.im/post/5b76437ae51d45666b5d9b05
metric： default={l2 for regression}, {binary_logloss for binary classification}, {ndcg for lambdarank}, type=multi-enum, options=l1, l2, ndcg, auc, binary_logloss, binary_error …
1. 使用num_leaves
因为LightGBM使用的是leaf-wise的算法，因此在调节树的复杂程度时，使用的是num_leaves而不是max_depth。
大致换算关系：num_leaves = 2^(max_depth)。它的值的设置应该小于2^(max_depth)，否则可能会导致过拟合。
2.对于非平衡数据集：可以param['is_unbalance']='true’
3. Bagging参数：bagging_fraction+bagging_freq（必须同时设置）、feature_fraction。bagging_fraction可以使bagging的更快的运行出结果，feature_fraction设置在每次迭代中使用特征的比例。
4. min_data_in_leaf：这也是一个比较重要的参数，调大它的值可以防止过拟合，它的值通常设置的比较大。
5.max_bin:调小max_bin的值可以提高模型训练速度，调大它的值和调大num_leaves起到的效果类似。
————————————————
原文链接：https://blog.csdn.net/weiyongle1996/article/details/78446244

————————————————
版权声明：本文为CSDN博主「SunBUPT」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/sunyusunyu2011/article/details/81607187
model = lgb.LGBMRegressor(objective='regression', n_estimators=12000, min_child_samples=20, num_leaves=20,
             learning_rate=0.005, feature_fraction=0.8,
             subsample=0.5, n_jobs=-1, random_state=50)

\# Train the model
model.fit(train_features, train_labels, eval_metric='rmse',
     eval_set=[(valid_features, valid_labels), (train_features, train_labels)],
     eval_names=['valid', 'train'], categorical_feature=cat_indices,
     early_stopping_rounds=2000, verbose=600)

model1 = LGBMRegressor(n_estimators=1600,objective = 'regression',learning_rate = 0.01,random_state = 50,metric='rmse')

model2=XGBRegressor(n_estimators=16000,objective = 'reg:linear',learning_rate = 0.01,n_jobs = -1,random_state = 50,eval_metric='rmse',silent=True
```

### ***\*XGB\****

```python
 X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
params ={'learning_rate': 0.4,
​     'max_depth': 20,         # 构建树的深度，越大越容易过拟合
​     'num_boost_round':2000,
​     'objective': 'multi:softprob', # 多分类的问题
​     'random_state': 7,
​     'silent':0,
​     'num_class':4,         # 类别数，与 multisoftmax 并用
​     'eta':0.8            #为了防止过拟合，更新过程中用到的收缩步长。eta通过缩减特征 的权重使提升计算过程更加保守。缺省值为0.3，取值范围为：[0,1]
​    }
model = xgb.train(params,xgb.DMatrix(X_train, y_train))
y_pred=model.predict(xgb.DMatrix(X_test))
```

\# Create the model：光伏预测

```python
model = xgb.XGBRegressor(objective='reg:linear', n_estimators=16000, min_child_weight=1, num_leaves=20,
             learning_rate=0.01, max_depth=6, n_jobs=20,
             subsample=0.6, colsample_bytree=0.4, colsample_bylevel=1)
model.fit(train_features, train_labels,
     eval_set=[(valid_features, valid_labels), (train_features, train_labels)],
     early_stopping_rounds=300, verbose=600)

model.fit(train,y_train) auc(model, train, test)

xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
model.fit(xgb1, train, predictors)
```

```python
train_data = pd.read_csv('.....')
train_label = pd.read_csv('.....')
test_data = pd.read_csv('.....')

cat_list = [] #catboost中需要处理的离散特征属性列
oof = np.zeros(train_data.shape[0])  #训练集长度
prediction = np.zeros(test_data.shape[0])   #测试集长度
seeds = [2017,2018,2019,2020]   #随机种子
num_model_seed = 1
```



### ***\*CATBOOST\****

介绍： https://flashgene.com/archives/78462.html 

#文档

 https://catboost.ai/docs/concepts/python-reference_utils_eval_metric.html 

```python
train_x, test_x, train_y, test_y = train_test_split(train_data, train_label, test_size=0.2, random_state = 2019)   #拆分训练集
for model_seed in range(num_model_seed):   #选用几个随机种子
  oof_cat = np.zeros(train_data.shape[0])
  prediction_cat = np.zeros(test_data.shape[0])
  skf = StratifiedKFold(n_splits=5, random_state=seeds[model_seed], shuffle=True) #五折交叉验证，shuffle表示是否打乱数据，若为True再设定随机种子random_state
  for index, (train_index, test_index) in enumerate(skf.split(train_data, train_label)): #将数据五折分割
    train_x, test_x, train_y, test_y = train_data.iloc[train_index], test_data.iloc[test_index], train_label.iloc[train_index], train_label.iloc[test_index]
    cbt_model = cbt.CatBoostClassifier(iterations=5000, learning_rate=0.1, max_depth=7, verbose=100, early_stopping_rounds=500,task_type='GPU', eval_metric='F1',cat_features=cat_list)   #设置模型参数，verbose表示每100个训练输出打印一次
    cbt_model.fit(train_x, train_y, eval_set=(test_x, test_y)) #训练五折分割后的训练集
    gc.collect() #垃圾清理，内存清理
    oof_cat[test_index] += cbt_model.predict_proba(test_x)[:,1] #
    prediction_cat += cbt_model.predict_proba(test_data)[:,1]/5
  print('F1', f1_score(train_label, np.round(oof_cat)))
  oof += oof_cat / num_model_seed   #五折训练集取均值
  prediction += prediction_cat / num_model_seed #测试集取均值
print('score', f1_score(train_label, np.round(oof)))
```



### ***\*LTSM\****

考虑到数据的周期和时序特性，在比赛后期使用了LSTM模型。经过估计，数据的周期在200左右，我们使用了的LSTM模型中的units数目为20

 

```py
from keras.models import Sequential,load_model
from keras.layers import Dense,LSTM

def do_LSTM_Regression():


  #construct LSTM based on keras
  model = Sequential()
  model.add(LSTM(200,activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2])))
  model.add(Dense(1,activation='relu'))
  model.compile(loss='mse', optimizer='adam')
  #model.fit(X_train, y_train, epochs=1100, batch_size=100, validation_data=(X_val, y_val), verbose=2, shuffle=False)
  model=load_model('lstm_model_8805_a8485.h5')# load the history best model to recovery the best result
  #prediction
  y_val_pred=model.predict(X_val)
  y_test_pred=model.predict(X_test)

  rmse=mean_squared_error(y_val,y_val_pred)**0.5
  score=1.0/(1.0+rmse)

  y_test_pred=np.array(y_test_pred).flatten()
  ID=pd.read_csv('public.test.csv')['ID']
  save_predictions(ID,y_test
```

 

 

## ***\*5训练\****

### ***\*训练前归一化\****

### 评估函数

#### 11.27

```
#mae
def custom_fun(preds, labels):
    temp=abs( preds- labels)
    res=np.sum(temp)/(len(temp)*4)
    result=1/(1+10*res)
    return result
```

计算方法

loss_train = np.sum((**y_true** - **y_predicted**) ** 2, axis=0) / (**X_test**.shape[0])  # RMSE
loss_train = loss_train ** 0.5

####  一个要真实值，一个要平均值

```python
    data_agg = data.groupby('model').agg({
        pred:  list,
        label: [list, 'mean']
    }).reset_index()
```



### ***\*Gridsearch\****

from sklearn.cross_validation import train_test_split

from sklearn.grid_search import GridSearchCV

 

train_x,X_test, train_y, y_test =train_test_split(X_train[feature],y,test_size=0.4, random_state=0)

 

 

model = xgb.XGBClassifier()

param_dist = {"max_depth": [10,30,50],

​       "min_child_weight" : [1],

​       "n_estimators": [500,1000],

​       "learning_rate": [0.05,0.1,0.2],}

grid_search = GridSearchCV(model, param_grid=param_dist, cv = 5,

​                  verbose=10, n_jobs=-1)

grid_search.fit(train_x, train_y)

grid_search.best_estimator_

### ***\*训练测试集获取\****

Test=data[data['label'].isnull()]

Train=data[data['label'].notnull()]

### 1交叉训练

#### 11.27

```python
#线下草稿，交叉验证5折

from sklearn.model_selection import train_test_split,KFold
n_estimators=默认10
lgb_model = lgb.LGBMRegressor( #林有夕
    num_leaves=150, reg_alpha=0., reg_lambda=0.01, objective='regression', metric='rmse',
    max_depth=-1, learning_rate=0.05, min_child_samples=100, n_jobs=-1,
    n_estimators=25, subsample=0.7, colsample_bytree=0.8, subsample_freq=1, random_state=2019
)

#交叉验证,n_estimators=800和n_splits=5

lgb_model.random_state = 2018
n_splits=2
kfold = KFold(n_splits, shuffle=True, random_state=lgb_model.random_state)
label='label'
data=dataall
#得到组合特征后的数据分为训练测试集
data['pre']=0
test_index =  data[label] == -1
train_data = data[~test_index]#(638157  
test_data = data[test_index]#(27470, 39)

submit=[]
#trainoff_user, testoff_user= train_test_split(df_trainuser1,random_state=2019, test_size=0.25) #这样子同一个用户的不在一起

#结合之后新特征生成怎么办不能这样写，要用五折生成id，然后从整个特征表中抽取有这个id的行） 
for trainoff_user_idx, testoff_user_idx in kfold.split(df_trainuser1):  
    #获取训练用户数据和验证用户数据
    trainuser=set(df_trainuser1.loc[trainoff_user_idx]['user_id']) #不要重复计算
    trainoff= train_data[train_data['user_id'].apply(lambda x:x in trainuser)]#取出datall的userid在训练五折之一的userid中的数据，set或者相等
    valuser=set(df_trainuser1.loc[testoff_user_idx]['user_id']) 
    valoff= train_data[train_data['user_id'].apply(lambda x:x in valuser)] #用训练数据构造验证集合
    #训练验证集
    lgb_model.random_state = lgb_model.random_state + 1                                                    
    train_x=trainoff[feature]
    train_y=trainoff[label].values
    val_x=valoff[feature]
    val_y=valoff[label].values
    lgb_model.fit(train_x, train_y, eval_set=[(val_x, val_y)],categorical_feature=cate_feature)

    #预测测试集，如果是时间问题还要所有训练+时间来一遍？？看一下腾讯，这个不用--时间不用重新预测吗??前三天的规则，前面模型
    trainoff['pre'] = lgb_model.predict(train_x)#每个fold预测的train是不一样的
    valoff['pre']   = lgb_model.predict(val_x)#验证
    test_data['pre'] += lgb_model.predict(test_data[feature])#输出

#取平均                  
test_data['pre'] = test_data['pre'] / n_splits

return pd.concat([train_data, test_data], ignore_index=True), predict_label

test_data['pre'].describe()

map_train = cal_map(trainoff) #也可以放在里面看每次交叉结果,线下验证结果,次数过少过拟合
trainoff = cal_map(valoff) #也可以放在里面看每次交叉结果,线下验证结果

valoff['pre'].describe()

valoff[['browsed','delivered','satisfied','pre']][1:50]

test_data['pre']
```

![image-20191118193624217](D:/ruanjiandata/Typora_mdpic/image-20191118193624217.png)

### ***\*训练集划分\****

***\*四种训练集划分\****

针对此问题，他们根据对数据的分析、特征的构建、以及对实际场景的思考，提出了四种训练集划分：

\1. 全量统计特征提取第七天特征——all-to-7

\2. 2. 全量数据的抽样统计——sample

\3. 3. 单独第七天的特征提取——only7

\4. 4. 全量数据——all

构造四种训练集划分的目的如下： 

1）构造出训练集中的差异性，方便模型融合 

2）在每组训练集中，对高维特征进行选择，选择后进行特征分组

####  me

```python
#线上线下
if LOCAl_TEST:
    tr_data = lgb.Dataset(train_df.iloc[:1500][features_columns], label=np.log1p(train_df.iloc[:1500][TARGET]))
    vl_data = lgb.Dataset(train_df.iloc[1500:][features_columns], label=np.log1p(train_df.iloc[1500:][TARGET]))
    eval_sets = [tr_data,vl_data]
else:
    tr_data = lgb.Dataset(train_df[features_columns], label=np.log1p(train_df[TARGET]))
    eval_sets = [tr_data]
```



### ***\*Trick\****

### ***\*数据泄露\****

试集只有shop_id和item_id，且行数少于训练集行数，考虑模型只训练测试集所包含的（shop_id,item_id）对，其他的匹配对不予以考虑

\# 数据泄漏

test_shop_ids = test['shop_id'].unique()

test_item_ids = test['item_id'].unique()

\#获取要预测的shop_id+item_id的训练数据

lk_train = train[train['shop_id'].isin(test_shop_ids)]

lk_train = lk_train[lk_train['item_id'].isin(test_item_ids)]

 

train_monthly = lk_train[['date', 'date_block_num', 'shop_id', 'item_category_id', 'item_id', 'item_price', 'item_cnt_day']]

## 6模型融合

```
#rmse越小的值评分影响越大，算法会导致更大的误差
a*pred1+（1-a）*pred2  #集合平均
(pred1**a)*(pred2**(1-a)) #算数平均
这个操作也是最终分数有近一个千的提升。在之前的比赛也使用过这种方法，非常值得借鉴。在最近的“全国高校新能源创新大赛”中的也依然适用。
```



#### 5.融合方式11.26

融合方式也可以有很多，stacking和blending，这里只选择了blending，并尝试了两者加权方式，算术平均和几何平均。****

#### 之前

https://www.cnblogs.com/nxf-rabbit75/p/10923549.html

把预测出来的结果作为新特征

![img](file:///C:\Users\sereny\AppData\Local\Temp\ksohtml103712\wps4.jpg) 

## ***\*7提交\****

save_predictions(ID,y_test_pred,'result_lstm.csv')

 

predictions=pd.DataFrame(list(zip(map(int,**IDs**),**predictions**)))
**predictions**.to_csv(**name**,header=**False**,index=**False**,sep=',')



#### 2list写到文件

```python
print('生成提交结果文件')
with open('submission.csv', 'w') as f:
    f.write('ID,Prediction\n')
    for id, pred in preds_list:
        f.write('{},{}\n'.format(id, pred))
```

 

# ***\*比赛分类\****

## ***\*时间序列比赛\****

季节效应

https://mp.weixin.qq.com/s?__biz=MzU1Nzc1NjI0Nw==&mid=2247484533&idx=1&sn=8cbc7576d01fc404a57306116910aa38

***\*差分平稳\****

根据所建模型预测未来时间序列的趋势曲线，常见模型包括ARMA，VAR，TAR，ARCH等。

基于机器学习的时间预测方法一般适用于多维时间序列分析，像KNN/SVM等，而SVR/RNN/LSTM/GRU等序列分析方法，也可以用于单维时间序列分析。

时序不稳定做差分，有代码

## ***\*销量预测\****

![img](file:///C:\Users\sereny\AppData\Local\Temp\ksohtml103712\wps5.jpg) 

![img](file:///C:\Users\sereny\AppData\Local\Temp\ksohtml103712\wps6.jpg) 

 

规律

价格太高，销量越小

https://blog.csdn.net/wlx19970505/article/details/101014813

### ***\*0方法\****

最奇葩的地方在于要利用产品23个月的历史销量来预测未来14个月的销量。当时的感觉就是这要是用模型的话，我想到有两种：

建立一个模型，只预测未来一个月，然后级联的预测未来14个月，这样误差会累积，感觉十分不靠谱没有做。

针对预测的每一个月建立一个模型，认为未来的14个月间隔时间太长，还是觉得不靠谱。

————————————————

### ***\*#\*******\*1\*******\*按照月份，商店和商品来计算出销量和价格的总值和均值:shop_id=province/item_id=model--本来都是每天的\****

 

df = df.sort_values('date').groupby(['date_block_num', 'shop_id', 'item_category_id', 'item_id'], as_index=False)

 

df = df.agg({'item_price':['sum', 'mean'], 'item_cnt_day':['sum', 'mean','count']})

\# Rename features.

df.columns = ['date_block_num', 'shop_id', 'item_category_id', 'item_id', 'item_price', 'mean_item_price', 'item_cnt', 'mean_item_cnt', 'transactions']

 

\# Rename features.

df.columns = ['date_block_num', 'shop_id', 'item_category_id', 'item_id', 'item_price', 'mean_item_price', 'item_cnt', 'mean_item_cnt', 'transactions']

df.head().append(df.tail())

 

### ***\*2构造数据\****

考虑到测试集可能会有不同的商店和商品的组合，这里我们对训练数据按照shop_id和item_id的组合进行扩充，缺失数据进行零填充，同时构造出具体的年，月信息：

### ***\*3特征\*******\* \****https://blog.csdn.net/s09094031/article/details/90347191

3.2 历史信息

将产生的信息做了融合。需要通过延迟操作来产生一些历史信息。比如可以将第0-33个月的销量作为第1-34个月的历史特征（延迟一个月）。按照以下说明，一共产生了15种特征。

 

每个商品-商店组合每个月销量的历史信息，分别延迟[1,2,3,6,12]个月。这应该是最符合直觉的一种操作。

所有商品-商店组合每个月销量均值的历史信息，分别延迟[1,2,3,6,12]个月。

每件商品每个月销量均值的历史信息，分别延迟[1,2,3,6,12]个月。

每个商店每个月销量均值的历史信息，分别延迟[1,2,3,6,12]个月。

每个商品类别每个月销量均值的历史信息，分别延迟[1,2,3,6,12]个月。

每个商品类别-商店每个月销量均值的历史信息，分别延迟[1,2,3,6,12]个月。

以上六种延迟都比较直观，直接针对商品，商店，商品类别。但是销量的变化趋势还可能与商品类别_大类，商店_城市，商品价格，每个月的天数有关，还需要做以下统计和延迟。可以根据模型输出的feature importance来选择和调整这些特征。

 

每个商品类别_大类每个月销量均值的历史信息，分别延迟[1,2,3,6,12]个月。

每个商店_城市每个月销量均值的历史信息，分别延迟[1,2,3,6,12]个月。

每个商品-商店_城市组合每个月销量均值的历史信息，分别延迟[1,2,3,6,12]个月。

除了以上组合之外，还有以下特征可能有用

 

每个商品第一次的销量

每个商品最后一次的销量

每个商品_商店组合第一次的销量

每个商品_商店组合最后一次的销量

每个商品的价格变化

每个月的天数

### ***\*4空值填充-预测为负数\****

  \# 将所有商品每个月的销量转化为一个37*4001的数组供之后处理负数预测值的时候使用

  product_quantity['product_date'] = product_quantity['product_date'].str[:7]

  quantity = product_quantity.groupby(['product_id', 'product_date']).sum()

  quantity.reset_index(inplace=True)

 

  quantity_arr = np.full((37, 4001), -1, dtype=np.int32)

  for idx in quantity.index:

​    pid = quantity.loc[idx, 'product_id']

​    date = quantity.loc[idx, 'product_date']

​    quantity_arr[(int(date[:4]) - 2014) * 12 + int(date[5:7]) -

​           1][pid] = quantity.loc[idx, 'ciiquantity']

 

  train_y = quantity['ciiquantity'].tolist()

​	

​	# 23个月均无数据的product_id

  invalid_pid = [i for i in range(1, 4001) if quantity_arr[:23, i].max() < 0]

  \# 将负数的预测改为前23个月有效的最小值,若无则为0

  history_min = [0] * 4001

  for i in range(1, 4001):

​    quantity_i = quantity_arr[:23, i]

​    if quantity_i.max() < 0:

​      history_min[i] = 0

​    else:

​      history_min[i] = quantity_i[quantity_i > -1].min()

### ***\*5后面无销量的设置为0\****

| for pred_y in ans:                                           |
| ------------------------------------------------------------ |
| out['ciiquantity_month'] = pred_y                            |
| # 将23个月均无数据且startdate,cooperatedate均在第24个月以后的清0 |
| for pid in invalid_pid:                                      |
| product_info_i = product_info.loc[pid]                       |
| dat = min(product_info_i['startdate'],                       |
| product_info_i['cooperatedate']) - 191 + 23                  |
| for j in range(23, dat.astype(np.int)):                      |
| out.loc[(j - 23) * 4000 + pid - 1, 'ciiquantity_month'] = 0  |
|                                                              |
| idx = out['ciiquantity_month'] < 0                           |
| out.loc[idx, 'ciiquantity_month'] = np.array(                |
| history_min)[out.loc[idx, 'product_id']]                     |
| final_out['ciiquantity_month'] += out['ciiquantity_month']   |
| out['ciiquantity_month'] = final_out['ciiquantity_month'] / len(ans) |

 

# 过程不一定经常

## 12.10practicalAI-master

### 树

DecisionTreeClassifier+RandomForestClassifier

```python
# DecisionTreeClassifier--- model
dtree = DecisionTreeClassifier(criterion="entropy", random_state=args.seed, 
                               max_depth=args.max_depth, 
                               min_samples_leaf=args.min_samples_leaf)
dtree.fit(X_train, y_train)
pred_train = dtree.predict(X_train)
#2
# Initialize Random forest
forest = RandomForestClassifier(
    n_estimators=args.n_estimators, criterion="entropy", 
    max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf)
# Train
forest.fit(X_train, y_train)
pred_test = forest.predict(X_test)
```



### 评估

```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# Accuracy
train_acc = accuracy_score(y_train, pred_train)
test_acc = accuracy_score(y_test, pred_test)
print ("train acc: {0:.2f}, test acc: {1:.2f}".format(train_acc, test_acc))
#train acc: 0.82, test acc: 0.70
# Calculate other evaluation metrics 
precision, recall, F1, _ = precision_recall_fscore_support(y_test, pred_test, average="binary")
print ("precision: {0:.2f}. recall: {1:.2f}, F1: {2:.2f}".format(precision, recall, F1))
#precision: 0.70. recall: 0.79, F1: 0.75

#2
# Feature importances
features = list(X_test.columns)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
indices = np.argsort(importances)[::-1]
num_features = len(importances)

# Plot the feature importances of the tree
plt.figure()
plt.title("Feature importances")
plt.bar(range(num_features), importances[indices], yerr=std[indices], 
        color="g", align="center")
plt.xticks(range(num_features), [features[i] for i in indices], rotation='45')
plt.xlim([-1, num_features])
plt.show()

# Print values
for i in indices:
    print ("{0} - {1:.3f}".format(features[i], importances[i]))
```

### Grid Search

```python
from sklearn.model_selection import GridSearchCV
# Create the parameter grid 
param_grid = {
    'bootstrap': [True],
    'max_depth': [10, 20, 50],
    'max_features': [len(features)],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [4, 8],
    'n_estimators': [5, 10, 50] # of trees
}
# Initialize random forest
forest = RandomForestClassifier()
# Instantiate grid search
grid_search = GridSearchCV(estimator=forest, param_grid=param_grid, cv=3, 
                           n_jobs=-1, verbose=1)
# Fit grid search to the data
grid_search.fit(X_train, y_train)
# See the best combination of parameters
grid_search.best_params_
# Train using best parameters
best_forest = grid_search.best_estimator_# See the best combination of parameters
best_forest.fit(X_train, y_train)
# Predictions
pred_train = best_forest.predict(X_train)
pred_test = best_forest.predict(X_test)

# Accuracy
train_acc = accuracy_score(y_train, pred_train)
test_acc = accuracy_score(y_test, pred_test)
print ("train acc: {0:.2f}, test acc: {1:.2f}".format(train_acc, test_acc))

# Calculate other evaluation metrics 
precision, recall, F1, _ = precision_recall_fscore_support(y_test, pred_test, average="binary")
print ("precision: {0:.2f}. recall: {1:.2f}, F1: {2:.2f}".format(precision, recall, F1))
```



### pytorch

#### cuda

```python

# Check CUDA
if not torch.cuda.is_available():
    args.cuda = False
args.device = torch.device("cuda" if args.cuda else "cpu")
print("Using CUDA: {}".format(args.cuda))
```



```python
# Creating a zero tensor
x = torch.Tensor(3, 4)
print ("x[:1, 1:3]: \n{}".format(x[:1]))
print ("x[:1, 1:3]: \n{}".format(x[:1, 1:3]))
x = torch.Tensor([[1, 2, 3],[4, 5, 6]])

print("Type: {}".format(x.type())) #torch.FloatTensor
print("Size: {}".format(x.shape)) #torch.Size([3, 4])
print("Values: \n{}".format(x))#3*4
x = torch.randn(2, 3) # normal distribution (rand(2,3) -> uniform distribution)二维数组
# Zero and Ones tensor
x = torch.zeros(2, 3)
x = torch.ones(2, 3)
x = torch.from_numpy(np.random.rand(2, 3))
x = x.long()
y = torch.t(x)# Transpose
z = x.view(3, 2)# Reshape

####
# Dangers of reshaping (unintended consequences)
x = torch.tensor([
    [[1,1,1,1], [2,2,2,2], [3,3,3,3]],
    [[10,10,10,10], [20,20,20,20], [30,30,30,30]]
])
a = x.view(x.size(1), -1)
b = x.transpose(0,1).contiguous()
c = b.view(b.size(0), -1)
# Dimensional operations
x = torch.randn(2, 3)
y = torch.sum(x, dim=0) #dim=1    add each row's value for every column
##
# Select with dimensional indicies
x = torch.randn(2, 3)
col_indices = torch.LongTensor([0, 2]) #选择第0 2列
chosen = torch.index_select(x, dim=1, index=col_indices) # values from column 0 & 2
print("Values: \n{}".format(chosen)) 

row_indices = torch.LongTensor([0, 1])
chosen = x[row_indices, col_indices] # values from (0, 0) & (2, 1)
print("Values: \n{}".format(chosen)) 


y = torch.cat([x, x], dim=0) # stack by rows (dim=1 to stack by columns)重复
##Gradient
# Tensors with gradient bookkeeping
x = torch.rand(3, 4, requires_grad=True)
y = 3*x + 2
z = y.mean()
z.backward() # z has to be scalar
print("Values: \n{}".format(x))
print("x.grad: \n", x.grad)

#cuda
# Is CUDA available?
print (torch.cuda.is_available())
# Creating a zero tensor
x = torch.Tensor(3, 4).to("cpu")
print("Type: {}".format(x.type()))
# If the code above return False, then go to Runtime → Change runtime type and select GPU under Hardware accelerator.
# Creating a zero tensor
x = torch.Tensor(3, 4).to("cuda")
print("Type: {}".format(x.type()))
```

### MLP

![image-20191210222823880](D:/ruanjiandata/Typora_mdpic/image-20191210222823880.png)

```python
# Convert to PyTorch tensors
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).long()
# Shuffle data
shuffle_indicies = torch.LongTensor(random.sample(range(0, len(X)), len(X)))
X = X[shuffle_indicies]


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm_notebook

# Linear model
class LogisticClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LogisticClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_in, apply_softmax=False):
        a_1 = self.fc1(x_in)
        y_pred = self.fc2(a_1)

        if apply_softmax:
            y_pred = F.softmax(y_pred, dim=1)

        return y_pred
# Initialize model
model = LogisticClassifier(input_dim=args.dimensions, 
                           hidden_dim=args.num_hidden_units, 
                           output_dim=args.num_classes)
print (model.named_modules)

# Optimization
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate) # Adam optimizer (usually better than SGD)
# Accuracy
def get_accuracy(y_pred, y_target):
    n_correct = torch.eq(y_pred, y_target).sum().item()
    accuracy = n_correct / len(y_pred) * 100
    return accuracy
# Training
for t in range(args.num_epochs):
    # Forward pass
    y_pred = model(X_train)
    
    # Accuracy
    _, predictions = y_pred.max(dim=1)
    accuracy = get_accuracy(y_pred=predictions.long(), y_target=y_train)

    # Loss
    loss = loss_fn(y_pred, y_train)
    
    # Verbose
    if t%20==0: 
        print ("epoch: {0:02d} | loss: {1:.4f} | acc: {2:.1f}%".format(t, loss, accuracy))

    # Zero all gradients
    optimizer.zero_grad()

    # Backward pass
    loss.backward()

    # Update weights
    optimizer.step()
# Predictions
_, pred_train = model(X_train, apply_softmax=True).max(dim=1)
_, pred_test = model(X_test, apply_softmax=True).max(dim=1)

# Train and test accuracies
train_acc = get_accuracy(y_pred=pred_train, y_target=y_train)
test_acc = get_accuracy(y_pred=pred_test, y_target=y_test)
print ("train acc: {0:.1f}%, test acc: {1:.1f}%".format(train_acc, test_acc))
```

### mlp2

```python
# Multilayer Perceptron 
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_in, apply_softmax=False):
        a_1 = F.relu(self.fc1(x_in)) # activaton function added!
        y_pred = self.fc2(a_1)

        if apply_softmax:
            y_pred = F.softmax(y_pred, dim=1)

        return y_pred
       # Initialize model
model = MLP(input_dim=len(df.columns)-1, 
            hidden_dim=args.num_hidden_units, 
            output_dim=len(set(df.tumor)))
# Optimization
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate) 
```

### 10object_oriented_ML

```python
import os
import json
import numpy as np
import time
import torch
import uuid
def set_seeds(seed, cuda):
def generate_unique_id():
def create_dirs(dirpath):
def check_cuda(cuda):
# Set seeds for reproducability
set_seeds(seed=config["seed"], cuda=config["cuda"])
# Generate unique experiment ID
config["experiment_id"] = generate_unique_id()
# Create experiment directory
config["save_dir"] = os.path.join(config["save_dir"], config["experiment_id"])
create_dirs(dirpath=config["save_dir"])
# Expand file paths to store components later
config["vectorizer_file"] = os.path.join(config["save_dir"], config["vectorizer_file"])
config["model_file"] = os.path.join(config["save_dir"], config["model_file"])
# Save config
config_fp = os.path.join(config["save_dir"], "config.json")
with open(config_fp, "w") as fp:
    json.dump(config, fp)
# Check CUDA
config["device"] = check_cuda(cuda=config["cuda"])
```

### 11cnn的pytorch知识

https://github.com/practicalAI/practicalAI/blob/master/notebooks/10_Convolutional_Neural_Networks.ipynb

#### model

```python
class SurnameModel(nn.Module):
    def __init__(self, num_input_channels, num_output_channels, num_classes, dropout_p):
        super(SurnameModel, self).__init__()
        
        # Conv weights
        self.conv = nn.ModuleList([nn.Conv1d(num_input_channels, num_output_channels,  kernel_size=f) for f in [2,3,4]])
        self.dropout = nn.Dropout(dropout_p)
       
        # FC weights
        self.fc1 = nn.Linear(num_output_channels*3, num_classes)

    def forward(self, x, channel_first=False, apply_softmax=False):
        
        # Rearrange input so num_input_channels is in dim 1 (N, C, L)
        if not channel_first:
            x = x.transpose(1, 2)
            
        # Conv outputs：conv+pool+relu
        z = [conv(x) for conv in self.conv]
        z = [F.max_pool1d(zz, zz.size(2)).squeeze(2) for zz in z]
        z = [F.relu(zz) for zz in z]
        
        # Concat conv outputs
        z = torch.cat(z, 1)
        z = self.dropout(z)

        # FC layer
        y_pred = self.fc1(z)
        
        if apply_softmax:
            y_pred = F.softmax(y_pred, dim=1)
        return y_pred
```

#### 训练

```python
  def plot_performance(self):
        # Figure size
        plt.figure(figsize=(15,5))

        # Plot Loss
        plt.subplot(1, 2, 1)
        plt.title("Loss")
        plt.plot(trainer.train_state["train_loss"], label="train")
        plt.plot(trainer.train_state["val_loss"], label="val")
        plt.legend(loc='upper right')

        # Plot Accuracy
        plt.subplot(1, 2, 2)
        plt.title("Accuracy")
        plt.plot(trainer.train_state["train_acc"], label="train")
        plt.plot(trainer.train_state["val_acc"], label="val")
        plt.legend(loc='lower right')

        # Save figure
        plt.savefig(os.path.join(self.save_dir, "performance.png"))

        # Show plots
        plt.show()
    
```



### 12Embeddings

```python
args = Namespace(
    seed=1234,
    data_file="harrypotter.txt",
    embedding_dim=100,
    window=5,
    min_count=3,
    skip_gram=1, # 0 = CBOW
    negative_sampling=20,
)
# Preprocessing
def preprocess_text(text):
    text = ' '.join(word.lower() for word in text.split(" "))
    text = re.sub(r"([.,!?])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
    text = text.strip()
    return text
sentences = [preprocess_text(sentence) for sentence in sentences]
sentences = [sentence.split(" ") for sentence in sentences]
model = Word2Vec(sentences=sentences, size=args.embedding_dim, 
                 window=args.window, min_count=args.min_count, 
                 sg=args.skip_gram, negative=args.negative_sampling)
model.wv.get_vector("potter")
model.wv.most_similar(positive="scar", topn=5)
# Save the weights 
model.wv.save_word2vec_format('model.txt', binary=False)
```

#### Pretrained embeddings

```python
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from io import BytesIO
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from zipfile import ZipFile
from urllib.request import urlopen

# Unzip the file (may take ~3 minutes)
resp = urlopen('http://nlp.stanford.edu/data/glove.6B.zip')
zipfile = ZipFile(BytesIO(resp.read()))
zipfile.namelist()
['glove.6B.50d.txt',
 'glove.6B.100d.txt',
 'glove.6B.200d.txt',
 'glove.6B.300d.txt']

zipfile.extract(embeddings_file)#'/content/glove.6B.100d.txt'
glove2word2vec(embeddings_file, word2vec_output_file)#word2vec_output_file：'/content/glove.6B.100d.txt.word2vec'
```

### 13RNN

```python
class Encoder(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, rnn_hidden_dim, 
                 num_layers, bidirectional, padding_idx=0):
        super(Encoder, self).__init__()
        
        # Embeddings
        self.word_embeddings = nn.Embedding(embedding_dim=embedding_dim,
                                            num_embeddings=num_embeddings,
                                            padding_idx=padding_idx)
        
        # GRU weights
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=rnn_hidden_dim, 
                          num_layers=num_layers, batch_first=True, 
                          bidirectional=bidirectional)

    def forward(self, x_in, x_lengths):
        # Word level embeddings
        z_word = self.word_embeddings(x_in)
        # Feed into RNN
        out, h_n = self.gru(z)     
        # Gather the last relevant hidden state
        out = gather_last_relevant_hidden(out, x_lengths)    
        return out
    
class Decoder(nn.Module):
    def __init__(self, rnn_hidden_dim, hidden_dim, output_dim, dropout_p):
        super(Decoder, self).__init__()
        
        # FC weights
        self.dropout = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(rnn_hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, encoder_output, apply_softmax=False):
        
        # FC layers
        z = self.dropout(encoder_output)
        z = self.fc1(z)
        z = self.dropout(z)
        y_pred = self.fc2(z)

        if apply_softmax:
            y_pred = F.softmax(y_pred, dim=1)
        return y_pred
    
class Model(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, rnn_hidden_dim,  hidden_dim, num_layers, bidirectional, output_dim, dropout_p, padding_idx=0):
        super(Model, self).__init__()
        self.encoder = Encoder(embedding_dim, num_embeddings, rnn_hidden_dim, 
                               num_layers, bidirectional, padding_idx=0)
        self.decoder = Decoder(rnn_hidden_dim, hidden_dim, output_dim, dropout_p)
        
    def forward(self, x_in, x_lengths, apply_softmax=False):
        encoder_outputs = self.encoder(x_in, x_lengths)
        y_pred = self.decoder(encoder_outputs, apply_softmax)
        return y_pred
   
model = Model(embedding_dim=args.embedding_dim, num_embeddings=1000, 
              rnn_hidden_dim=args.rnn_hidden_dim, hidden_dim=args.hidden_dim, 
              num_layers=args.num_layers, bidirectional=args.bidirectional, 
              output_dim=4, dropout_p=args.dropout, padding_idx=0)
print (model.named_parameters)
```



### 14计算机视觉

#### Transfer learning

```python
d
```



## 数据处理

### 1填充 knn 数据

from fancyimpute import KNN
data_x =pd.DataFrame(KNN(k=6).fit_transform(train_data_x), columns=features)



# 工具

### 正则

```python
mport re
#用户也可以在此进行自定义过滤字符
'''\\\可以过滤掉反向单杠和双杠，/可以过滤掉正向单杠和双杠，第一个中括号里放的是英文符号，
第二个中括号里放的是中文符号，第二个中括号前不能少|，否则过滤不完全'''
r1 = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
r2 = u'\s+;'
for i in range(numrows):
	tem=re.sub(r1, '', rows[i][3]) #过滤内容中的各种标点符号
	cur.execute("INSERT INTO new(uid,mid,time,content) VALUES(%s, %s, %s, %s)", (rows[i][0], rows[i][1], rows[i][2], tem))
```



### 种子\使用

```python
########################### Helpers
#################################################################################
## Seeder
# :seed to make all processes deterministic     # type: int
def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    
## Simple "Memory profilers" to see memory usage
def get_memory_usage():
    return np.round(psutil.Process(os.getpid()).memory_info()[0]/2.**30, 2) 
        
def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


SEED = 42
LOCAl_TEST = False
seed_everything(SEED)
```

### ***\*输入输出工具\****

https://blog.csdn.net/pipisorry/article/details/52208727

 

输出格式控制

pandas dataframe数据全部输出，数据太多也不用省略号表示。

 

pd.set_option('display.max_columns',None)

或者

 

with option_context('display.max_rows', 10, 'display.max_columns', 5):

数据输入输出



 

 

 

#### ***\*CSV\****

通常来说，数据是CSV格式，就算不是，至少也可以转换成CSV格式。

 

读取csv文件 read_csv

 

lines = pd.read_csv(checkin_filename, sep='\t', header=None,names=col_names, parse_dates=[1], skip_blank_lines=True, index_col=0).reset_index()

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')

data = pd.read_csv('AirPassengers.csv', parse_dates='Month', index_col='Month',date_parser=dateparse)

df = pd.read_csv(‘’, header=0, sep='\t', converters={'Time(GMT)': dateParse})

data_df.to_csv(path,index=False, sep='\t', header=False, encoding='utf-8')

#### ***\*HDF5\****

df.to_hdf('foo.h5','df')

pd.read_hdf('foo.h5','df')

#### ***\*构造特征名\****

para_feat = ['Parameter{0}'.format(i) for i in range(1, 11)]

或者用for循环

### ***\*进度条\****

from tqdm import tqdm

 

for item in tqdm(log_df[['request_id','request_timestamp','uid','position','aid_info']].values,total=len(log_df)):

 

### ***\*2.减少内存的函数(kaggle上)\****

 其他: https://medium.com/@vincentteyssier/optimizing-the-size-of-a-pandas-dataframe-for-low-memory-environment-5f07db3d72e 

```python
def reduce_mem_usage(df, verbose=True):
    start_mem = df.memory_usage().sum() / 1024**2 
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
#                 if not np.isfinite(df[col]).all(): # Integer does not support NA,
#                     continue
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased from {:5.2f} to {:5.2f} Mb ({:.1f}% reduction)'.format(start_mem,end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
```

```python

```



### ***\*减少内存2.1\****

def downcast_dtypes(df):

  cols_float64 = [c for c in df if df[c].dtype == 'float64']

  cols_int64_32 = [c for c in df if df[c].dtype in ['int64', 'int32']]

  df[cols_float64] = df[cols_float64].astype(np.float32)

  df[cols_int64_32] = df[cols_int64_32].astype(np.int16)

  return df

 

# 杂

### series序列

 [i for i in dir(pd.Series.str) if not i.startswith('_')]#str属性
addr.str.count(r'\d') #Series.str.count：对Series中所有字符串的个数进行计数；

dt对象的使用：Series数据类型：datetime

>>> daterng = pd.Series(pd.date_range('2017', periods=9, freq='Q'))
>>> daterng.dt.day_name() #星期



### rsuffix

train=sales.join(items, on='item_id',rsuffix='_').drop(['item_id_', 'shop_id_', 'item_category_id_'], axis=1)#左右合并

\#append  #rsuffix是用来标记重复的column列，合并之后会被drop掉

上下合并

### log变换

```py
df[attr_feat] = np.log1p(df[attr_feat])
y=df[attr_feat]
import numpy as np
Y = np.log1p(Y)
back = np.expm1(Y)
```

### 其他

for i in prob_cols:

  sub[i] = sub.groupby('Group')[i].transform('mean')

sub = sub.groupby('Group')[labels].mean().reset_index()

 

**import** time
**import** lightgbm **as** lgb
**from** sklearn.metrics **import** mean_squared_error
**from** sklearn.model_selection **import** GridSearchCV
**from** sklearn.datasets **import** load_iris
**from** sklearn.model_selection **import** train_test_split

\# 加载数据
iris = load_iris()
data = iris.data
target = iris.target
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)
start=time.time()
\# 创建模型，训练模型
gbm = lgb.LGBMRegressor(objective='regression', num_leaves=31, learning_rate=0.05, n_estimators=20,device='gpu') #
gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='l1', early_stopping_rounds=5)

\# 测试机预测
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)

\# 模型评估

print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)

\# feature importances
print('Feature importances:', list(gbm.feature_importances_))

\# 网格搜索，参数优化
estimator = lgb.LGBMRegressor(num_leaves=31)
param_grid = {
  'learning_rate': [0.01, 0.1, 1],
  'n_estimators': [20, 40]
}
gbm = GridSearchCV(estimator, param_grid)
gbm.fit(X_train, y_train)
end=time.time()
print('耗费时间'+str(end-start)+'s')
print('Best parameters found by grid search are:', gbm.best_params_)

### ***\*2\****

ad_static.describe()  查看

ad_static.dtypes

 

(ad_static==0).astype(int).sum(axis=0) #查看0的个数

df.ix[:,~((df==1).all()|(df==0).all())] # 删除包含0和1的列

df = df[df.line_race != 0] #删除0行

 

num = df.isna().sum()  # 按列统计nan值个数

***\*COUNT 非NA数量\****

oldad['imp']=oldad.apply(**lambda** x:(df[df['adId']== x.adId]['imp']),axis=1)

val.insert(0,'pre_imp',prediction)  #val['pre_imp'] = val_prediction  # ['adId','Adbid','pre_imp]

\# 调整后和原来的连接有参考单调性代码
val.sort_values(by=["adId", "Adbid"], inplace=**True**)  # True返回的是排序原来也变化
standard = val.groupby(by='adId')[['Adbid', 'pre_imp']].mean()  # 选取均值出价作为基准

**if** len(aids)==0:
  aids.append([int(line[0]),0,"NaN","NaN"])

 

 

Nunique ：获取长度

### ***\*数据统计-计算平均/nunique出价-group后merge\****

tmp = pd.DataFrame(**train_df**.groupby(['aid','request_day']).size()).reset_index()
tmp_1 = pd.DataFrame(**train_df**.groupby(['aid','request_day'])['bid'].mean()).reset_index()
tmp.columns=['aid','request_day','imp']
**del** **train_df**['bid']
tmp_1.columns=['aid','request_day','bid']
train_df=**train_df**.drop_duplicates(['aid','request_day'])
train_df=**train_df**.merge(tmp,on=['aid','request_day'],how='left')
train_df=**train_df**.merge(tmp_1,on=['aid','request_day'],how='left')

 

### ***\*排序\****

Df.sort_values(by=['aid','op_time'])

### ***\*腾讯赛\****

### ***\*抽取部分数据\****

op_df=op_df.sample(frac=0.3)

### ***\*，连接的数据转换\****

df[f]=df[f].apply(**lambda** **x**:' '.join([str(int(float(y))) **for** y **in** str(x).split(',')]))  

 

2

**for** item **in** df[f].values:
  **try**:
    items.append(int(item))
  **except**:
    items.append(-1)

### ***\*Df.query\****

Cmean = df['C'].mean() #6.0 result1 = df[(df.A < Cmean) & (df.B < Cmean)] result1 = df.query('A < @Cmean and B < @Cmean')#等价

 

### ***\*pivot_table\****

pivot_table()和groupby()的用途类似，但更加灵活，可以对columns做更多处理。

查看每件商品每个月的销量

sales_by_item_id = sales_train.pivot_table(index=['item_id'],values=['item_cnt_day'], columns='date_block_num', aggfunc=np.sum, fill_value=0).reset_index()

sales_by_item_id.columns = sales_by_item_id.columns.droplevel().map(str)

sales_by_item_id = sales_by_item_id.reset_index(drop=True).rename_axis(None, axis=1)

sales_by_item_id.columns.values[0] = 'item_id'

### ***\*模型分类预测后处理数据\****

print('logloss',log_loss(pd.get_dummies(y).values, oof))

print('ac',accuracy_score(y, np.argmax(oof,axis=1)))

print('mae',1/(1 + np.sum(np.absolute(np.eye(4)[y] - oof))/480))

## ***\*功能\****

### ***\*时间序列--最近几个月总和为 0\****

outdated_items = train[train.loc[:,'27':].sum(axis=1)==0]
print('Outdated items in test set:', test[test['item_id'].isin(outdated_items['item_id'])]['item_id'].nunique())

### ***\*时间序列trick\****

https://blog.csdn.net/s09094031/article/details/90347191

只训练要预测的shop+item

连续几个月没有销量：将这些商品的销量大胆地设置为零。

Tips：新商店，可以直接用第33个月来预测34个月的销量，因为它没有任何历史数据。而已经关闭的商店，销量可以直接置零

 

## ***\*好的代码格式\****

![img](file:///C:\Users\sereny\AppData\Local\Temp\ksohtml103712\wps7.jpg) 

# ***\*新函数\****

### ***\*D\*******\*f.\*******\*replace\*******\*(a,b)用b批量替换a\****

### ***\*pd.isnull(data[col])\*******\*定位所有空值。列\****

 

for col in [ 'md_ry_mean']:

  lgb_col_na = pd.isnull(data[col])#

  data[col] = data[col].replace(0,1)

  data.loc[lgb_col_na,col] = \

  ((((data.loc[(data['regYear'].isin([2017]))&(data['regMonth'].isin([1,2,3,4])), col].values /

  data.loc[(data['regYear'].isin([2016]))&(data['regMonth'].isin([1,2,3,4])), col].values)))*

  data.loc[(data['regYear'].isin([2017]))&(data['regMonth'].isin([1,2,3,4])), col].values * 1.03)

————————————————

用repeat和tile扩充数组元素，例如

Np.tile(a)

### ***\*Groupby\*******\*（\**** ***\*as_index\****

 as_index用于是否作为index

df.groupby(['province'], as_index=False)['label'].mean()

pro_df = df.groupby(['province']).agg({'label':['mean','sum']}).reset_index()

 

### ***\*获取列名\****

data.columns/data.keys()

专置

df.groupby(["adcode", "model"])["salesVolume"].apply(lambda x: pd.DataFrame(np.array(x)).T).reset_index().drop("level_2", axis=1)##12个月的转置,adcode0-12月销量

 

### ***\*分成多列\*******\*dummies\****

pd.get_dummies(y).values

\# data = pd.get_dummies(data, columns=['Quality_label'])

### ***\*Python遍历\*******\*enumerate\****

for i, f in enumerate(prob_cols):

sub[f] = prediction[:, i]

 

### ***\*转换目标log\****

np.exmp1(log_predictions)。

log(1+目标)格式

### ***\*标准差\****

app_train_test.sub(app_train_test.mean()).div(app_train_test.std()).abs()

 

### ***\*特征添加\*******\*insert\*******\*（\*******\*position，feature，number）\****

**data**.insert(14,'vdc_A_square',(**data**['电压A']/(**data**['转换效率A']+0.001))**2)

### ***\*Align\****

 

[https://blog.csdn.net/maymay_/article/details/80253068 将轴上的两个对象与每个轴索引的指定连接方法连接](https://blog.csdn.net/maymay_/article/details/80253068  将轴上的两个对象与每个轴索引的指定连接方法连接)

### ***\*Pipline\****

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV

np.random.seed(0)
titanic_url = ('https://raw.githubusercontent.com/amueller/'
               'scipy-2017-sklearn/091d371/notebooks/datasets/titanic3.csv')
data = pd.read_csv(titanic_url)
# 我们将建立我们的分类器用以下特征# numberic features（数值特征）#   - age: float#   - fare: float# categorical features （类别特征）#   - embarked: categories encoded as strings {'C', 'S', 'Q'}.#   - sex: categories encoded as strings {'female', 'male'}.#   - pclass: ordinal integers {1, 2, 3}.
# 预处理数值特征和类别特征
numeric_features = ['age', 'fare']
numeric_transformer = Pipeline(steps=[
  ('imputer', SimpleImputer(strategy='median')),
  ('scaler', StandardScaler())])


categorical_features = ['embarked', 'sex', 'pclass']
categorical_transformer = Pipeline(steps=[
  ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
  ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
  transformers=[
​    ('num', numeric_transformer, numeric_features),
​    ('cat', categorical_transformer, categorical_features)])

# 添加分类器到预处理的pipeline中
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression(solver='lbfgs'))])


X = data.drop('survived', axis=1)
y = data['survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


clf.fit(X_train, y_train)
print("model score: %.3f" % clf.score(X_test, y_test))
```

 