# ***\*学习资源\****

Numpy：https://blog.csdn.net/ScarlettYellow/article/details/80458815

#  工具

 所有代码折叠：ctrl+shift+- 所有代码展开：ctrl+shift++ 折叠某一点：ctrl+- 展开某一层：ctrl++ 

pycharm分块

```
%% 
# In[1]
%%time 运行输出整个Cell的运行时间。

a = 1
%store a
%store -r a #读取
```

### ***\*使用脚本将py2代码转为py3\****

Python安装目录下的Scripts/2to3.exe可以将Pyhon2代码转换为Python3。 


```python
1使用命令2to3 -w file.py可以直接在原地修改文件。
2\
pip install Modernize
python-modernize -w example.py
python-modernize -w 0404Kaggle-Ensemble-Guide-master
3\
pip install future
futurize --stage1 -w 0404Kaggle-Ensemble-Guide-master
#一次性所有项目
```

# ***\*Python报错汇总\****

```

```

### 内存不够

Unable to allocate array with shape (10, 20216100) and data type float64

解决：

import gc
gc.collect()

### Cannot open pip-script.py

Python3要用pip3

### 安装了tensorflow在spyder不能用

命令行下：

Conda create -n tensorflow python=3.6

Activate tensorflow

spyder就能打开可以导入tensorflow'的spyder

https://www.cnblogs.com/HongjianChen/p/8385547.html

 

***\*import cv2\****

下载好ipencv相应的python版本的

在windows中，直接找到对应的cv2.pyd，拷贝到python路径中：anaconda2/lib/python2.7/site-packages/ ，就可以import

 

2.7版本不能caffe，吧packages的caffe要换一下就好了，或者先importcaffe。。我去，浪费半天，，，

3.6不能cv2，不管了。。。。

***\*import错误pycharm\****

 

要把路径导入os.chdir(0

### [Python忽略warning警告错误](http://www.cnblogs.com/blueel/p/3529517.html)

 

import warnings

warnings.filterwarnings("ignore")

### python的print错误，括号

Missing parentheses in call to 'print'——python语法错误

Python2:print"Hello world"

python3:print("Hello world")



### ***\*import同个文件夹错误\****

把当前文件夹作为唯一的

pycharm不会将当前文件目录自动加入自己的sourse_path。右键make_directory as-->sources path将当前工作的文件夹加入source_path就可以了

### ***\*Python目录文件导入问题\****

```python
import sys# 打印环境变量
print(sys.path)
print(sys.argv[2])
```

os.chdir() 方法用于改变当前工作目录到指定的路径。***\*很重要，很多出问题的原因所在\****

os.getcwd()

\#读取文件

import  os

os.chdir('F:/Projects/python/TEST/tianchi1_9/clothes')

导入同个文件的目录：sys.path.append('b模块的绝对路径')

```python
 with open(one_path, "r") as fp:
  for line in fp.readlines():
​    if line.strip().endswith(":"):
​      continue
​    userID, _ , _ = line.split(",")
​    users.add(userID)
```



###  ***\*安装igraph问题\****

 最终，在 http://www.lfd.uci.edu/~gohlke/pythonlibs/#python-igraph

上根据自己的python版本下载python_igraph‑0.7.1.post6‑cp27‑none‑win_amd64.whl， 通过

pip 安装whl文件：pip install 文件名.whl 。

具体方法：在cmd命令窗口中，

找到存放python_igraph‑0.7.1.post6‑cp27‑none‑win_amd64.whl的路径，再输入命令pip install python_igraph‑0.7.1.post6‑cp27‑none‑win_amd64.whl 

\--------------------- 

### ***\*查看版本\****

import lingpy

lingpy.__version__

 

### ***\*代码一行的问题\****

用谷歌或者其他浏览器

常用

### ***\*导入文件的时候导入模块错乱问题0401\****

很多和文件成为了source-root错乱

## ***\*环境问题\****

### ***\*导入文件的时候导入模块错乱问题0401\****

### ***\*更改jupyter工作路径\****

https://www.jianshu.com/p/8f3b4333c979

将打开的Jupyter Notebook程序关闭，然后找到桌面快捷方式，右键=>属性，然后把目标后面输入框最后的“%USERPROFILE%”这个参数去掉后，确定。否则之后做的其它修改无法生效。可以看到如下图：

 

Anaconda选择安装时，如果是默认安装路径：可以在：C:\Users\Administrator\.jupyterp 这个文件夹下面找到一个叫jupyter_notebook_config.py的配置文件。将它用记事本打开：查询找到"c.NotebookApp.notebook_dir ="这个字符串，将它前面的“#”号去掉，等号后面赋值你的默认工作目录  如下图所示：

 

1、ipython_notebook_config.py改成工作路径

c.NotebookManager.notebook_dir = u'D:\\code\\'

2、把“目标”中的 %USERPROFILE% 替换成你想要的目录，e.g.：D:\python-workspace。，起始位置也改成这个

# 模块-了解

## xlrd模块

```python
import xlrd
tables = data.sheets()  #tables[0]
data2 = xlrd.open_workbook('sample2.xls')
data2.sheet_by_index(0) #索引获取表
table = data.sheet_by_name(sheet_name)
table.row_values(0)#获取行的值
table.col_values(0)#获取列的值
#获取行数和列数
table.nrows 
table.ncols
#循环行列表数据
for i in range(table.nrows):
    print(table.row_values(i))
其他参考：https://blog.csdn.net/brucewong0516/article/details/79081320
```



## zip

#### zip

```python
>>>a = [1,2,3]
>>> b = [4,5,6]
>>> c = [4,5,6,7,8]
>>> zipped = zip(a,b)     # 打包为元组的列表
[(1, 4), (2, 5), (3, 6)]
>>> zip(a,c)              # 元素个数与最短的列表一致
[(1, 4), (2, 5), (3, 6)]
>>> zip(*zipped)          # 与 zip 相反，*zipped 可理解为解压，返回二维矩阵式
[(1, 2, 3), (4, 5, 6)]
#2
nums = ['flower','flow','flight']
for i in zip(*nums):
    print(i)
#   ('f', 'f', 'f')
#('l', 'l', 'l')
#3列表元素依次相连：
l = ['a', 'b', 'c', 'd', 'e','f']
print zip(l[:-1],l[1:])  #[('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'e'), ('e', 'f')]
```



### 2环境查看占用内存等

```python
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
seed_everything(SEED)
```



### 1、itertools.product方法

```python
list1 = ['a', 'b']
list2 = ['c', 'd']
for i in itertools.product(list1, list2):
     print i
'''求笛卡尔积
('a', 'c')
('a', 'd')
('b', 'c')
('b', 'd')'''
chars= ['abc', 'def', 'ghi']
[''.join(x) for x in itertools.product(*chars)]
```

### 环境配置

pytorch可以直接用，不需要别的操作

文件夹访问python -m http.server
cd Project/Sereny/Surprise-master
http://server_ip:8000


cd  ~/Project/Sereny/metricfactorization-Douban
source activate tensorflow
nvidia-smi

os.chdir() 方法用于改变当前工作目录到指定的路径。很重要，很多出问题的原因所在
os.getcwd()
import os
os.chdir('F:/Projects/python/TEST/tianchi1_9/clothes')
要导入通个文件夹的模块
sys.path.append('F:/Projects/python/TEST/tianchi1_9/shoplocation')

### ***\*启动\****

if __name__ == "__main__":

main()

### ***\*Sh文件\****

export CUDA_VISIBLE_DEVICES=0,1,2,3

for((i=0;i<5;i++));  

do  

 

python run_bert.py \

--model_type bert \

--model_name_or_path bert-base-chinese \

done  

 

### ***\*SPACY\****

https://blog.csdn.net/u012436149/article/details/79321112



### ***\*if __name__ == 'main':\****

 下的代码只有在第一种情况下（即文件作为脚本直接执行）才会被执行，而import到其他脚本中是不会被执行的。

 ***\*if __name__ == '__main__':\**** 的作用：***\*防止被被其他文件导入时显示多余的程序主体部分\****

### 12.15Random完整



https://www.jianshu.com/p/214798dd8f93

numpy.random.normal函数，有三个参数（loc, scale, size），分别l代表生成的高斯分布的随机数的均值、方差以及输出的size.

1、numpy.random.rand(N,K)：根据给定维度生成[0,1)之间的数据，包含0，不包含1；；；返回0-1之间，N*K个数

2、numpy.random.randn(d0,d1,...,dn)：randn函数返回一个或一组样本，具有标准正态分布。

3、numpy.random.randint(low, high=None, size=None, dtype='l')

返回随机整数，范围区间为[low,high），包含low，不包含high

参数：low为最小值，high为最大值，size为数组维度大小，dtype为数据类型，默认的数据类型是np.int

high没有填写时，默认生成随机数的范围是[0，low)

4、numpy.random.random_integers(low, high=None, size=None)：范围区间为[low,high]

 

4 生成[0,1)之间的浮点数:size=(2,2)

```python
numpy.random.random_sample(size=None)

numpy.random.random(size=None)

numpy.random.ranf(size=None)

numpy.random.sample(size=None)

numpy.random.choice(a, size=None, replace=True, p=None)
```

从给定的一维数组中生成随机数

参数： a为一维数组类似数据或整数；size为数组维度；p为数组中的数据出现的概率

a为整数时，对应的一维数组为np.arange(a)

5、numpy.random.seed()

np.random.seed()的作用：使得随机数据可预测。

当我们设置相同的seed，每次生成的随机数相同。如果不设置seed，则每次会生成不同的随机数

### 排序bisect

一个有趣的python排序模块：bisect

https://www.cnblogs.com/skydesign/archive/2011/09/02/2163592.html



### 归一化





二、将数据特征缩放至某一范围(scalingfeatures to a range)

另外一种标准化方法是将数据缩放至给定的最小值与最大值之间，通常是０与１之间，可用[MinMaxScaler](#sklearn.preprocessing.MinMaxScaler)实现。或者将最大的绝对值缩放至单位大小，可用[MaxAbsScaler](#sklearn.preprocessing.MaxAbsScaler)实现。

使用这种标准化方法的原因是，有时数据集的标准差非常非常小，有时数据中有很多很多零（稀疏数据）需要保存住０元素。

 

### ****Shutil模块：文件移动\****

Inport shutil

Shutil.move(src,dst) ，#原来位置，目标位置

# 分专题常用

## 12seaborn

 https://github.com/fengdu78/Data-Science-Notes 

## 11matplotlib

 https://github.com/fengdu78/Data-Science-Notes 

```python
# num=3表示图片上方标题 变为figure3，figsize=(长，宽)设置figure大小
plt.figure(num=3,figsize=(8,5))
plt.plot(x,y2)
# 红色虚线直线宽度默认1.0
plt.plot(x,y1,color='red',linewidth=1.0,linestyle='--')
plt.show()
```



## 10字典

item_dict.setdefault(item,[]).extend(genres)

## 9scikit-learn

.逻辑回归建模：我们把需要的feature字段取出来，转成numpy格式，使用scikit-learn中的LogisticRegression建模

```python
from sklearn.linear_model import LogisticRegression# 求出Logistic回归的精确度得分
clf = LogisticRegression(solver='lbfgs', multi_class='ovr', max_iter=5000, random_state=42)
#2
from sklearn.ensemble import RandomForestClassifier
# RandomForestClassifier轻松替换LogisticRegression分类器
clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
#一样
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print('Accuracy score of the {} is {:.2f}'.format(clf.__class__.__name__, accuracy))
```

```python
from sklearn.model_selection import train_test_split
X_breast_train, X_breast_test, y_breast_train, y_breast_test = train_test_split(X_breast, y_breast, stratify=y_breast, random_state=0, test_size=0.3)
from sklearn.ensemble import GradientBoostingClassifie

clf = GradientBoostingClassifier(n_estimators=100, random_state=0)
clf.fit(X_breast_train, y_breast_train)
y_pred = clf.predict(X_breast_test)

from sklearn.metrics import balanced_accuracy_score
accuracy = balanced_accuracy_score(y_breast_test, y_pred)
print('Accuracy score of the {} is {:.2f}'.format(clf.__class__.__name__, accuracy))

```

### 预处理

```python
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

```



## 8scipy

 https://github.com/fengdu78/Data-Science-Notes/blob/master/4.scipy/1.scipy-intro.ipynb 

## 7pandas



### Pandas 11.18

优秀网址https://pandas.pydata.org/pandas-docs/stable/getting_started/comparison/comparison_with_sql.html

 

```python
s = pd.Series([1,3,6,np.nan,44,1])
df1 = pd.DataFrame(np.arange(12).reshape(3,4))
df2 = pd.DataFrame({
    'A': [1,2,3,4],
    'B': pd.Timestamp('20180819'),
    'C': pd.Series([1,6,9,10],dtype='float32'),
    'D': np.array([3] * 4,dtype='int32'),
    'E': pd.Categorical(['test','train','test','train']),
    'F': 'foo'
})
dates = pd.date_range('20180819', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6,4)),index=dates, columns=['A','B','C','D']

print(df2.index)
print(df2.values)
print(df2.sort_index(axis=1,ascending=True))
print(df2.sort_values(by='C',ascending=False))
#制定行              
print(df.loc['acolumn'])
print(df.loc[:,'A':'B'])
print(df.loc[:,['A','B']])
print(df.iloc[3,1])# 根据序列iloc 类似矩阵 
#loc等参考
#https://github.com/fengdu78/Data-Science-Notes/blob/master/3.pandas/3.pandas_beginner/pandas_beginner.ipynb
#画图
data.plot()
plt.show
                  
x = data.plot.scatter(x='A',y='B',color='DarkBlue',label='Class1')
# 将之下这个 data 画在上一个 ax 上面
data.plot.scatter(x='A',y='C',color='LightGreen',label='Class2',ax=ax)
plt.show()
```



## 6numpy

```python
#黄海广
array = np.array([ [1,3,5],[4,6,9]], dtype=np.int32)
array.ndim
a = np.empty((3,4))# 创建全空数组
a.reshape((2,3))
np.arange(10,21,2) # 10-20的数据，步长为2
np.linspace(1,10,20) # 开始端1，结束端10，且分割成20个数据，生成线段
np.max(a,axis=1))
np.argmin(A)#s索引
np.mean(A) #np.average(A)
np.cumsum(A) 
np.vstack((A,B))  #vertical stack 上下合并
np.hstack((A,B))#左右
A[np.newaxis,:] # [1 1 1]变为[[1 1 1]]
A[np.newaxis,:].shape # (3,)变为(1, 3)
A[:,np.newaxis]
A = A[:,np.newaxis] # 数组转为矩阵
C = np.concatenate((A,B,B,A),axis=0)# axis=0纵向合并
np.hsplit(A,2) # 等价于print(np.split(A,2,axis=1))# 纵向分割
np.bincount(x,weights=w)
np.around([1.2798,2.357,9.67,13], decimals=1)
np.diff(x,axis=0) 
np.floor([-0.6,-1.4,-0.1,-1.8,0,1.4,1.7])
np.ceil([1.2,1.5,1.8,2.1,2.0,-0.5,-0.6,-0.3])
```

numpy里面有很多数组矩阵的用法，遇到一个就记一个。

### ***\*1.\****[***\*np.logical_and/or/not (逻辑与/或/非)\****](https://blog.csdn.net/JNingWei/article/details/78651535)

#### ***\*np.logical_and(逻辑与)\****

\>>> np.logical_and(True, False)

False

\>>> np.logical_and([True, False], [False, False])

array([False, False], dtype=bool)

\>>> x = np.arange(5)

\>>> np.logical_and(x>1, x<4)

array([False, False, True, True, False], dtype=bool)

np.logical_or(逻辑或)

np.logical_not(逻辑非)

### **2.** ***\*numpy中的nonzero()的用法\****

nonzero(a)返回数组a中值不为零的元素的下标，它的返回值是一个长度为a.ndim(数组a的轴数)的元组

\>>> b1=np.array([True, False, True, False]) 

\>>> np.nonzero(b1) 

(array([0, 2], dtype=int64),) 

 



***\**为什么使用numpy?\**\***

***\**1.\**\***python的list在内存中是分散存储的,numpy是把数据存在连续的内存中，因此遍历更快。

***\**2.\**\***缓存会直接把自己快从ram加在到cpu寄存器中。在连续的内存中，直接利用cpu矢量化指令计算，加载寄存器中多个连续浮点数。

***\**3.\**\***numpy中的矩阵计算可采用多线程方式，发挥了多核cpu计算的优势。

 

***\**需要注意:\**\***

 

避免采用隐式拷贝,而是采用就地操作的方式。

***\**1.\**\*** 数字乘以2，x*=2，而不要写成 y=x*2。这样做速度直接提升两倍。

***\**2.\**\*** 二维数组有两个维度，axis=0 跨行(纵向),axis=1 跨列（横向)。

 

Numpy中有两个重要对象:

ndarray -> 处理多维数组

ufunc -> 处理数组

 

***\**1.基本属性\**\***

```python
aa = np.arange(15).reshape(3,5)
aa[1,1] = 11
**#打印二维数组**
print (aa)
**#数组大小(每个维度的大小)**
print (aa.shape)
**#元素属性**
print (aa.dtype)
**#每个元素字节大小**
print (aa.itemsize)
```

 

 

***\**2.最大值 & 最小值\**\***

```python
a = np.array([[1,2,3], [4,5,6], [7,8,9]])
#全元素中最小值
print (np.amin(a))
#[1,4,7], [2,5,8], [3,6,9]，每组对应位置元素比较，取最小值
print (np.amin(a,0))
#[1,4,7], [2,5,8], [3,6,9]，每组元素和其他组元素比较，取最小值
print (np.amin(a,1))
#全元素中最小值
print (np.amax(a))
#[1,4,7], [2,5,8], [3,6,9]，每组对应位置元素比较,取最大值
print (np.amax(a,0))
#[1,4,7], [2,5,8], [3,6,9]，每组元素和其他组元素比较，取最大值
print (np.amax(a,1))
```

 

**3.求中位数**

b = np.array([[1,2,3], [4,5,6], [7,8,9]])
**#最大值和最小值的差**
print (np.ptp(b))
**#每组对应位置元素比较**
print (np.ptp(b,0))
**#每组元素和其他元素比较**
print (np.ptp(b,1))

 

***\**4.百分位数\**\*** 

```python
b = np.arange(1,10).reshape(3,3)
print (np.percentile(b, 50))
print (np.percentile(b, 50, axis=0))
print (np.percentile(b, 50, axis=1))
```

 

***\**5.中位数 & 平均数\**\***

```python
c = np.arange(1,10).reshape(3,3)
print(c)
# 求中位数
print (np.median(c))
print (np.median(c, axis=0))
print (np.median(c, axis=1))
# 求平均数
print (np.mean(c))
print (np.mean(c, axis=0))
print (np.mean(c, axis=1))
```

 

**6.平均值 & 加权平均值**

d = np.array([1,2,3,4])
wts = np.array([1,2,3,4])
**#平均值**
print (np.average(d))
**#加权平均值**
print (np.average(d,weights=wts))

 

**7.方差 & 标准差**

e = np.arange(1,5)
print (np.std(e))**#方差**
print (np.var(e))**#标准差**

 

***\**8.排序\**\***

```python
f = np.array([[4,3,5],[2,3,1]])
print (np.sort(f))
print (np.sort(f, axis=None))
\#各组比较
print (np.sort(f, axis=0))
\#同组比较
print (np.sort(f, axis=1))
```

 

***\**9. X -> ndarray\**\***

```python
#数组
src = [1,2,3]
dist = np.asarray(src)
print (dist)
#元组
src = (1,2,3)
dist = np.asarray(src)
print (dist)

 

*10.切片 & 索引***

src = np.arange(11)
# 从索引2到9，间隔为2
slc = slice(2,9,2)
print (src[slc])
b = src[2:9:2]
print (b)
```

 

## 5数组

```python
remove
pop
del a[1]
names.index("Xu")
names.sort() #按照ASCII码排序
names.reverse() # 逆序
names.extend(names2)#合并
names2=names.copy()# 浅copy此时names2与names指向相同

# 12.完整克隆
import copy
# 浅copy与深copy
'''浅copy与深copy区别就是浅copy只copy一层，而深copy就是完全克隆'''
names=[1,2,3,4,["zhang","Gu"],5]
# names2=copy.copy(names) # 这个跟列表的浅copy一样
names2=copy.deepcopy(names) #深copy
names[3]="斯"
names[4][0]="张改"
print(names,names2)
```



## 4数据类型

python数组的使用

2010-07-28 17:17

1、Python的数组分三种类型：
(1) list 普通的链表，初始化后可以通过特定方法动态增加元素。
定义方式：arr = [元素]

(2) Tuple 固定的数组，一旦定义后，其元素个数是不能再改变的。
定义方式：arr = (元素)

(2) Dictionary 词典类型， 即是Hash数组。
定义方式：arr = {元素k:v}

 

List.pop

https://www.cnblogs.com/ybjourney/p/4767726.html

## 3df操作数据

[iterrows(), iteritems(), itertuples()对dataframe进行遍历](

## 2编码

​      for line in f:

​        temp = line.decode().split("\t")

**字符串**在Python内部的表示是unicode编码,因此,在做编码**转换**时,通常需要以unicode作为中间编码,

```python
msg="我爱北京天安门"
print(msg.encode(encoding="utf-8")) # str转bytes,编码 b'\xe6\x88\x91\xe7\x88\
print(msg.encode(encoding="utf-8").decode(encoding="utf-8")) # bytes转str,解码  我爱北京天安

# utf-8与gbk互相转化需要通过Unicode作为中介
s="我爱北京天安门"  # 默认编码为Unicode
print(s.encode("gbk")) # Unicode可直接转化为gbk  b'\xce\xd2\xb0\xa
print(s.encode("utf-8")) # Unicode可直接转化为utf-8  b'\xe6\x88\x91\xe7\x88
print(s.encode("utf-8").decode("utf-8").encode("gb2312")) #b'\xce\xd2\xb0\x
## 此时s.encode("utf-8")即转为utf-8了，然后转为gb2312，则需要先告诉Unicode你原先的编码是什么，即s.encode("utf-8").decode("utf-8"),再对其进行编码为gb2312，即最终为s.encode("utf-8").decode("utf-8").encode("gb2312")

```



## 1路径

import失败

Os.chdir(‘目录’)

***\*注：sys.path模块是动态的修改系统路径\****

 

用sys.path.append就行了。当这个append执行完之后，新目录即时起效，以后的每次import操作都可能会检查这个目录。如同解决方案所示，可以选择用sys.path.insert(0,…，这样新添加的目录会优先于其他目录被import检查。

 

0.1是序列位置，，sys.path

sys.path.insert(0, ***\*'F:/Projects/python/TEST/1026caffe_visual'\****) 

http://blog.csdn.net/sinat_27693393/article/details/70037718?locationNum=11&fps=1

https://www.cnblogs.com/keye/p/9673393.html)

 

# 文件读取

### pickle

```python
import pickle
# 写入一个文件，用写入二进制的格式
f = open('master_dictionary.pkl', 'wb')
pickle.dump(master_dictionary, f, -1)
f.close()

data=pickle.load(open('master_dictionary.pkl'.'rb'))


fr = open('master_dictionary.pkl','rb')
data1 = pickle.load(fr)

data2 = pickle.load(fr)
print(data2)
fr.close()

```



### pickle_df

```python
#pandas数据pickling比保存和读取csv文件要快2-3倍（lz测试不准，差不多这么多）。
train_df.to_pickle(os.path.join(CWD, 'middlewares/train_df'))
train_df= pd.read_pickle(os.path.join(CWD, 'middlewares/train_df'))
#不过lz测试了一下，还是直接pickle比较快，比pd.read_pickle快2倍左右。
pickle.dump(ltu_df, open(os.path.join(CWD, 'middlewares/ltu_df.pkl'), 'wb'))
ltu_df = pickle.load(open(os.path.join(CWD, 'middlewares/ltu_df.pkl'), 'rb'))
```



### 2文件

```python
f=open('ly.txt','r',encoding='utf-8') # 文件句柄 'w'为创建文件，之前的数据就没了
data=f.read()
f.write("\n阿斯达所，\n天安门上太阳升")

print(data)
f.close()

# with语句---为了避免打开文件后忘记关闭，可以通过管理上下文
with open('ly.txt','r',encoding='utf-8') as f:
    for line in f:
        print(line.strip())
```



```python
df = pd.read_csv('../jd',sep='\t',error_bad_lines=False) 
```



### 保存

2输入：s=raw_input()  ,a=input(‘aa’)

3二进制文件：F=open(‘score.dat’,wb)#可以保存数据****

```python
json.dump(self.item_matrix,  open('data/item_profile.json','w'))
```

### 输入输出

```python
age=int(input("age:"))  #如果不用int()就会报错(虽然输入为数字，但是print(type(age))为str型)，因为python如果不强制类型转化，就会默认字符型
info='''Name:{0}Age:{1} Job:{2}'''.format(name,age,job)
```

# ***\*Python杂1.11系统\****

### ***\*CGI\****

CGI(Common Gateway Interface)，通用网关接口，它是一段程序，运行在服务器上如：HTTP服务器，提供同客户端HTML页面的接口。

CGI程序可以是Python脚本、Perl脚本、Shell脚本、C或者C++程序等。

 

### ***\*gc.collect()1.11\****

del 删除的其实是一个对象的 ***\*引用\****， 那么 Python 中是否有方法可以直接删除一个对象呢 ？

Python垃圾回收(GC)机制，让用户从繁琐的手动维护内存的工作中，当一个对象的引用计数为0时，那该对象将会被垃圾回收机制回收

import gc

gc.collect()

### ***\*Python常用\****

‘ ‘.join

 

### ***\*Six模块作用1.11\****

Six 出现了。正如它的介绍所说，它是一个专门用来兼容 Python 2 和 Python 3 的库。它解决了诸如 urllib 的部分方法不兼容， str 和 bytes 类型不兼容等“知名”问题。

# ***\*语法\****

### ***\*python语法小技巧\****

1、 结合目录{}/{}***\*'.format\****



### ***\*Dict\****

### ***\*字典dict\****

```python
d.keys()[0]
dic['var5'] = '添加一个值' # 任意添加元素 
dic['var4'] = '任意修改其中一个值' # 任意修改元素 
del dic['var1'] # 任意删除元素 
print(dic) dic.clear() # 清空词典所有条目 
print(dic) del dic # 删除词典
```

#### ***\*operator.itemgette\****

有了上面的operator.itemgetter函数，也可以用该函数来实现，例如要通过student的第三个域排序，可以这么写：

sorted(students, key=operator.itemgetter(2)) 

sorted函数也可以进行多级排序，例如要根据第二个域和第三个域进行排序，可以这么写：

sorted(students, key=operator.itemgetter(1,2)) 

文链接！

 

logger.info("Start to process No.{} image.".format(num))

https://blog.csdn.net/amanfromearth/article/details/80265843

project_root = pathlib.Path()
inputPath = project_root / "data" / "test_images"

 



### ***\*函数map\****

https://blog.csdn.net/SeeTheWorld518/article/details/46959871

格式： 
map(func, seq1[, seq2,…]) 
第一个参数接受一个函数名，后面的参数接受一个或多个可迭代的序列，返回的是一个集合

 

### ***\*时间\****

```python
tic = time.time() #现在测量一下当前时间
\#向量化的版本
c = np.dot(a,b)
toc = time.time()
print(“Vectorized version:” + str(1000*(toc-tic)) +”ms”) #打印一下向量化的版本的时间



start = time.clock()#或者time.time

\# running

end = time.clock()

print end-start
```



### ***\*Python的collections模块中defaultdict类型的用法\****

 



# 画图

## 11.29matplotlib

Data-Science-Notes/5.data-visualization

```python
# Figure size
plt.figure(figsize=(15,5))
# Plot test data
plt.subplot(1, 2, 2)
plt.title("Test")
plt.scatter(X_test, y_test, label="y_test")
plt.plot(X_test, pred_test, color="red", linewidth=1, linestyle="-", label="lm")
plt.legend(loc='lower right')

# Show plots
plt.show()
```



### 1figure

```python
import  matplotlib.pyplot as plt
import numpy as np
#2步骤
#定义函数
x=np.linspace(-3,3,50)#产生-3到3之间50个点
y1=2*x+1
#step1
plt.figure(num=3,figsize=(8,5))#可以没有参数，# num=3表示图片上方标题 变为figure3，figsize=(长，宽)设置figure大小
#step2#两个重叠
plt.plot(x,y2)
plt.plot(x,y1,color='red',linewidth=1.0,linestyle='--')# 红色虚线直线宽度默认1.0
#step3
plt.show()
```



### 2设置坐标轴

```python
#step3之前其他
plt.xlim((-1,2))#设置x轴范围
plt.ylim((-2,3))#设置轴y范围
#设置坐标轴含义， 注：英文直接写，中文需要后面加上fontproperties属性
plt.xlabel(u'价格',fontproperties='SimHei')
plt.ylabel(u'利润',fontproperties='SimHei')
# 设置x轴刻度# -1到2区间，5个点，4个区间，平均分：[-1.,-0.25,0.5,1.25,2.]
new_ticks=np.linspace(-1,2,5)#print(new_ticks)
plt.xticks(new_ticks)
plt.yticks([-2,-1.8,-1,1.22,3.],
           ['非常糟糕','糟糕',r'$good\ \alpha$',r'$really\ good$','超级好'],fontproperties='SimHei')

# 设置边框/坐标轴
gca='get current axis/获取当前轴线'
ax=plt.gca()
# spines就是脊梁，即四个边框
# 取消右边与上边轴
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
# matlibplot并没有设置默认的x轴与y轴方向，下面就开始设置默认轴
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
# 设置坐标原点(0,-1)
ax.spines['bottom'].set_position(('data',-1)) # 也可以是('axes',0.1)后面是百分比，相当于定位到10%处
ax.spines['left'].set_position(('data',0))

```

### 3.Legend 图例

```python

#设置legend图例直接两个放在一起写
l1,=plt.plot(x,y2) # 可添加label属性，只不过如果这里添加了，下面legend再添加，下面的就会覆盖此处的！
l2,=plt.plot(x,y1,color='red',linewidth=1.0,linestyle='--')# 红色虚线直线宽度默认1.0
#prop={'family':'SimHei','size':15}显示中文，loc默认是best,给你放在一个合适的位置上，
plt.legend(handles=[l1,l2],prop={'family':'SimHei','size':15},loc='lower right',labels=['直线','曲线'])
plt.show()
```

### 4.Annotation 标注

```python
1'''其中参数xycoords='data' 是说基于数据的值来选位置, xytext=(+30, -30) 和 textcoords='offset points'
对于标注位置的描述 和 xy 偏差值, arrowprops是对图中箭头类型的一些设置.'''
plt.annotate(r'$2x+1=%s$'%y0,xy=(x0,y0),xycoords='data',xytext=(+30,-30),textcoords='offset points',fontsize=16,arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.2'))
2# 添加注释 text：其中-3.7, 3,是选取text的位置, 空格需要用到转字符\ ,fontdict设置文本字体.
plt.text(-3.7,3,r'$This\ is\ the\ some\ text.\mu\ \sigma_i\ \alpha_t$',
         fontdict={'size':'16','color':'red'})
3# k--表示黑色虚线，k代表黑色，--表示虚线,lw表示线宽；把两个点放进去plot一下，画出垂直于x轴的一条线，[x0,x0]表示两个点的x,[0,y0]表示两个点的y
plt.plot([x0,x0],[0,y0],'k--',lw=2.5)
```

### 3.tick能见度

```python
plt.plot(x, y, linewidth=10, zorder=1) # 设置 zorder 给 plot 在 z 轴方向排序
# 对被遮挡的图像调节相关透明度，本例中设置 x轴 和 y轴 的刻度数字进行透明度设置
for label in ax.get_xticklabels()+ax.get_yticklabels():
    label.set_fontsize(12)
    '''
    其中label.set_fontsize(12)重新调节字体大小，bbox设置目的内容的透明度相关参，
    facecolor调节 box 前景色，edgecolor 设置边框， 本处设置边框为无，alpha设置透明度.
    '''
    # 其中label.set_fontsize(12)重新调节字体大小，bbox设置目的内容的透明度相关参，
#     facecolor调节 box 前景色，edgecolor 设置边框， 本处设置边框为无，alpha设置透明度.
    label.set_bbox(dict(facecolor='white',edgecolor='none',alpha=0.7))

```

### 3.交互式-图和子图

```python
%matplotlib notebook #用Jupyter notebook进行可交互式的绘图
fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)#figure是2x2，选中第1个
ax2 = fig.add_subplot(2, 2, 2)
fig  #画两个
其他的看链接
```



## 之前

```python
plt.figure(figsize=(10, 6))
plt.xticks(rotation=90,fontsize=12)
uniques = [len(train[col].unique()) for col in variables]
sns.set(font_scale=1.2)
ax = sns.barplot(variables, uniques, log=True)
ax.set(xlabel='feature', ylabel='unique count of each feature', title='Number of unique values for each feature')
for p, uniq in zip(ax.patches, uniques):

  height = p.get_height()
  ax.text(p.get_x()+p.get_width()/2.,
​      height + 10,
​      uniq,
​      ha="center") 
```

简而言之，我理解的pivot()的用途就是，将一个dataframe的记录数据整合成表格，而且是按照pivot(‘index=xx’,’columns=xx’,’values=xx’)来整合的。还有另外一种写法， 

但是官方貌似并没有给出来，就是pivot(‘索引列’，‘列名’，‘值’)。

 

 

### ***\*matplotlib学习\****

### ***\*Seaborn作图\****

https://www.jianshu.com/p/f2ec097aedfd

https://zhuanlan.zhihu.com/p/24464836 

 

seaborn同matplotlib一样，也是Python进行数据可视化分析的重要第三方包。但 seaborn 是在 matplotlib 的基础上进行了更高级的API封装，使得作图更加容易，图形更加漂亮

博主并不认为seaborn可以替代matplotlib。虽然 seaborn 可以满足大部分情况下的数据分析需求，但是针对一些特殊情况，还是需要用到 matplotlib 的。换句话说，matplotlib 更加灵活，可定制化，而 seaborn 像是更高级的封装，使用方便快捷。

应该把seaborn视为matplotlib的补充，而不是替代物



## ***\*Seaborn\****

是基于matplotlib的Python可视化库。 它提供了一个高级界面来绘制有吸引力的统计图形。Seaborn其实是在matplotlib的基础上进行了更高级的API封装，从而使得作图更加容易，不需要经过大量的调整就能使你的图变得精致。***\*但应强调的是，应该把Seaborn视为matplotlib的补充，而不是替代物。\****

# 深度学习Tensorflow

## ***\*Tensorflow\****

tensorflow中 tf.reduce_mean函数：https://blog.csdn.net/dcrmg/article/details/79797826

 

```python
 from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

def main(_):
 # 输入数据
 mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

 # 创建模型
 x = tf.placeholder(tf.float32, [None, 784])
 W = tf.Variable(tf.zeros([784, 10]))
 b = tf.Variable(tf.zeros([10]))
 y = tf.matmul(x, W) + b

 # 定义损失函数cross_entropy和优化器
 y_ = tf.placeholder(tf.float32, [None, 10])
 cross_entropy = tf.reduce_mean(
   tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
 train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

 # 定义会话
 sess = tf.InteractiveSession()
 tf.global_variables_initializer().run()
 # 训练

 for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


 # 测试训练得到的模型
 correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
 accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
 print(sess.run(accuracy, feed_dict={x: mnist.test.images,y_: mnist.test.labels}))

## *使用**argparse*
def main():
  parser = argparse.ArgumentParser()

  ## Required parameters
  parser.add_argument("--data_dir", default=None, type=str, required=True,
​            help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
  parser.add_argument("--model_type", default=None, type=str, required=True,
​            help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
  args = parser.parse_args()


#使用参数
  # Setup CUDA, GPU & distributed training
  if args.local_rank == -1 or args.no_cuda:
​    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
​    args.n_gpu = torch.cuda.device_count()
```



### ***\*Tf的变化\****

```python
Summary functions have been consolidated under the tf.summary namespace.
· tf.audio_summary should be renamed to tf.summary.audio 
· tf.contrib.deprecated.histogram_summary should be renamed to tf.summary.histogram 
· tf.contrib.deprecated.scalar_summary should be renamed to tf.summary.scalar
· tf.histogram_summary should be renamed to tf.summary.histogram
· tf.image_summary should be renamed to tf.summary.image 
· tf.merge_all_summaries should be renamed to tf.summary.merge_all 

· tf.merge_summary should be renamed to tf.summary.merge 

· tf.scalar_summary should be renamed to tf.summary.scalar 

· tf.train.SummaryWriter should be renamed to tf.summary.FileWriter

 
```

 

https://www.cnblogs.com/hypnus-ly/p/8040951.html

https://blog.csdn.net/data8866/article/details/61922007

 

from __future__ import print_function

import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')

sess = tf.Session()

print(sess.run(hello))

 

 

2

```python
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

# Define some operations**
**add = tf.add(a, b)
mul = tf.multiply(a, b)

# Launch the default graph.**
with tf.Session() as sess:
  # Run every operation with variable input
*  print("Addition with variables: %i"* % sess.run(add, feed_dict={a: 2, b: 3}))
  print("Multiplication with variables: %i" % sess.run(mul, feed_dict={a: 2, b: 3}))

 
```

3

matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.],[2.]])
product = tf.matmul(matrix1, matrix2)



## ***\*4训练\****

```python
*import* tensorflow *as* tf
*from* tensorflow.examples.tutorials.mnist *import*  input_data

mnist = *input_data.read_data_sets*(*"F:/Projects/python/TEST/tensorflow1017/tensor1231/data/"*, one_hot=*True*)

#创建一个交互式Session**
**sess = tf.InteractiveSession()

#创建两个占位符，x为输入网络的图像，y_为输入网络的图像类别**
x = tf.**placeholder*(*"float"*, shape=[*None*, 784])
y_ = tf.placeholder(*"float"*, shape=[*None****, 10])

#权重初始化函数**
**def** weight_variable(shape):
  #输出服从**截尾正态分布**的随机值
*  initial = tf.truncated_normal(shape, stddev=0.1)
  **return* tf.Variable(initial)

#偏置初始化函数**
**def** bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  **return**** tf.Variable(initial)

#创建卷积op**
#x 是一个4维张量，shape为[batch,height,width,channels]
#卷积核移动步长为1。填充类型为SAME,可以不丢弃任何像素点
**def** conv2d(x, W):
  **return** tf.nn.conv2d(x, W, strides=[1,1,1,1], padding=**"SAME"****)

#创建池化op**
#采用最大池化，也就是取窗口中的最大值作为结果
#x 是一个4维张量，**shape为[batch,height,width,channels]**
#ksize表示pool窗口大小为2x2,也就是高2，宽2
#strides，表示在height和width维度上的步长都为2
**def** max_pool_2x2(x):
  **return** tf.nn.max_pool(x, ksize=[1,2,2,1],strides=[1,2,2,1], padding=**"SAME"****)


#第1层，卷积层**
#初始化W为[5,5,1,32]的张量，表示卷积核大小为5*5，第一层网络的输入和输出神经元个数分别为1和32
W_conv1 = weight_variable([5,5,1,32])
#初始化b为[32],即输出大小
**b_conv1 = bias_variable([32])

#把输入x(二维张量,shape为[batch, 784])变成4d的x_image，x_image的shape应该是[batch,28,28,1]**
**#-1表示自动推测这个维度的size**
**x_image = tf.reshape(x, [-1,28,28,1])

#把x_image和权重进行卷积，加上偏置项，然后应用ReLU激活函数，最后进行max_pooling**
#h_pool1的输出即为第一层网络输出，shape为[batch,14,14,1]
**h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#第2层，卷积层**
#卷积核大小依然是5*5，这层的输入和输出神经元个数为32和64
**W_conv2 = weight_variable([5,5,32,64])
b_conv2 = weight_variable([64])

#h_pool2即为第二层网络输出，shape为[batch,7,7,1]**
**h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#第3层, 全连接层**
#这层是拥有1024个神经元的全连接层
#W的第1维size为7*7*64，7*7是h_pool2输出的size，64是第2层输出神经元个数
*W_fc1 = weight_variable([77*64, 1024])
b_fc1 = bias_variable([1024])

#计算前需要把第2层的输出reshape成[batch, 7*7*64]的张量**
*h_pool2_flat = tf.reshape(h_pool2, [-1, 77*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#Dropout层**
#为了减少过拟合，在输出层前加入dropout
keep_prob = tf.placeholder(**"float"****)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#输出层**
#最后，添加一个softmax层
#可以理解为另一个全连接层，只不过输出时使用softmax将网络输出值转换成了概率
**W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#预测值和真实值之间的交叉墒**
**cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))

#train op, 使用ADAM优化器来做梯度下降。学习率为0.0001**
**train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#评估模型，tf.argmax能给出某个tensor对象在某一维上数据最大值的索引。**
#因为标签是由0,1组成了one-hot vector，返回的索引就是数值为1的位置
**correct_predict = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

#计算正确预测项的比例，因为tf.equal返回的是布尔值，**
#使用tf.cast把布尔值转换成浮点数，然后用tf.reduce_mean求平均值
accuracy = tf.reduce_mean(tf.cast(correct_predict, **"float"****))

#初始化变量**
**sess.run(tf.initialize_all_variables())

#开始训练模型，循环20000次，每次随机从训练集中抓取50幅图像**
**for** i **in** range(10000):
  batch = mnist.train.next_batch(50)
  **if** i%100 == 0:
    #每100次输出一次日志
*    train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})
    print (*"step %d, training accuracy %g"** % (i, train_accuracy))

  train_step.run(feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})

print (*"test accuracy %g"* % accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0}))

 

eval() 其实就是tf.Tensor的Session.run() 的另外一种写法。你上面些的那个代码例子，如果稍微修改一下，加上一个Session context manager：

with tf.Session() as sess:

 print(accuracy.eval({x:mnist.test.images,y_: mnist.test.labels}))

其效果和下面的代码是等价的：

with tf.Session() as sess:

 print(sess.run(accuracy, {x:mnist.test.images,y_: mnist.test.labels}))
```



 

## ***\*5tfrecord生成\****

```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


def _int64_feature(value):  # 生成整数型的属性**
*  *return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):  # 生成字符串型的属性**
*  *return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


mnist = input_data.read_data_sets("F:/Projects/python/TEST/tensorflow1017/tensor1231/data", dtype=tf.uint8, one_hot=True)
images = mnist.train.images

labels = mnist.train.labels  # 训练数据所对应的正确答案，可作为一个属性保存在TFRecord中**
pixels = images.shape[1]
num_examples = mnist.train.num_examples  # 训练数据的图像分辨率，可以作为Example中的一个属性

****

filename = "F:/Projects/python/TEST/tensorflow1017/tensor1231/data"  # 输出TFRecord文件的地址
writer = tf.python_io.TFRecordWriter(filename)  # 通过writer来写TFRecord文件**
for** index in range(num_examples):
  image_raw = images[index].tostring()  # 将图像矩阵转化为一个字符串**
*  example = tf.train.Example(features=tf.train.Feature(feature={
    'pixels'*: _int64_feature(pixels),
    'label': _int64_feature(np.argmax(labels[index])),
    'image_raw': _bytes_feature(image_raw)
  }))
  writer.write(example.SerializeToString())  # 将一个Example写入TFRecord文件
**writer.close()

 
```



## ***\*6读取tfreocord\****


看程序

## ***\*7tensorboard\****

\1. with tensorflow .name_scope(layer_name): 

 

Sh文件打开

用git里面的sh.exe

# ***\*Xgboost学习（博客）\****

## ***\*Python安装xgboost\****

win10 python3.5
1>[Python Extension Packages for Windows](https://link.zhihu.com/?target=http%3A//www.lfd.uci.edu/~gohlke/pythonlibs/%23xgboost)下载对应版本，我的是64位，python3.5所以选35，amd64

![img](D:/ruanjiandata/Typora_mdpic/wps23.jpg) 2>pip install D:\xgboost-0.6-cp35-cp35m-win_amd64.whl![img](file:///C:\Users\sereny\AppData\Local\Temp\ksohtml16892\wps24.jpg)

 

 

测试：import xgboost as xgb  成功

 

## ***\*简介\****

如果你的预测模型表现得有些不尽如人意，那就用XGBoost吧。XGBoost算法现在已经成为很多数据工程师的重要武器。它是一种十分精致的算法，可以处理各种不规则的数据。 

## ***\*学习链接-参数调节\****

http://www.cnblogs.com/haobang008/p/5909207.html

http://blog.csdn.net/wzmsltw/article/details/50994481

 

## ***\*提取特征\****

#### ***\*Pycharm绘图\****

导入matplotlib.pyplot库绘图再调用matplotlib.pyplot.show()能绘制图显示

 

1、我使用%pylab查看matplotlib后端，发现居然是agg。兄弟姐妹们，agg是不会画图的

2、重启

 

#### ***\*错误Mismatch between array dtype ('float32') and format specifier ('%\****



##### ***\*学习随机森林模型\****

http://www.cnblogs.com/jasonfreak/p/5720137.html

https://www.cnblogs.com/amberdata/p/7203632.html

 

sklearn提供了sklearn.ensemble库，其中包括随机森林模型(分类)。但之前使用这个模型的时候，要么使用默认参数，要么将调参的工作丢给调参算法（grid search等）。今天想来深究一下到底是如何选择参数，如何进行调参

 

 ***\*集成学习\****

目前的集成学习方法大致分为两类：

　　a：个体学习器之间存在强依赖关系、必须串行生成的序列化方法：代表为boosting

　　b：个体学习器之间不存在强依赖关系、可以同时生成的并行化方法：代表为bagging和随机森林

本文主要关注第二类。

Random Forest是Bagging的一个扩展变体。RF在以决策树为基学习器构建Bagging的基础上，进一步在决策树的训练过程中引入了随机属性选择。即在树的内部节点分裂过程中，不再是将所有特征，而是随机抽样一部分特征纳入分裂的候选项。

 

在sklearn.ensemble库中，我们可以找到Random Forest分类和回归的实现：RandomForestClassifier和RandomForestRegression。

 

 

 

##### ***\*技巧\****

我们这里用scikit-learn中的RandomForest来拟合一下缺失的年龄数据(注：RandomForest是一个用在原始数据中做不同采样，建立多颗DecisionTree，再进行average等等来降低过拟合现象，提高结果的机器学习算法

5、	

『对数据的认识太重要了！我们还是统计统计，画些图来看看属性和结果之间的关系好了，

有Cabin记录的似乎获救概率稍高一些，先这么着放一放吧。

子女

 

 

『特征工程(feature engineering)太重要了！』 先从最突出的数据属性开始吧，对，Cabin和Age，有丢失数据实在是对下一步工作影响太大。

 

###  

 

 

### ***\*python的Couldn't connect to console process.\*（荒废了至少1天）\****

你看一下你的配置，打开File菜单下的Settings
Console项目下有个Python Console子项目，右边有一个Working directory设置

原来是要设置成当前工程目录然后重启

 

实际：编译器有问题，是ana3，不是python3，重启控制板

os.chdir() 方法用于改变当前工作目录到指定的路径。

os.getcwd()

 

 

### ***\*以后学习查找（或者有空学\****  

[PIL 中的 Image 模块](http://www.cnblogs.com/way_testlife/archive/2011/04/20/2022997.html)

 

## Opencv

#### ***\*为什么使用Python-OpenCV\****

 

虽然python 很强大，而且也有自己的图像处理库PIL，但是相对于OpenCV 来讲，它还是弱小很多。跟很多开源软件一样OpenCV 也提供了完善的python 接口，非常便于调用。OpenCV 的稳定版是2.4.8，最新版是3.0，包含了超过2500 个算法和函数，几乎任何一个能想到的成熟算法都可以通过调用OpenCV 的函数来实现，超

用opencv，就算图像的路径是错的，OpenCV 也不会提醒你的，但是当你使用命 
令print img时得到的结果是None。

#### ***\*Waitkey\****

第一个参数： 等待x ms，如果在此期间有按键按下，则立即结束并返回按下按键的

 

ASCII码，否则返回-1

 

如果x=0，那么无限等待下去，直到有按键按下

另外，在imshow之后如果没有waitKey语句则不会正常显示图像。

## ***\*Python\****

StratifiedKFold交叉验证

 

路径

***\*if not\**** (os.path.exists(path_x) ***\*or\**** os.path.exists(path_y)):

 

 

np.random.uniform)  **#从一个均匀分布[low,high,（a,b）)中随机采样a,b维数个,**

 

save_path = ***\*'./results/idx{}_{}-sampling_winsize{}_{}samples_overlap{}'\****.format(test_idx, sampling, win_size, num_samples, overlapping)

\#{}用来后面的format填入

Flatten转化成一位数据

# ***\*Python运算\****

abs(x)：返回整数的绝对值，如abs(-10)返回10。

ceil(x)：返回数字的向上取整，如math.ceil(4.1)返回5。

exp(x)：返回e的x次幂，如math.exp(1)返回2.718281828459045。

fabs(x)：返回浮点数的绝对值，如math.fabs(-10) 返回10.0。

floor(x)：返回数字的向下取整，如math.floor(4.9)返回4。

log(x,base)：如math.log(math.e,math.e)返回1.0，math.log(100,10)返回2.0`。

log10(x)：返回以10为基数的x的对数，如math.log10(100)返回2.0。

max(x1,x2,...)：返回给定参数的最大值，参数可以为序列。

min(x1,x2,...)：返回给定参数的最小值，参数可以为序列。

modf(x)：以元组的形式返回，（小数部分,整数部分）。两部分的数值符号与x相同，整数部分以浮点型表示。

pow(x, y)：xyx^yxy运算后的值。

round(x [,n])：返回浮点数x的四舍五入值，如给出n值，则代表舍入到小数点后的位数。

sqrt(x)：返回数字x的平方根，返回类型为实数，如math.sqrt(4)返回2.0。

atan2(y, x)：返回给定的X及Y坐标值的反正切值。

hypot(x, y)：返回欧几里德范数sqrt(x2+y2)sqrt(x^2 + y^2)sqrt(x2+y2)。

tan(x)：返回x弧度的正切值。

degrees(x)：将弧度转换为角度，如degrees(math.pi/2) ， 返回90.0。

radians(x)：将角度转换为弧度

 

acos(x)：返回x的反余弦弧度值。

 

除了上述常用的数学函数，math库中还定义了两个常用的数学常量：

pi——圆周率，一般以π来表示。

e——自然常数。

 

一般有两种常用方法来使用math中的函数：

 

  import math

print(math.abs(3))

 

### ***\*2运算\****

\1. /是精确除法，/在python 2 中是向下取整运算

//是向下取整除法，%是求模

\2. %求模（余）是基于向下取整除法规则的

\3. 四舍五入取整round, 向零取整int, 向下和向上取整函数math.floor, math.ceil

\4. //和math.floor在CPython中的不同

\5. /在python 2 中是向下取整运算

\6. C中%是向零取整求模。

7.a**b a的几次方

模和除数同号(除非能整除是模是0)

\---------------------

 

 

 

· i//j：表示整数除法。例如8//2的值为int类型4，9//4的值为int类型2。即整数除法只取整数商，去掉小数部分。

· · i/j：表示对象i除以对象j，无论i和j的类型是int还是float，结果都为float，如10/4结果为2.5。

· · i%j：表示int对象i除以int对象j的余数，即数学的“模”运算。

· 

**radians**() 方法将角度转换为弧度

 

前后端联调，host映射

***\*不断来回切换本地mock数据和后端接口数据时\****就遇到了联调的问题。

吗

 

 

师兄使用了mock数据

 

首先，前后端进行定制接口，定制完成后各自进行开发。前端的开发者使用mock数据进行开发，开发完成后进行真实环境的联调，找出开发中的问题，再进行测试、上线等流程。

新工具mockserver晚点再看



 

 

# ***\*Python学习\****

 



### ***\*文件读写\****

#### ***\*1open\****

1、 f = open('/file', 'r')

2、with open('/file', 'r') as f:

print(f.read())

使用r：

是推荐使用的打开文本文件的模式。因为使用此模式打开文本文件时，python默认为我们做了一些处理，比如：假设在windows下，将本来应该读入的换行符\r\n处理成\n,方便我们处理。（值得一提的是，当你将\n写入文件时，python也会默认将其替换成\r\n，如果你是win系统的话）

补充：其实是启用了通用换行符支持（UNS），它默认开启。 

***\*使用rb\****： 
则python不会对文本文件预处理了，你读入的\r\n依然是\r\n.

 

### 

### ****全0初始化\****

matrix = np.zeros((users,items)) 

## ***\*Sklearn\****

### ***\*数据集使用\****

### ***\*sklearn的datasets使用\****

· load_<dataset_name> 本地加载数据

· fetch_<dataset_name> 远程加载数据

· make_<dataset_name> 构造数据集

学习链接：https://www.jianshu.com/p/faeee62d60c4

#### ***\*load_svmlight_\****f***\*ile\****

scikit-learn支持多种格式的数据，包括经典的iris数据，LibSVM格式数据等等。为了方便起见，推荐使用LibSVM格式的数据，详细见LibSVM的官网。

from sklearn.datasets importload_svmlight_file，导入这个模块就可以加载LibSVM模块的数据

t_X,t_y=load_svmlight_file("filename")

函数参数：

 

load_svmlight_file(f, n_features=None, dtype=<type 'numpy.float64'>, multilabel=False,zero_based='auto', query_id=False)

 

参数介绍：

 

'f'为文件路径

 

'n_features'为feature的个数，默认None自动识别feature个数

 

'dtype'为数据集的数据类型（不是很了解），默认是np.float64

 

'multilable'为多标签，多标签数据的具体格式可以参照(这里)

 

'zero_based'不是很了解

 

'query_id'不是很了解。

 

load_svmlight_file文档(文档链接)

### ***\*1request\****

http://docs.python-requests.org/zh_CN/latest/user/quickstart.html

对requests获取的原始数据，有两种获取形式，一个是r.content一个是r.text。

r = requests.get('https://github.com/timeline.json')

r.encoding

 

from io import BytesIO

f=BytesIO(r.content)

要操作二进制数据，就需要使用BytesIO

 

### ***\*2\*******\*libsvm所用数据格式\****

Label 1:value 2:value ….

Label：是类别的标识，比如上节train.model中提到的1 -1，你可以自己随意定，比如-10，0，15。当然，如果是回归，这是目标值，就要实事求是了。

Value：就是要训练的数据，从分类的角度来说就是特征值，数据之间用空格隔开

比如: -15 1:0.708 2:1056 3:-0.3333

## ***\*numpy\****

### ***\*1numpy.reshape\****

y.reshape(-1,1)  -1表示未知行数，1列

### ***\*enumerate\****

\>>>seq = ['one', 'two', 'three'] >>> for i, element in enumerate(seq): ... print i, element ... 

### ***\*yield\****

简要理解：yield就是 return 返回一个值，并且记住这个返回的位置，下次迭代就从这个位置后(下一行)开始

def loadfile(path):

  with open(path,"r") as f:

​    for i,line in enumerate(f):

​      yield line

 

### ***\*生成点击字典的格式\****

 

def read_ratings(path, pivot=0.8):
  **"""** **
**    **Return:****
**            **点击的字典形式，格式为{userId : { movieId : rating}}****
**  **"""****
**  train_set = dict()
  test_set = dict()

  for line in loadfile(path):
    user, movie, rating, _ = line.split("::")
    if random.random() < pivot:
      train_set.setdefault(user, {})
      train_set[user][movie] = int(rating)
    else:
      test_set.setdefault(user, {})
      test_set[user][movie] = int(rating)

  return train_set, test_set

 

### ***\*引入：\*******\*from __future__ import\****

from __future__ import absolute_import

 

来引入系统的标准string.py，否则会先查找当前目录有无

 

from __future__ import print_function

查阅了一些资料，这里mark一下常见的用法！

首先我们需要明白该句语句是python2的概念，那么python3对于python2就是future了，也就是说，在python2的环境下，超前使用python3的print函数。

所以以后看到这个句子的时候，不用害怕，只是把下一个新版本的特性导入到当前版本！

 

# ***\*12.16图片numpy等学习\****

![img](file:///C:\Users\sereny\AppData\Local\Temp\ksohtml16892\wps25.png)![img](file:///C:\Users\sereny\AppData\Local\Temp\ksohtml16892\wps26.png)![img](file:///C:\Users\sereny\AppData\Local\Temp\ksohtml16892\wps27.png)![img](file:///C:\Users\sereny\AppData\Local\Temp\ksohtml16892\wps28.png)![img](file:///C:\Users\sereny\AppData\Local\Temp\ksohtml16892\wps29.png)![img](file:///C:\Users\sereny\AppData\Local\Temp\ksohtml16892\wps30.png)![img](D:/ruanjiandata/Typora_mdpic/wps31.png)![img](file:///C:\Users\sereny\AppData\Local\Temp\ksohtml16892\wps32.png)![img](file:///C:\Users\sereny\AppData\Local\Temp\ksohtml16892\wps33.png)![img](file:///C:\Users\sereny\AppData\Local\Temp\ksohtml16892\wps34.png) 

# ***\*12.17\****

### ***\*imp.reload作用\****

***\*reload函数的作用是重新加载已经导入过的包\**** 
在python中由.py结尾的文件都是一个可以调用执行的模块， 
但是由于导入模块会浪费资源（模块导入必须找到代码文件，并且把它转化为字节码，还要运行代码）

所以python在一个模块中已经导入了一个包，如果这个***\*包里面的代码再发生变化的话\****，在这个模块中的包还是按照没有变更之前的代码运行，但是这显然是不行的，这时候就***\*需要用reload了\****。

 

reload方法在imp模块中，用的时候需要导入

 

from imp import reload

reload(模块名)

reload的参数必须是一个模块，而且是一个已经到如果的模块

 

 

# ***\*一、argparse模块\****

python标准库模块argparse用于解析命令行参数，编写用户友好的命令行界面，该模块还会自动生成帮助信息，并在所给参数无效时报错。 
首先看一个例子

https://blog.csdn.net/guoyajie1990/article/details/76739977

 

 

文件夹被python解释器视作package需要满足两个条件：

　　***\*1、文件夹中必须有__init__.py文件，该文件可以为空，但必须存在该文件。\****

命令行的python命令不导入当前目录的包

环境变量，导入的不是对的文件，重启成功

路径没改，，就想着一切都是对的

### ***\*向量的计算\****

http://www.ai-start.com/dl2017/html/lesson1-week2.html#header-n337

 

![img](D:/ruanjiandata/Typora_mdpic/wps35.jpg) 

![img](D:/ruanjiandata/Typora_mdpic/wps36.jpg) 

不完全确定一个向量的维度(***\*dimension\****)，我经常会扔进一个断言语句(***\*assertion statement\****)

### ***\*2019.02.20\****

## ***\*正则表达式\****

X=input(‘please input a stri  ng:’)

Pattern=re.compile(r’\b[a-z]{3}\b’)

Print(pattern.findal(x))

 

http://www.runoob.com/python/python-reg-expressions.html

 

***\*re\*******\*.\*******\*match\****(pattern, string, flags=0)

***\*re\*******\*.\*******\*search\****(pattern, string, flags=0)#扫描整个字符串并返回第一个成功的匹配。

## ***\*re.match与re.search的区别\****

re.match只匹配字符串的开始，如果字符串开始不符合正则表达式，则匹配失败，函数返回None；而re.search匹配整个字符串，直到找到一个匹配。

## ***\*检索和替换\****

Python 的 re 模块提供了re.sub用于替换字符串中的匹配项。

语法：

re.sub(pattern, repl, string, count=0, flags=0)

· pattern : 正则中的模式字符串。repl : 替换的字符串 string : 要被查找替换的原始字符串。count : 模式匹配后替换的最大次数，默认 0 表示替换所有的匹配。

 

 

# ***\*爬虫\****

### ***\*Python requests模块params与data的区别\****

param = {"wd": "莫烦Python"}
r = requests.get('http://www.baidu.com/s', params=param)

相当于：[https://www.baidu.com/s?wd=%E8%8E%AB%E7%83%A6Python](https://www.baidu.com/s?wd=莫烦Python)

### ***\*Cookies\****

cookies 就是用来衔接一个页面和另一个页面的关系. 比如说当我登录以后, 浏览器为了保存我的登录信息, 将这些信息存放在了 cookie 中. 然后我访问第二个页面的时候, 保存的 cookie 被调用, 服务器知道我之前做了什么, 浏览了些什么. 像你在网上看到的广告, 为什么都可能是你感兴趣的商品? 你登录淘宝, 给你推荐的为什么都和你买过的类似? 都是 cookies 的功劳, 让服务器知道你的个性化需求.

 

我用 requests.post + payload 的用户信息发给网页, 返回的 r 里面会有生成的 cookies 信息. 接着我请求去登录后的页面时, 使用 request.get, 并将之前的 cookies 传入到 get 请求. 这样就能已登录的名义访问 get 的页面了.

 

### [Python urllib模块urlopen()与urlretrieve()详解](https://www.cnblogs.com/qqhfeng/p/5785373.html)

一个获取数据，一个直接下载数据

 

### ***\*分布式爬虫\****

进程池

Window下多进程出错：永远把实际执行功能的代码加入到带保护的区域中：if __name__ == '__mian__':（原因：https://blog.csdn.net/qq_36708806/article/details/79731276）

 

### ***\*fire.Fire()\****

fire是python中用于生成命令行界面(Command Line Interfaces, CLIs)的工具，不需要做任何额外的工作，只需要从主模块中调用fire.Fire()，它会自动将你的代码转化为CLI，Fire()的参数可以说任何的python对象

https://www.cnblogs.com/cnhkzyy/p/9574560.html

 

Cuda是硬件基础，GPU是实现工具

# ***\*224\****

### ***\*语法学习\****

1unratedItems = nonzero(dataMat[user, :].A == 0)[1]  **#****mat.A把矩阵转换成array类型,,nonzero得到非0元素位置**

2根据字典中值的大小，对字典中的项排序 

itemScores.append((item, estimatedScore))

sorted(dict2.iteritems(),key=lambda item:item[1],reverse=True)

3numpy：shape函数是numpy.core.fromnumeric中的函数，它的功能是查看矩阵或者数组的维数。

4、list[start:end:step]

[::2]第

5、eval获取输入：a=eval(raw_input())

 

 

有时候很奇怪的不行就重启，速度找到原因

# 报错问题

### ***\*Can't find model 'en_core_web_sm'\****

python -m spacy download en_core_web_sm

# 方法

### ***\*sklearn.metrics\*******\*评估函数\****

***\*accuracy_score,recall_score,roc_curve,roc_auc_score,confusion_matrix\****

1、accuracy_score(y_true, y_pred) 

2、召回率 =提取出的正确信息条数 /样本中的信息条数。通俗地说，就是所有准确的条目有多少被检索出来了。

klearn.metrics.recall_score(y_true, y_pred, labels=None, pos_label=1,average='binary', sample_weight=None)

 

1、分类报告：

sklearn.metrics.***\*classification_report\****(y_true, y_pred, labels=None, target_names=None,sample_weight=None, digits=2)

，显示主要的分类指标，返回每个类标签的精确、召回率及[F1](https://www.baidu.com/s?wd=F1&tn=24004469_oem_dg&rsv_dl=gh_pl_sl_csd)值

2、accuracy_score

Pipeline：https://blog.csdn.net/wateryouyo/article/details/53909636

 

### 训练测试划分1

from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.33, random_state=42)

### ***\*统计数量画图\****

fig = plt.figure(figsize=(8,4))
sns.barplot(x = train[***\*'Conference'\****].unique(), y=train[***\*'Conference'\****].value_counts())

 

Count

INFO_counts = Counter(INFO_clean) **#元素作为key，其计数作为value。count.most_common****
**INFO_common_words = [word[0] ***\*for\**** word ***\*in\**** INFO_counts.most_common(20)]





### ***\*矩阵\****

 

scipy中稀疏矩阵coo_matrix, csr_matrix 的使用

当对离散数据进行拟合预测时，往往要对特征进行onehot处理，但onehot是高度稀疏的向量，如果使用List或其他常规的存储方式，对内存占用极大。 

这时稀疏矩阵类型 coo_matrix / csr_matrix 就派上用场了！

 

这两种稀疏矩阵类型csr_matrix存储密度更大，但不易手工构建。coo_matrix存储密度相对小，但易于手工构建，常用方法为先手工构建coo_matrix，如果对内存要求高则使用 tocsr() 方法把coo_matrix转换为csr_matrix类型。

ratings = coo_matrix((df.rating, (df.user_id, df.item_id)))
ratings = ratings.tocsr()

矩阵中非零元素的数量nnz

size_of_bucket = int(ratings.nnz / K) #nnz非0个数/数据数量/K交叉验证数量

csr的Indices非0元素的行坐标

 

# ***\*用PrettyPrinter，让Python输出更漂亮，你值得拥有\****

该处的 pass 便是占据一个位置，因为如果定义一个空函数程序会报错，当你没有想好函数的内容是可以用 pass 填充，使程序可以正常运行。

 %matplotlib inline 可以在Ipython编译器里直接使用，功能是可以内嵌绘图，并且可以省略掉plt.show()这一步。

 

 一些常用python预处理方法 https://blog.csdn.net/bryan__/article/details/51228971

 



 

### ***\*Torch\****

pip install torch torchvision

# ***\*4.6\****

###  super(B, self).__init__("foo")

 

class A(object):

  def __init__(self, arg):

​    print "Inside class A init. arg =", arg

 

class B(A):

  def __init__(self):

​    super(B, self).__init__("foo")

​    print "Inside class B init"

 

\>>> b = B()

Inside class A init. arg = foo

Inside class B init

### ***\*4.6python文件运行 sys.argv\****

```python
first_file = sys.argv[1]

second_file = sys.argv[2]

 

def corr(first_file, second_file):

 first_df = pd.read_csv(first_file,index_col=0)

 second_df = pd.read_csv(second_file,index_col=0)

 prediction = first_df.columns[0]

 print("Finding correlation between: %s and %s" % (first_file,second_file))

 print("Column to be measured: %s" % prediction)

 

corr(first_file, second_file)
```



### ***\*from __future__ import print_function、from __future__ import absolute_import\****

是把python3的特性导入到python2！

于这句from __future__ import absolute_import的作用: 

直观地看就是说”加入绝对引入这个新特性”。说到绝对引入，当然就会想到相对引入。那么什么是相对引入呢?比如说，你的包结构是这样的: 

pkg/ 

pkg/init.py 

pkg/main.py 

pkg/string.py

 

如果你在main.py中写import string,那么在Python 2.4或之前, Python会先查找当前目录下有没有string.py, 如果是想引入系统自带的标准string.py。这时候你就需要from __future__ import absolute_import了。这样，你就可以用import string来引入系统的标准string.py, 

用from pkg import string来引入当前目录下的string.py了

 

### ***\*Glob\****

是python自己带的一个文件操作相关模块，用它可以查找符合自己目的的文件

\#获取上级目录的所有.py文件  

print glob.glob(r'../*.py') #相对路径  

 

### ***\*Counter(scores[(j,k)]).most_common(1)[0][0]))\****

most_common(n) 按照counter的计数，按照降序，返回前n项组成的list; n忽略时返回全部

### ***\*md5\**** 

python3.x已经把md5 module移除了。要想用md5得用hashlib module,以下是帮助手册中给的标准调用

python 2.7下

import md5

m = md5.new()

m.update("Nobody inspects the spammish repetition")

md5value=m.hexdigest()

 

md5.new("Nobody").hexdigest()

 

python3：

import hashlib

m = hashlib.md5()

m.update(b"Nobody inspects the spammish repetition") #参数必须是byte类型，否则报Unicode-objects must be encoded before hashing错误

md5value=m.hexdigest()

print(md5value)  #bb649c83dd1ea5c9d9dec9a18df0ffe9

# ***\*414\****

### ***\*for _ in range(num_len)\****

_ 你可以当它是一个变量，但一般习惯不用这个变量。这个循环的用途是循环5次

 

 

# ***\*4.27\****

***\*调用sess.run()的时候，程序是否执行了整个图\****

session.run([fetch1, fetch2])

***\*tensorflow并没有计算整个图，只是计算了与想要fetch 的值相关的部分\****

[***\*https://blog.csdn.net/u012436149/article/details/52908692\****](https://blog.csdn.net/u012436149/article/details/52908692)

***\*道feed_dict的作用是给使用placeholder创建出来的tensor赋值。其实，他的作用更加广泛：feed 使用一个 值临时替换一个 op 的输出结果.\****

 

# ***\*5.11\****

### ***\*ubuntu下命令行下光标的控制\****

https://blog.csdn.net/u013617648/article/details/73194684  Ctrl + d 删除一个字符(删除光标后字符)，类似于通常的Delete(删除光标前字符)键（命令行若无所有字符，则相当于exit；处理多行标准输入时也表示eof）

Ctrl + h 退格删除一个字符(删除光标前字符)，相当于通常的Backspace键

Ctrl + u 删除光标之前到行首的字符

Ctrl + k 删除光标之前到行尾的字符

 

 

Ctrl + c 取消当前行输入的命令，相当于Ctrl + Break

Ctrl + a 光标移动到行首（Ahead of line），相当于通常的Home键

Ctrl + e 光标移动到行尾（End of line）

Alt + f 光标向前（Forward）移动到下一个单词 

Alt + b 光标往回（Backward）移动到前一个单词

Ctrl + w 删除从光标位置前到当前所处单词（Word）的开头

Alt + d 删除从光标位置到当前所处单词的末尾

Ctrl + y 粘贴最后一次被删除的单词

 

 

Ctrl + l 清屏，相当于执行clear命令

Ctrl + p 调出命令历史中的前一条（Previous）命令，相当于通常的上箭头

Ctrl + n 调出命令历史中的下一条（Next）命令，相当于通常的上箭头

 

### ***\*查找数据\****

ad_static = ad_static[ad_static.adCreatetime != 0] #(726621, 7)

y=ad_static1[ad_static1['adIndustryid'].str.contains(',')]

 

### ***\*缺失值：https://blog.csdn.net/ustbbsy/article/details/80748978\****

data.isnull().any()  #每一列是否有空值

data.isnull().any().sum()

data = data.replace('null',np.NaN)

data.fillna(0) 

 

### ***\*时间\*******\*戳\*******\*转换：\****https://blog.csdn.net/google19890102/article/details/51355282

https://blog.csdn.net/haimianjie2012/article/details/83957207

 

pd.to_datetime(1550345479 , unit='s')
pd.to_datetime(20190320020301 ,format='%Y%m%d')
time.gmtime(1550345479)
data['adReqday']=data['adReqtime'].map(**lambda**  x: time.gmtime(x))
data['adReqsecond']=pd.to_datetime(data['adReqtime'] , unit='s') #秒需要的时候再换算

 

s.map(pd.Timestamp.date) 或者 s.map(lambda x: pd.to_datetime(x.date()))

 

但是pd.Timestamp.date会将数据的类型从datetime类型转换成date类型，在pd中显示是object类型；而转换成datetime的函数pd.to_datetime特别慢而耗时。

\--------------------- 

作者：-柚子皮- 

来源：CSDN 

原文：https://blog.csdn.net/pipisorry/article/details/52209377 

版权声明：本文为博主原创文章，转载请附上博文链接！

**#转换成时间数组** timeArray = time.strptime(dt, "%Y-%m-%d %H:%M:%S") **#转换成时间戳** timestamp = time.mktime(timeArray)

M=Pd.to_datatime(字符串/时间戳) :变成timestamp('2019-02-21 00:00:00')

M.year/month/day/hour/second/minute

![img](D:/ruanjiandata/Typora_mdpic/wps37.jpg) 

 

就是datetime.datetime

Time.gmtime(时间戳)得到struct![img](D:/ruanjiandata/Typora_mdpic/wps38.jpg)

![img](D:/ruanjiandata/Typora_mdpic/wps39.jpg) 

 

 

![img](D:/ruanjiandata/Typora_mdpic/wps40.jpg) 

ad_static['adCreatetime']=ad_static['adCreatetime'].map(**lambda** x: time.strftime('%Y-%m-%d %H:%M:%S',time.localtime (x)))

df1['adReqtime']=data['adReqtime'].map(**lambda** x:  time.mktime(time.strptime(x,"%Y-%m-%d %H:%M:%S")))

 

![img](D:/ruanjiandata/Typora_mdpic/wps41.jpg) 

#### ***\*时间python\****

https://blog.csdn.net/weixin_41789707/article/details/83009235

在Python中，通常有这三种方式来表示时间：时间戳、元组(struct_time)、格式化的时间字符串：

 

(1)时间戳(timestamp) ：通常来说，时间戳表示的是从1970年1月1日00:00:00开始按秒计算的偏移量。我们运行“type(time.time())”，返回的是float类型。

 

(2)格式化的时间字符串(Format String)： ‘2018-04-18’

 

(3)元组(struct_time) ：struct_time元组共有9个元素共九个元素:(年，月，日，时，分，秒，一年中第几周，一年中第几天等）

\--------------------- 

作者：南阜止鸟 

来源：CSDN 

原文：https://blog.csdn.net/weixin_41789707/article/details/83009235 

版权声明：本文为博主原创文章，转载请附上博文链接！

### ***\*其他\****

 

df['abc']=df['abc'].map(lambda x: x**2)) #某一列修改

df[df['列名'].isin([相应的值])] #得到某一个值所在的行:

df.duplicated() #查找数据重复的

df[df.duplicated==True].shape

 

df[df['adId'].isin([394352])] #得到某一个值所在的行:

 

 df[df['ids'].str.contains("ball")]

 

 

 df1=df1[~df1['A'].isin([1])] # 删除/选取某列含有特殊数值的行

 

source_df[~(source_df['date'].map(lambda d: d.split('/')[0])).isin([year])]

 

 

ad_static1=ad_static[ ~ ad_logdata['adSize'].isin(,)]  #删除某列包含特殊字符的列

把有逗号分隔的行删除【pandas

 

ad_static1=ad_static[ ~ ad_logdata['adSize'].isin(",")]  #删除某列包含特殊字符的列

ad_static1=ad_static[~(ad_static['adIndustryid'].astype(str).map(lambda x: x.contains([','])))]

ad_static1=ad_static

ad_static2=ad_static[~ad_static[ad_static['adIndustryid'].astype(str).str.contains(",")]]

 

### ***\*#\*******\*1获取数据\*******\*#使用HDF5格式能快速读取和写入pandas。\****

用HDF5格式能快速读取和写入panda,h5接受的数据是矩阵跟mat方法一致，但是具有更强的压缩性能

```python
import h5py

data= pd.HDFStore('../data/data.h5')

data[‘adlogdata’] = df  

data.get('df') /data['df']  /store.x

del data['wp']

data.close()  \store.is_open

\#打开

import pandas as pd

import numpy as np

data= pd.read_hdf('../data/csv/adlogdata.h5')

data= pd.read_hdf('../data/csv/data.h5',key='adlogdata')

 

ad_logdata=data['adlogdata']

ad_static=data['adlogdata']

ad_operation=data['adoperation']

data.keys

 
```

 

###  ***\*#画图\*******\*linux\****

\# 绘图并保存，重要不显示图片会推出，顺序不能换

```python
import** matplotlib
matplotlib.use('agg') #不显示图像
**import** matplotlib.pyplot **as** plt

plt.scatter(x, y, alpha=0.6)  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）

plt.savefig('./test.png')#保存图片


**import** matplotlib
matplotlib.use('agg')
**import** matplotlib.pyplot **as** plt

**import** numpy **as** np  # 数组相关的库

\# 创建横纵坐标数据
N = 10
x = np.random.rand(N)  # 包含10个均匀分布的随机值的横坐标数组，大小[0, 1]
y = np.random.rand(N)  # 包含10个均匀分布的随机值的纵坐标数组
\# 设置坐标轴名称
plt.xlabel('x-label-English')
plt.ylabel('y-label-English')
\# 设置标题
plt.title('title',fontsize=20,verticalalignment='bottom') # 设置字体大小，垂直底部对齐

\# 绘图并保存
plt.scatter(x, y, alpha=0.6)  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
plt.savefig('./test.png')#保存图片

 plt.savefig('_labeled.jpg')

 savefig(fname, dpi=None, facecolor='w', edgecolor='w',

​    orientation='portrait', papertype=None, format=None,

​    transparent=False, bbox_inches=None, pad_inches=0.1,

​    frameon=None, metadata=None)

 

​		

df = pd.DataFrame(np.random.rand(50, 4), columns=['a', 'b', 'c', 'd'])

ax=df.plot.scatter(x='a', y='b')

fig = ax.get_figure()

fig.savefig('label_distribution.png')

 

 

df.plot.scatter(x=ad_logdata.expoAdbid)
```

 

### ***\*#减少数据类型\*******\*，减小pandas数据\****

\# 对数据按照格式进行压缩重新存储

def compressData(inputData):

  '''

  :parameters: inputData: pd.Dataframe

  :return: inputData: pd.Dataframe

  :Purpose: 

  压缩csv中的数据，通过改变扫描每列的dtype，转换成适合的大小

  例如: int64, 检查最小值是否存在负数，是则声明signed，否则声明unsigned，并转换更小的int size

  对于object类型，则会转换成category类型，占用内存率小

  参考来自：https://www.jiqizhixin.com/articles/2018-03-07-3

  '''

  for eachType in set(inputData.dtypes.values):

​    \##检查属于什么类型

​    if 'int' in str(eachType):

​      \## 对每列进行转换

​      for i in inputData.select_dtypes(eachType).columns.values:

​        if inputData[i].min() < 0:

​          inputData[i] = pd.to_numeric(inputData[i],downcast='signed')

​        else:

​          inputData[i] = pd.to_numeric(inputData[i],downcast='unsigned')    

​    elif 'float' in str(eachType):

​      for i in inputData.select_dtypes(eachType).columns.values:  

​        inputData[i] = pd.to_numeric(inputData[i],downcast='float')

​    elif 'object' in str(eachType):

​      for i in inputData.select_dtypes(eachType).columns.values: 

​        inputData[i] = trainData7[i].astype('category')

  return inputData

\--------------------- 

作者：rdd-mylover 

来源：CSDN 

原文：https://blog.csdn.net/qq_24831889/article/details/82919058 

版权声明：本文为博主原创文章，转载请附上博文链接！

#### ***\*转化object：Df1=df1.infer_objects()\****

https://blog.csdn.net/u013385925/article/details/80250316

Gl是pandas.df数据

gl=ad_logdata50

gl_int = gl.select_dtypes(include=['int'])

converted_int = gl_int.apply(pd.to_numeric,downcast='unsigned')

gl_float = gl.select_dtypes(include=['float'])

converted_float = gl_float.apply(pd.to_numeric,downcast='float')

 

***\*optimized_gl = gl.copy()\****

***\*optimized_gl[converted_int.columns] = converted_int\****

optimized_gl[converted_float.columns] = converted_float

开始强制转换

datafloat=['Adpctr','Adqualityecpm','Adtotalecpm']
dataint=['adReqid','adReqtime','adPostionid','userId','adId','adSize','Adbid']
**for** fea **in** datafloat:
  ad_logdata[[fea]] = ad_logdata[[fea]].astype('float32')
**for** fea **in** dataint:
  ad_logdata[[fea]] = ad_logdata[[fea]].astype('int32')




### ***\*Python相同id数据groupby合并\****

pandas按照某一列计数求和,Pandas分组运算后数据的合并

df["id"].value_count()

df.groupby('A').mean()

df['num'].groupby(df['adId'])

 

 

df['Adbid'].groupby(df['adId']).mean()

 

***\*#字符连接\****

data = pd.DataFrame({'id':[1,1,1,2,2,2],'value':['A','B','C','D','E','F']})

data['value'] = data['value'].apply(lambda x:','+ x) #都变成“,A”

data1 = data.groupby(by='id').sum()#字符串相加

data1['value'] = data1['value'].apply(lambda x :x[1:]) #去除‘,’

 

### ***\*Pandas—DataFrame的读取、保存、增、删、查、改：http://www.zhongruitech.com/209160154.html\****

 

pandas.read_csv（file,sep="\t",header=标题行,names=列名/None，prefix，engine=c（更快）/python（更完善），nrows =100，iterator =False（True逐块处理文件））

\#分块读取

reader = pd.read_csv('./train.csv', iterator=True)

try:

  df = reader.get_chunk(70000) # 读取70000行数据

except StopIteration:

  print ("Iteration is stopped.")

print (df.info())

 

DataFrame.to_csv(path，sep=“,”，header=True（默认保存列名，index=True(默认保存索引0不保存)，columns : 要写的列）

 na_rep : 用于替换空数据的字符串，默认为''

 float_format : 设置浮点数的格式（几位小数点）

 

\##

增加一列数据

增加一行数据 

### ***\*数据处理同类id合并，数值字符串（写了3小时）\****

https://codeday.me/bug/20171205/104918.html

Groupby：https://www.cnblogs.com/lemonbit/p/6810972.html

####  ***\*分组后Pandas的索引转换为列数据\****

\>>> df = pd.DataFrame(np.arange(12).reshape((4,3)), index=[['a','a','b','b'], [1,2,1,2]], columns=['green','blue','red'])>>> df.index.names=['key1','key2']

1、normal_df['index1'] = normal_df.index

2、 normal_df.reset_index()

## ***\*常用：变量存储\****

H5读取更快

ad_operation.to_hdf('adoperation.h5', key='adoperation')#保存

ad_operation= pd.read_hdf('adoperation2.h5') #759545--758052(2)

 

'''读取原来h5'''
df1 = pd.read_hdf('../data/csv/adlogdata.h5') #df=ad_logdata
'''获取转化了adSize的，所以处理之前就确定是否一致'''
ad_logdata50=df.infer_objects()
ad_logdata_col=['adId','Adbid', 'Adpctr','Adqualityecpm','Adtotalecpm','adReqtime',  'adPostionid','userId','adSize','adIdcount']
ad_logdata50.columns=ad_logdata_col

 

import pickle

pickle.dump(m,open('../a.pkl','wb+')) #b二进制重要

m=pickle.load(open('../a.pkl','rb'))

 

df存储： np.save（'df.npy',  df.values)

numpy读取转换回df,没有了header：

df=pd.DataFrame( np.load('df.npy')，allow_pickle=True)   

df=df.infer_objects()

ad_logdata_col=['Adbid','Adpctr','Adqualityecpm','Adtotalecpm','adReqtime','adPostionid','userId','adSize','adId'

]
ad_logdata.columns=ad_logdata_col

 

 

### ***\*画图plot\****

x=list(df2['adSize'].values)
y=list(df2['adIdexpcount'].values)
plot.close() #会叠加
x= [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66]
y=[170.61190027540223, 3146.5714285714284, 55.26495726495727, 13.5, 42.43217054263566, 57.285714285714285, 3158.0, 88.5, 44.74765100671141, 282.7105263157895, 47807.0, 442.8556701030928, 4.371108343711083, 25]
plot.ylabel('count')
plot.xlabel("adsize")
plot.plot(x,y,label='adSize-count') #折线图list，list
plot.savefig('./plot2.png')  # 保存图

 

plt.scatter(df2['adSize'],df2['adIdexpcount'],alpha=0.6)  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）

### ***\*模型\****

![img](D:/ruanjiandata/Typora_mdpic/wps42.jpg) 

![img](D:/ruanjiandata/Typora_mdpic/wps43.jpg) 

 

、数据统计

len(users['Age'].unique())  
\#len(set(users['Age']))  都可以

 

 

# ***\*5.15\****

### ***\*CountVectorizer\****

利用CountVectorizer（）进行分词并统计出现频数

返回的数据即为CSR存储的数据

如果对这一类的数据进行像稠密数据一样的concat等操作的时候可以使用

 

sparse.hstack 相当于稠密矩阵中的pd.concat(axis=1)

sparse.vstack 相当于稠密矩阵中的pd.concat(axis=0)

\--------------------- 

### ***\*数据操作语句\****

https://blog.csdn.net/a940902940902/article/details/84729223

pandas表连接可以使用 join 以及merge
对于连接之后可能会由于操作不慎出现部分元素为NaN的情况 这时我们需要找出这些情况
df.isnull().any() 会显示各列是否存在为NaN的元素

#### ***\*输出数组中出现次数最多的元素\****

可以使用np.bincount() 以及 np.argmax() 组合
对于 np.bincount() 输出的数组比x的最大值大1 输出数组每个位置的输出值 代表该位置的值在原始数组中出现的次数

value_counts对Series值进行统计并排序

使用value_counts 输出Series 对应的出现次数 返回的是一个Series数组 ，如果想要得到各个值在所有值中所占的比重 可以加上 normalize=True 输出的则是一个Series，index是属性名 valus是对应属性值所占的比重

 

get_dummies() 和 factorize（）

get_dummies ：

aim: 得到category类型特征的one-hot 编码

arguments:Series or dataFrame

pd.get_dummies(pd.Series(list(“abcaa”)))

\--------------------- 

pd.factorize()
相当于label encoding

对于category特征

for i in trainData.columns[trainData.dtype==‘object’]:

trainData[i]=trainData[i].factorize()[0]

可以简单对trainData进行label encoding

 

drop_dumplicates()

\--------------------- 

drop_dumplicates() 来删除数据中完全重复的数据文！

## ***\*python基础（5）：深入理解 python 中的赋值、引用、拷贝、作用域\****

 

## ***\*0824\**** 

Python的append可以直接训练测试结合

https://blog.csdn.net/sinat_29957455/article/details/84961936

构建字典

 

dict = dict.fromkeys(('Google', 'Runoob', 'Taobao'), 10)

 

title_Dict = {}

title_Dict.update(dict.fromkeys(['Capt','Col','Major','Dr','Rev'],'Officer'))

# ***\*830\****



### ***\*离散特征处理\****

 

one_hot_feature = ['desire_jd_salary_id', 'cur_salary_id', 'cur_degree_id', 'live_city_id']

for feature in one_hot_feature:

​		try:

​			data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))  # 按照大小顺序排列给从0开始的数字,转化为int，特征都转化成数字类别

​		except:

​			data[feature] = LabelEncoder().fit_transform(data[feature]) #data[feature].apply(int)		try:

​			data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))  # 按照大小顺序排列给从0开始的数字,转化为int，特征都转化成数字类别

​		except:

​			data[feature] = LabelEncoder().fit_transform(data[feature]) #data[feature].apply(int)

​		

enc = OneHotEncoder()

train_x = train[['creativeSize']]  # 广告素材大小,只是数值型用于构造df

test_x = test[['creativeSize']]

for feature in one_hot_feature:

  enc.fit(data[feature].values.reshape(-1, 1))  # -1表示未知行数，1列

  train_a = enc.transform(train[feature].values.reshape(-1, 1))

  test_a = enc.transform(test[feature].values.reshape(-1, 1))

  train_x = sparse.hstack((train_x, train_a))  # 方便和后面生成的稀疏特征进行拼接，生成df

  \# <8798814x68727 sparse matrix of type '<class 'numpy.float64'>'with 175325048 stored elements in COOrdinate format>

  test_x = sparse.hstack((test_x, test_a))

print('one-hot prepared !')

 

 

 

### ***\*re正则表达式查找\****

https://docs.python.org/zh-cn/3/library/re.html

 

### ***\*类别特征\****

```python
from sklearn.preprocessing import LabelEncoder
le =LabelEncoder()
le.fit(["paris", "paris", "tokyo", "amsterdam"])#分别为[0,0,1,2]
le.transform(["tokyo", "tokyo", "paris"]) 

from sklearn import  preprocessing
print('***********LabelEncoder***********')
test1=pd.DataFrame({'city':['beijing','shanghai','shenzhen'],'age':[21,33,23],'target':[0,1,0],'city22':['shenzhen','shanghai','beijing']})
label = preprocessing.LabelEncoder()
test1['city']= label.fit_transform(test1['city'])##第一个出现的就是1
test1['city22']= label.fit_transform(test1['city22'])
print(test1)
print('***********OneHotEncoder***********')
\#要先lebel，categorical_features默认是‘all’表示对所有特征编码，[1]是变量的索引，表示第2个变量；
\#sparse 缺省状态下是True表示输出为稀疏矩阵。未经编码的变量放在右边。
enc=preprocessing.OneHotEncoder(categorical_features=[0], sparse=False)#自己onehot后和别人连接
test1=enc.fit_transform(test1)
print(test1)
test=test1
test=pd.DataFrame({'age':[21,33,23],'target':[0,1,0],'city22':['shenzhen','shanghai','beijing']})

print('***********factorize***********')
test=test.apply(lambda x: pd.factorize(x)[0])  ##?????
print(test)
print(pd.factorize(test['age']))

print('***********get_dummies***********')
print(pd.get_dummies(test['age'],prefix='age'))

 
```



### ***\*Python多列数据处理\****

天池看--泰坦尼克

文字学会观察数据，不同类型的数据，名称都有用

性别与是否生存的关系 

```python
print(train_data.groupby(['Sex','Survived'])['Survived'].count())

train_data[['Sex','Survived']].groupby(['Sex']).mean()

train_data[['Sex','Survived']].groupby(['Sex']).mean().plot.bar()

\#各种画图学习ing。。。

\#画图1

fig,ax = plt.subplots(1,2, figsize = (18,5))

ax[0].set_yticks(range(0,110,10))

sns.violinplot("Pclass","Age",hue="Survived",data=train_data,split=True,ax=ax[0])

ax[0].set_title('Pclass and Age vs Survived') 

ax[1].。。。

plt.show()

\#画图2

plt.figure(figsize=(15,5))

plt.subplot(121)

train_data['Age'].hist(bins=100)

plt.xlabel('Age')

plt.ylabel('Num')

 

plt.subplot(122)

train_data.boxplot(column='Age',showfliers=False)

plt.show()
```

 

\#

```python
dummy

embark_dummies = pd.get_dummies(train_data['Embarked']) #自动编码成值

train_data = train_data.join(embark_dummies)

train_data.drop(['Embarked'], axis=1, inplace=True)

\#正则

re.compile("([a-zA-Z]+)").search(x).group()

\#4.2 Factoring

\#很多情况下我们需要将数值做Scaling使其范围大小一样，否则大范围数特征将会有更高的权重。比如：Age的范围可能只是0-100，而income的范围可能是0-10000000

from sklearn import preprocessing

\#4.3 Scaling

assert np.size(train_data['Age']) == 891

\# StandardScaler will subtract the mean from each value then scale to the unit varience

from sklearn import preprocessing

scaler = preprocessing.StandardScaler()

train_data['Age_scaled'] = scaler.fit_transform(train_data['Age'].values.reshape(-1,1))

 

 

\#4.4Binning

\# 在将数据Binning化后，要么将数据factorize化，要么dummies化。

train_data['Fare_bin'] = pd.qcut(train_data['Fare'],5)

\# factorize

train_data['Fare_bin_id'] = pd.factorize(train_data['Fare_bin'])[0]

\# dummies

fare_bin_dummies_df = pd.get_dummies(train_data['Fare_bin']).rename(columns=lambda x: 'Fare_' + str(x))

train_data = pd.concat([train_data, fare_bin_dummies_df], axis=1)

 

 

 

from sklearn import ensemble

from sklearn import model_selection

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import RandomForestRegressor

 

def fill_missing_age(missing_age_train, missing_age_test):

  missing_age_X_train = missing_age_train.drop(['Age'], axis=1)

  missing_age_Y_train = missing_age_train['Age']

  missing_age_X_test = missing_age_test.drop(['Age'], axis=1)

 

  \# model 1  gbm

  gbm_reg = GradientBoostingRegressor(random_state=42)

  gbm_reg_param_grid = {'n_estimators': [2000], 'max_depth': [4], 'learning_rate': [0.01], 'max_features': [3]}

  gbm_reg_grid = model_selection.GridSearchCV(gbm_reg, gbm_reg_param_grid, cv=10, n_jobs=25, verbose=1, scoring='neg_mean_squared_error')

  gbm_reg_grid.fit(missing_age_X_train, missing_age_Y_train)

  print('Age feature Best GB Params:' + str(gbm_reg_grid.best_params_))

  print('Age feature Best GB Score:' + str(gbm_reg_grid.best_score_))

  print('GB Train Error for "Age" Feature Regressor:' + str(gbm_reg_grid.score(missing_age_X_train, missing_age_Y_train)))

  missing_age_test.loc[:, 'Age_GB'] = gbm_reg_grid.predict(missing_age_X_test)

  print(missing_age_test['Age_GB'][:4])

 

  \# model 2 rf

  rf_reg = RandomForestRegressor()

  rf_reg_param_grid = {'n_estimators': [200], 'max_depth': [5], 'random_state': [0]}

  rf_reg_grid = model_selection.GridSearchCV(rf_reg, rf_reg_param_grid, cv=10, n_jobs=25, verbose=1, scoring='neg_mean_squared_error')

  rf_reg_grid.fit(missing_age_X_train, missing_age_Y_train)

  print('Age feature Best RF Params:' + str(rf_reg_grid.best_params_))

  print('Age feature Best RF Score:' + str(rf_reg_grid.best_score_))

  print('RF Train Error for "Age" Feature Regressor' + str(rf_reg_grid.score(missing_age_X_train, missing_age_Y_train)))

  missing_age_test.loc[:, 'Age_RF'] = rf_reg_grid.predict(missing_age_X_test)

  print(missing_age_test['Age_RF'][:4])

 

  \# two models merge

  print('shape1', missing_age_test['Age'].shape, missing_age_test[['Age_GB', 'Age_RF']].mode(axis=1).shape)

  \# missing_age_test['Age'] = missing_age_test[['Age_GB', 'Age_LR']].mode(axis=1)

 

  missing_age_test.loc[:, 'Age'] = np.mean([missing_age_test['Age_GB'], missing_age_test['Age_RF']])

  print(missing_age_test['Age'][:4])

 

  missing_age_test.drop(['Age_GB', 'Age_RF'], axis=1, inplace=True)

 

return missing_age_test

 
```



### ***\*p\*******\*andas中进行数据类型转换有三种基本方法：\****

 

使用astype()函数进行强制类型转换

自定义函数进行数据类型转换

使用Pandas提供的函数如to_numeric()、to_datetime()

data['客户编号'].astype('object')

 

## ***\*Eval、\*******\*ast.literal_eval()\****

使用eval可以实现从元祖，列表，字典型的字符串到元祖，列表，字典的转换，此外，eval还可以对字符

串型的输入直接计算。比如，她会将'1+1'的计算串直接计算出结果。 

https://blog.csdn.net/Jerry_1126/article/details/68831254

安全处理方式***\*ast.literal_eval\****.

***\*出于安全考虑，对字符串进行类型转换的时候，最好使用ast.literal_eval()函数\****

 

***\*P\*******\*ython使用\****T-SNE，kmeans，pca，knn

https://zhuanlan.zhihu.com/p/87100453***\*!\****



## ***\*有用\****

### ***\*yield\**** 

带有 yield 的函数不再是一个普通函数，而是一个生成器generator，可用于迭代，

### ***\*codecs.open\****

https://www.cnblogs.com/buptldf/p/4805879.html

codecs.open可以指定一个编码打开文件，一般不会出现编码的问题