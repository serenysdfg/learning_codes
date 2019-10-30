思路：先标准化
df.sub([1,2,3,4], axis='columns')
df.sub(df.mean()).div(df.std()).abs() #减去均值/标准差  的绝对值：标准化

#Pandas 矩阵运算：1是纵向
sub/mlu/div
DataFrame.add([1,2,33], axis=1, level=None, fill_value=None)
#mask
和定位有点像

添加特征二次三次试试