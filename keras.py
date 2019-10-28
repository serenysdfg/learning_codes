# -*- coding: utf-8 -*-
# @createTime : 2019.10.28 13:36
# @Author : sereny
# keras中文文档：https://keras-cn.readthedocs.io/en/latest/models/model/

# keras 两种训练模型方式fit和fit_generator(节省内存):https://blog.csdn.net/u011311291/article/details/79900060
# In[1]
# from keras.models import Model
from keras.layers import Input, Dense

a = Input(shape=(32,))
b = Dense(32)(a)
model = Model(inputs=a, outputs=b)

model = Model(inputs=[a1, a2], outputs=[b1, b3, b3])
'''
常用Model属性
model.layers：组成模型图的各个层
model.inputs：模型的输入张量列表
model.outputs：模型的输出张量列表'''

compile()
compile(self, optimizer, loss, metrics=None, loss_weights=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None)
fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)
evaluate(self, x, y, batch_size=32, verbose=1, sample_weight=None)
predict(self, x, batch_size=32, verbose=0)
train_on_batch(self, x, y, class_weight=None, sample_weight=None)

# In[2]
#利用Python的生成器，逐个生成数据的batch并进行训练。生成器与模型将并行执行以提高效率。例如，该函数允许我们在CPU上进行实时的数据提升，同时在GPU上进行模型训练
fit_generator(self, generator, steps_per_epoch, epochs=1, verbose=1, callbacks=None, validation_data=None, validation_steps=None, class_weight=None, max_q_size=10, workers=1, pickle_safe=False, initial_epoch=0)
###例如
# def generate_arrays_from_file(path):
    while 1:
    f = open(path)
    for line in f:
        # create numpy arrays of input data
        # and labels, from each line in the file
        x1, x2, y = process_line(line)
        yield ({'input_1': x1, 'input_2': x2}, {'output': y})
    f.close()

model.fit_generator(generate_arrays_from_file('/my_file.txt'),
        steps_per_epoch=10000, epochs=10)