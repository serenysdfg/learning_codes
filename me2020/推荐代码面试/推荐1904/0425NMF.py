# -*- coding: utf-8 -*-
# @createTime : 2019/4/25 16:20
# @Author : sereny

#!/bin/python

from numpy import *

def load_data(file_path):
    f = open(file_path)
    V = []
    for line in f.readlines():
        lines = line.strip().split("\t")
        data = []
        for x in lines:
            data.append(float(x))
        V.append(data)
    return mat(V)

def train(V, r, k, e):
    m, n = shape(V)
    W = mat(random.random((m, r)))
    H = mat(random.random((r, n)))

    for x in range(k):
        V_pre = W * H
        E = V - V_pre #error
        #print E
        err = 0.0
        for i in range(m):
            for j in range(n):
                err += E[i,j] * E[i,j]
        print (err)

        if err < e:
            break

        a = W.T * V
        b = W.T * W * H
        #c = V * H.T
        #d = W * H * H.T
        for i_1 in xrange(r):
            for j_1 in xrange(n):
                if b[i_1,j_1] != 0:
                    H[i_1,j_1] = H[i_1,j_1] * a[i_1,j_1] / b[i_1,j_1]

        c = V * H.T
        d = W * H * H.T
        for i_2 in range(m):
            for j_2 in range(r):
                if d[i_2, j_2] != 0:
                    W[i_2,j_2] = W[i_2,j_2] * c[i_2,j_2] / d[i_2, j_2]

    return W,H


if __name__ == "__main__":
    file_path = "./data1"

    V = load_data(file_path)
    W, H = train(V, 2, 100, 1e-5 )

    print (V)
    print (W)
    print (H)
    print (W * H)

