import numba
import os
import numpy as np
import copy
import sys
from constant.constant import DATA_BASE_PATH,DEV_PATH
import numpy.ctypeslib as npct
from ctypes import *
import pickle
import sys
import utils.globalvar as gl
from constant.constant import DEV_PATH
lib = cdll.LoadLibrary(os.path.join(DEV_PATH,'utils','oputils.dll'))
import pandas as pd


def _Corr(
        X,
        Y,
        d,
):
    Y = np.array(Y,dtype =c_float)
    X = np.array(X,dtype = c_float)
    res = Y.copy()
    lib.faster_func.restypes = npct.ndpointer(dtype = c_float, ndim = 1, flags="C_CONTIGUOUS")
    lib.faster_func.argtypes = [
        npct.ndpointer(dtype=c_float, ndim=1, flags="C_CONTIGUOUS"),
        npct.ndpointer(dtype = c_float, ndim = 1, flags="C_CONTIGUOUS"),
        npct.ndpointer(dtype = c_float, ndim = 1, flags="C_CONTIGUOUS"),
        c_int,
        c_int,
        c_int,
        c_int
    ]
    lib.faster_func(res,X,Y,gl.get_value('trade_num'),gl.get_value('stock_num'),0,d)
    return res

def _Delay(X,d):
    '''
        说明文档:
                5日滞后
        参数:
                X：numpy.ndarray
                    数据x
        返回：
                data:numpy.ndarray
                    计算后的数据
    '''
    X = np.array(X,dtype =c_float)
    res = X.copy()
    lib.faster_func.restypes = npct.ndpointer(dtype = c_float, ndim = 1, flags="C_CONTIGUOUS")
    lib.faster_func.argtypes = [
        npct.ndpointer(dtype=c_float, ndim=1, flags="C_CONTIGUOUS"),
        npct.ndpointer(dtype = c_float, ndim = 1, flags="C_CONTIGUOUS"),
        npct.ndpointer(dtype = c_float, ndim = 1, flags="C_CONTIGUOUS"),
        c_int,
        c_int,
        c_int,
        c_int
    ]
    lib.faster_func(res,X,X,gl.get_value('trade_num'),gl.get_value('stock_num'),3,d)
    return res

def _Delta(X,d):
    '''
        说明文档:
                5日数值差
        参数:
                X：numpy.ndarray
                    数据x
        返回：
                data:numpy.ndarray
                    计算后的数据
    '''
    X = np.array(X,dtype =c_float)
    res = X.copy()
    lib.faster_func.restypes = npct.ndpointer(dtype = c_float, ndim = 1, flags="C_CONTIGUOUS")
    lib.faster_func.argtypes = [
        npct.ndpointer(dtype=c_float, ndim=1, flags="C_CONTIGUOUS"),
        npct.ndpointer(dtype = c_float, ndim = 1, flags="C_CONTIGUOUS"),
        npct.ndpointer(dtype = c_float, ndim = 1, flags="C_CONTIGUOUS"),
        c_int,
        c_int,
        c_int,
        c_int
    ]
    lib.faster_func(res,X,X,gl.get_value('trade_num'),gl.get_value('stock_num'),6,d)
    return res

def _Std(X,d):
    '''
        说明文档:
                5日标准差
        参数:
                X：numpy.ndarray
                    数据x
        返回：
                data:numpy.ndarray
                    计算后的数据
    '''
    X = np.array(X,dtype =c_float)
    res = X.copy()
    lib.faster_func.restypes = npct.ndpointer(dtype = c_float, ndim = 1, flags="C_CONTIGUOUS")
    lib.faster_func.argtypes = [
        npct.ndpointer(dtype=c_float, ndim=1, flags="C_CONTIGUOUS"),
        npct.ndpointer(dtype = c_float, ndim = 1, flags="C_CONTIGUOUS"),
        npct.ndpointer(dtype = c_float, ndim = 1, flags="C_CONTIGUOUS"),
        c_int,
        c_int,
        c_int,
        c_int
    ]
    lib.faster_func(res,X,X,gl.get_value('trade_num'),gl.get_value('stock_num'),2,d)
    return res



def _Min(X,d):
    '''
        说明文档:
                5日最小值
        参数:
                X：numpy.ndarray
                    数据x
        返回：
                data:numpy.ndarray
                    计算后的数据
    '''
    X = np.array(X,dtype =c_float)
    res = X.copy()
    lib.faster_func.restypes = npct.ndpointer(dtype = c_float, ndim = 1, flags="C_CONTIGUOUS")
    lib.faster_func.argtypes = [
        npct.ndpointer(dtype=c_float, ndim=1, flags="C_CONTIGUOUS"),
        npct.ndpointer(dtype = c_float, ndim = 1, flags="C_CONTIGUOUS"),
        npct.ndpointer(dtype = c_float, ndim = 1, flags="C_CONTIGUOUS"),
        c_int,
        c_int,
        c_int,
        c_int
    ]
    lib.faster_func(res,X,X,gl.get_value('trade_num'),gl.get_value('stock_num'),5,d)
    return res

def _Max(X,d):
    '''
        说明文档:
                5日最小值
        参数:
                X：numpy.ndarray
                    数据x
        返回：
                data:numpy.ndarray
                    计算后的数据
    '''
    X = np.array(X,dtype =c_float)
    res = X.copy()
    lib.faster_func.restypes = npct.ndpointer(dtype = c_float, ndim = 1, flags="C_CONTIGUOUS")
    lib.faster_func.argtypes = [
        npct.ndpointer(dtype=c_float, ndim=1, flags="C_CONTIGUOUS"),
        npct.ndpointer(dtype = c_float, ndim = 1, flags="C_CONTIGUOUS"),
        npct.ndpointer(dtype = c_float, ndim = 1, flags="C_CONTIGUOUS"),
        c_int,
        c_int,
        c_int,
        c_int
    ]
    lib.faster_func(res,X,X,gl.get_value('trade_num'),gl.get_value('stock_num'),4,d)
    return res


def _ArgSort(X,d):
    '''
       说明文档:
               5日序列中的数据位置
       参数:
               X：numpy.ndarray
                   数据x
       返回：
               data:numpy.ndarray
                   计算后的数据
    '''
    np_append = np.zeros((gl.get_value('stock_num'), d)) * np.NaN
    stock_num=gl.get_value('stock_num')
    trade_num=gl.get_value('trade_num')
    X = X.reshape((stock_num,trade_num))
    X = np.hstack((X,np_append)).reshape(((trade_num+d)*stock_num,))
    @numba.jit(nopython=True,cache=True)
    def _ArgSort_(X,d):
        data = []
        for i in range(stock_num):
            for j in range(trade_num):
                if(X[i * (trade_num+d) + j] is np.NaN):
                    data.append(np.NaN)
                else:
                    index = 1
                    Data = X[i * (trade_num+d) + j]
                    for k in range(d-1):
                        if(X[i * (trade_num+d) + j + k + 1]> Data and X[i * (trade_num+d) + j + k + 1] is not np.NaN):
                            index = index+1
                    data.append(index)
        return np.array(data)
    data = _ArgSort_(X,d)
    return data

def _Mean(X,d):
    '''
        文档说明:
                5日平均
        参数:
                X:numpy.ndarray
                       数据x
        返回:
                data:numpy.ndarray
                       计算后的数据
    '''
    X = np.array(X,dtype =c_float)
    res = X.copy()
    lib.faster_func.restypes = npct.ndpointer(dtype = c_float, ndim = 1, flags="C_CONTIGUOUS")
    lib.faster_func.argtypes = [
        npct.ndpointer(dtype=c_float, ndim=1, flags="C_CONTIGUOUS"),
        npct.ndpointer(dtype = c_float, ndim = 1, flags="C_CONTIGUOUS"),
        npct.ndpointer(dtype = c_float, ndim = 1, flags="C_CONTIGUOUS"),
        c_int,
        c_int,
        c_int,
        c_int
    ]
    lib.faster_func(res,X,X,gl.get_value('trade_num'),gl.get_value('stock_num'),1,d)
    return res

def _ArgSortMax(X,d):
    '''
        文档说明:
               10日最大值位置
        参数:
                X:numpy.ndarray
                       数据x
        返回:
                data:numpy.ndarray
                       计算后的数据
    '''
    np_append = np.zeros((gl.get_value('stock_num'), d)) * np.NaN
    stock_num=gl.get_value('stock_num')
    trade_num=gl.get_value('trade_num')
    X = X.reshape((stock_num, trade_num))
    X = np.hstack((X, np_append)).reshape(((trade_num + d) * stock_num,))
    @numba.jit(nopython=True,cache=True)
    def _ArgSortMax_(X):
        data = []
        for i in range(stock_num):
            for j in range(trade_num):
                if(np.max(X[i * (trade_num + d) + j:i * (trade_num + d) + j + d]) is np.NaN):
                    data.append(np.NaN)
                else:
                    data.append(np.argmax(X[i * (trade_num + d) + j:i * (trade_num + d) + j + d]))
        return np.array(data)

    data = _ArgSortMax_(X)
    return data

def _ArgSortMin(X,d):
    '''
        文档说明:
                60日最小值位置
        参数:
                X:numpy.ndarray
                       数据x
        返回:
                data:numpy.ndarray
                       计算后的数据
    '''
    np_append = np.zeros((gl.get_value('stock_num'), d)) * np.NaN
    stock_num=gl.get_value('stock_num')
    trade_num=gl.get_value('trade_num')
    X = X.reshape((stock_num, trade_num))
    X = np.hstack((X, np_append)).reshape(((trade_num + d) * stock_num,))
    @numba.jit(nopython=True,cache=True)
    def _ArgSortMin_(X):
        data = []
        for i in range(stock_num):
            for j in range(trade_num):
                if(np.min(X[i * (trade_num + d) + j:i * (trade_num + d) + j + d]) is np.NaN):
                    data.append(np.NaN)
                else:
                    data.append(np.argmax(X[i * (trade_num + d) + j:i * (trade_num + d) + j + d]))
        return np.array(data)

    data = _ArgSortMin_(X)
    return data

reg_func_list = {}
f_locals = copy.copy(sys._getframe().f_locals)
for key in f_locals.keys():
    if(
            callable(f_locals[key])==True
            and key[0:2]!='__'
            and key!= '_pydev_stop_at_break'
            and key[0]=='_'
    ):
        reg_func_list[key] = f_locals[key].__code__.co_argcount
