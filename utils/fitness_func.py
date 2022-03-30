import numpy as np
import pickle
import os
import sys
from constant.constant import DATA_BASE_PATH,DEV_PATH
import copy
import sys
import utils.globalvar as gl
from ctypes import *
import pandas as pd
lib = cdll.LoadLibrary(os.path.join(DEV_PATH,'utils','fitness_dll.dll'))
import numpy.ctypeslib as npct
from copy import deepcopy
def _ic_ir(x,y,w):
    '''
    说明文档:
            计算适应度函数
    参数:
            x:numpy.ndarray
                因子序列
            y：numpy.ndarray
                收益率序列
            w：numpy.ndarray
                无效输入数据
    返回:
            ic_ir:numpy.ndarray
                输出ic_ir数据
    '''
    stocknum = gl.get_value('stocknum')
    tradecount = gl.get_value('tradecount')
    x = x.reshape(stocknum,tradecount).T.reshape(stocknum*tradecount,)
    y = y.reshape(stocknum,tradecount).T.reshape(stocknum*tradecount,)
    w = np.isnan(x) + np.isnan(y)
    def _corr(x,y,w):
        index = np.where(w==False)[0]
        if (len(index) > 5):
            return np.nanmean((x-np.nanmean(x))*(y-np.nanmean(y)))/(np.nanstd(x)*np.nanstd(y))#np.corrcoef(x[index], y[index])[0,1]
        else:
            return np.NaN
    ic= [
        _corr(
            np.argsort(x[i*stocknum:(i+1)*stocknum]),
            np.argsort(y[i*stocknum:(i+1)*stocknum]),
            w[i*stocknum:(i+1)*stocknum]
        )
        for i in range(tradecount)
    ]
    return np.abs(np.nanmean(ic))/(np.nanstd(ic))

def _ic_ir_test(x,y,w):
    '''
    说明文档:
            计算适应度函数
    参数:
            x:numpy.ndarray
                因子序列
            y：numpy.ndarray
                收益率序列
            w：numpy.ndarray
                无效输入数据
    返回:
            ic_ir:numpy.ndarray
                输出ic_ir数据
    '''
    stocknum = gl.get_value('stocknum')
    tradecount = gl.get_value('tradecount')
    y1 = deepcopy(y)
    x1=deepcopy(x)
    for col in x.columns:
        x[col] = x[col].values.reshape(stocknum,tradecount).T.reshape(stocknum*tradecount,)
    for col in y.columns:
        y[col] = y[col].values.reshape(stocknum,tradecount).T.reshape(stocknum*tradecount,)
    w = np.isnan(x['ret'].values) + np.isnan(y['factor_0'].values)
    def _corr(x,y,w):
        index = np.where(w==False)[0]
        if (len(index) > 5):
            return np.nanmean((x-np.nanmean(x))*(y-np.nanmean(y)))/(np.nanstd(x)*np.nanstd(y))#np.corrcoef(x[index], y[index])[0,1]
        else:
            return np.NaN
    ic= [
        _corr(
            np.argsort(np.argsort(x['ret'].values[i*stocknum:(i+1)*stocknum])),
            np.argsort(np.argsort(y['factor_0'].values[i*stocknum:(i+1)*stocknum])),
            w[i*stocknum:(i+1)*stocknum]
        )
        for i in range(tradecount)
    ]
    return np.abs(np.nanmean(ic))/(np.nanstd(ic))

def _stat(x,y,w):
    '''
    说明文档:
            计算适应度函数
    参数:
            x:numpy.ndarray
                因子序列
            y：numpy.ndarray
                收益率序列
            w：numpy.ndarray
                无效输入数据
    返回:
            ic_ir:numpy.ndarray
                输出ic_ir数据
    '''
    stocknum = gl.get_value('stocknum')
    tradecount = gl.get_value('tradecount')
    x=x.T
    close = x[0]
    end_date = x[1]
    predict_chg = x[2]
    factor = y.reshape(stocknum,tradecount).T.reshape(stocknum*tradecount,).astype(c_float)
    price = close.reshape(stocknum,tradecount).T.reshape(stocknum*tradecount,).astype(c_float)
    end_date = end_date.reshape(stocknum,tradecount).T.reshape(stocknum*tradecount,).astype(c_int)
    predict_chg = predict_chg.reshape(stocknum,tradecount).T.reshape(stocknum*tradecount,).astype(c_float)
    params = np.array([30, 1000.0,0.9,0.1,0.005,0.02], dtype=c_float).astype(c_float)
    res = factor.copy()
    lib.fitness.restypes = c_float
    lib.fitness.argtypes = [
        npct.ndpointer(dtype=c_float, ndim=1, flags="C_CONTIGUOUS"),
        npct.ndpointer(dtype = c_float, ndim = 1, flags="C_CONTIGUOUS"),
        npct.ndpointer(dtype = c_float, ndim = 1, flags="C_CONTIGUOUS"),
        npct.ndpointer(dtype=c_float, ndim=1, flags="C_CONTIGUOUS"),
        npct.ndpointer(dtype=c_int, ndim=1, flags="C_CONTIGUOUS"),
        c_int,
        c_int,
        npct.ndpointer(dtype=c_float, ndim=1, flags="C_CONTIGUOUS"),
        c_int
    ]
    l = lib.fitness(res,factor,price,predict_chg,end_date,int(gl.get_value('stocknum')),int(gl.get_value('tradecount')),params,0)
    print(res[0])
    if(np.isnan(res[0])):
        op=1
    return res[0]

fitness_func_list = []
f_locals = copy.copy(sys._getframe().f_locals)
for key in f_locals.keys():
    if(
            callable(f_locals[key])==True
            and key[0:2]!='__'
            and key!= '_pydev_stop_at_break'
    ):
        fitness_func_list.append(key)
if __name__ == '__main__':
    data = pd.read_csv('C_test.csv')
    op=1