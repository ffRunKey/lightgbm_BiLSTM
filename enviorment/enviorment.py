'''
    @author:SHUI YUE
    @date:2021-12-2
'''
import pandas as pd
import os
from constant.constant import DATA_BASE_PATH

class Environment:
    def __init__(self):
        '''
            函数:
                    初始化环境类
            参数:
                    无
            返回:
                    无
        '''
        self.__data = {}
        self.__name_data = []
        self.__report = {}
        self.__name_report = []
        self.__account=None
        pass

    def set_account(self,account):
        '''
            函数:
                    设置账户类
            参数:
                    account：object
                        账户类
            返回:
                    无
        '''
        self.__account = account

    def get_account(self):
        '''
            函数:
                    返回账户类
            参数:
                    无
            返回:
                    __account:object
                        账户
        '''
        return self.__account

    def set_data(self,data):
        '''
            函数:
                    设置数据字典
            参数:
                    data:object
                        数剧类
            返回:
                    无
        '''
        self.__data = data
        self.__name_data = list(data.keys())

    def get_data(self):
        '''
            函数:
                    返回数据类
            参数:
                    无
            返回:
                    无
        '''
        return self.__data

    def add_data(self,data,name):
        '''
            函数:
                    增加数据帧
            参数:
                    data:object,
                        数据帧
                    name:str
                        数据帧名字
            返回:
                    无
        '''
        self.__data[name] = data
        self.__name_data.append(name)

    def del_data(self,name):
        '''
            函数:
                    删除数据帧
            参数:
                    name:str
                        数据帧名称
            返回:
                    无
        '''
        del self.__data[name]
        self.__name_data.remove(name)

    def query_data(self,name):
        '''
            函数:
                    查询数据帧
            参数:
                    name:str
                        数据帧名称
            返回:
                    __data:object
                        数据帧
        '''
        return self.__data[name]


if __name__ == '__main__':
    pass