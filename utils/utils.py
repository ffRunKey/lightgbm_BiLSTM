import pandas as pd
import numpy as np
import ast
import os
from constant.constant import CSV_FILE,H5_FILE
def save(data,path,file_name):
    if(file_name.split('.')[-1]==CSV_FILE):
        data.to_csv(os.path.join(path,file_name))
    elif(file_name.split('.')[-1]==H5_FILE):
        h = pd.HDFStore(os.path.join(path,file_name),'w')
        h['data'] = data
        h.close()
    else:
        return

def conv(data, trade_date="trade_date", sec_code="stock_code", method=1):
    """
    文档名称:
        数据转换为以日期为列，证券号为行的数据格式
    参数：
            data:pandas.DataFrame
                需要计算的数据
            TradeDateIni:str defaults:"trade_date"
                日期做为索引
            CodeIni:str defaults:"stock_code"
                证券号作为行
    返回：
            data_o:pandas.DataFrame
                计算得到的结果
    """

    if method == 0:
        return pd.pivot_table(
            data=data, index=[trade_date], columns=[sec_code], dropna=False
        )
    else:
        return data.set_index([trade_date, sec_code]).unstack(fill_value=np.NaN)


def unconv(data, name="col"):
    """
    文档说明:
    参数:
            data:pandas.DataFrame
                所需转换的数据
            name:str defualts:"col"
                数据名称，默认'col'
    返回：
            data_o:pandas.DataFrame
                所需计算结果
    """
    data = data.stack().replace({np.inf: np.NaN, -np.inf: np.NaN})
    return pd.DataFrame(data=data.values, index=data.index, columns=[name])



class CallCollector(ast.NodeVisitor):
    '''
        该类用于提取python文件中的函数
    '''
    def __init__(self):
        self.calls = []
        self.current = None

    def visit_Call(self, node):
        # new call, trace the function expression
        self.current = ''
        self.visit(node.func)
        self.calls.append(self.current)
        self.current = None

    def generic_visit(self, node):
        if self.current is not None:
            print(
                "warning: {} node in function expression not supported".format(
                    node.__class__.__name__
                )
            )
        super(CallCollector, self).generic_visit(node)

    # record the func expression
    def visit_Name(self, node):
        if self.current is None:
            return
        self.current += node.id

    def visit_Attribute(self, node):
        if self.current is None:
            self.generic_visit(node)
        self.visit(node.value)
        self.current += '.' + node.attr

