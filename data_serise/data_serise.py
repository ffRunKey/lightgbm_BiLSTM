'''
    @author:SHUI YUE
    @date:2021-12-2
'''
import pandas as pd
import os
import tushare as ts
from constant.constant import (
    DATA_BASE_PATH
)
from constant.constant import *
ts.set_token('d20adcb83377edd4aac4952601b4fd83c2fa0c219aa2dc753d9b9853')
class Base_Data_Serise:
    def __init__(self):
        '''
            函数:
            参数:
            返回:
        '''
        self.__data_serise = None
        self.__tushare =ts.pro_api()
        pass


    def get_tushare(self):
        return self.__tushare

    def get_data_serise(
            self
    ):
        '''
            函数:
            参数:
            返回:
        '''
        return self.__data_serise

    def set_data_serise(
            self,
            data_serise
    ):
        '''
            函数:
            参数:
            返回:
        '''
        self.__data_serise = data_serise

class Level2_Data_Serise(Base_Data_Serise):
    def __init__(self):
        Base_Data_Serise.__init__(self)
        pass


class Daily_Data_Serise(Base_Data_Serise):
    def __init__(self):
        Base_Data_Serise.__init__(self)
        pass

    def pickle_as_local_file(self,file_name,path= DATA_BASE_PATH):
        with pd.HDFStore(
            os.path.join(
                path,
                file_name
            ),
            mode = 'w',
        ) as f:
            f['data'] = self.get_data_serise()
        f.close()

    def read_local_file(self,file_name,path = DATA_BASE_PATH):
        data = pd.read_hdf(os.path.join(
            path,
            file_name
        )).rename(columns = {'ts_code':COM_SEC})
        self.set_data_serise(data)

    def read_from_tushare(
            self,
            begin_date,
            end_date
    ):
        trade_cal = self.get_tushare().trade_cal(start_date=begin_date,end_date=end_date,is_open = 1)['cal_date'].tolist()
        count = 0
        for date in trade_cal:
            print(date)
            if(count==0):
                data = self.get_tushare().daily(trade_date=date)
            else:
                data = pd.concat([data,self.get_tushare().daily(trade_date=date)],axis = 0)
            count = count +1
        self.set_data_serise(data)




class Mintue_Data_Serise(Base_Data_Serise):
    def __init__(self):
        Base_Data_Serise.__init__(self)
        pass



if __name__ == '__main__':
    daily_data_market = Daily_Data_Serise()
    daily_data_market.read_from_tushare('20151231','20171231')
    daily_data_market.pickle_as_local_file('market.h5')
    pass