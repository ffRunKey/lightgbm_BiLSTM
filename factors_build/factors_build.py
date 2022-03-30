import lightgbm as lgb
import pandas as pd
import numpy as np
import datetime
from utils.utils import *
import itertools
import utils.new_oputils as oputils
import utils.globalvar as gl
from copy  import deepcopy
from constant.constant import (
    COM_DATE,
    COM_SEC,
    TRADE_NUM,
    STOCK_NUM,
)
from ini.ini import *
from constant.constant import *
gl._init()
from factors_build.constant import *
class Factors_Build():
    def __init__(self,factor_builds,env):
        self.factor_builds =factor_builds
        self.__env = env

    def get_env(self):
        return self.__env

    def set_env(self,env):
        self.__env = env


    def __factor_builds(self):
        def _fun_cal(X, _program, mode='run',feature_name = None):
            if (isinstance(_program[0], int)):
                return X[:, _program[0]], _program[1:]
            else:
                arg = []
                _program2 = None
                for i in range(_program[0].arity):
                    if (i == 0):
                        Y, _program2 = _fun_cal(X, _program[1:], mode)
                    else:
                        Y, _program2 = _fun_cal(X, _program2, mode)
                    arg.append(Y)
                if (_program[0].name not in Oputis):
                    if (mode == 'run'):
                        return _program[0](*arg), _program2
                    else:
                        return _program[0](*arg), _program2
                else:
                    return _program[0].function(*arg), _program2

        data = self.get_env().query_data(Factors_Data)
        res = self.get_env().query_data(Index_Data)
        for i in range(len(self.factor_builds._best_programs)):
            _program_ = self.factor_builds._best_programs[i]
            _program = _program_.program
            X = data.astype(float)
            Y, pro = _fun_cal(X.values, _program)
            if(i<50):
                res['factor_'+str(i)] = Y
            else:
                break
        return res


    def run(self,is_local = False):
        if is_local == False:
            market = self.get_env().query_data(Market_Data).get_data_serise()


            code = list(set(market[COM_SEC]))
            trade_date = list(set(market[COM_DATE]))
            trade_date.sort()
            gl.set_value(STOCK_NUM, len(code))
            gl.set_value(TRADE_NUM, len(trade_date))

            chg1 = conv(market[[COM_SEC, COM_DATE, Close]], trade_date=COM_DATE, sec_code=COM_SEC,
                        method=1).fillna(method='ffill')

            chg = unconv(chg1.shift(-1)/chg1.shift(0)-1,name = 'ret').reset_index()
            param_combinations = list(itertools.product(*[code, trade_date]))
            data_index = pd.DataFrame(
                param_combinations,
                columns=[COM_SEC, COM_DATE]
            ).sort_values(
                [COM_DATE, COM_SEC]
            )

            data = pd.merge(
                data_index,
                market,
                on = [COM_DATE, COM_SEC],
                how = 'left'
            ).sort_values(by = [COM_SEC,COM_DATE],ascending=[True,False])

            data = pd.merge(
                data,
                chg,
                on = [COM_DATE, COM_SEC],
                how = 'left'
            ).sort_values(by = [COM_SEC,COM_DATE],ascending=[True,False])

            self.get_env().add_data(data[[COM_SEC,COM_DATE,Ret]],Index_Data)
            del data[COM_DATE]
            del data[COM_SEC]
            del data['ret']
            self.get_env().add_data(data, Factors_Data)
            res = self.__factor_builds()

            data = pd.merge(
                market[[COM_DATE, COM_SEC]],
                res,
                on = [COM_DATE, COM_SEC],
                how = 'left'
            )
            h = pd.HDFStore(os.path.join(RESULTS,Factors_Data+'.h5'),'w')
            h['data'] = data
            h.close()
            h = pd.HDFStore(os.path.join(RESULTS,Index_Data+'.h5'),'w')
            h['data'] = data_index
            h.close()
        else:
            data = pd.read_hdf(os.path.join(RESULTS,Factors_Data+'.h5'))
            data_index = pd.read_hdf(os.path.join(RESULTS,Index_Data+'.h5'))
        self.get_env().add_data(data,Factors_Data)
        self.get_env().add_data(data_index,Index_Data)
        pass

    def feature_choice(
            self,
            days=21,
            is_local = False
    ):
        if(is_local):
            feature_info = pd.read_hdf(os.path.join(RESULTS,Feature_Info+'.h5'))
        else:
            factors = self.get_env().query_data(Factors_Data)
            factors = factors[
                factors[COM_DATE]>='2010-01-01'
            ]
            trade_list = list(set(factors[COM_DATE]))
            trade_list.sort()
            if len(trade_list)%days==0:
                n = int(len(trade_list)/days)-7
            else:
                n = int(len(trade_list)/days)-6
            feature_info = pd.DataFrame()
            begin_index = 147
            feature = list(factors.columns)
            feature.remove(COM_SEC)
            feature.remove(COM_DATE)
            feature.remove(Ret)
            for i in range(n):

                end_date = days*i+begin_index-21
                begin_date = days * i
                trade_date = days*i+begin_index
                print(trade_list[trade_date])
                train_data = factors[
                    (factors[COM_DATE]<=trade_list[end_date]) &
                    (factors[COM_DATE]>=trade_list[begin_date])
                ]
                model = lgb.LGBMRegressor()
                model.fit(train_data[feature],train_data[Ret])
                feature_info_cell = pd.DataFrame(columns=Info_Fields)
                feature_info_cell[Importance] = model.feature_importances_
                feature_info_cell[Feature_Name] = model.feature_name_
                feature_info_cell = feature_info_cell.sort_values(by=Importance).tail(10)
                feature_info_cell[COM_DATE] = trade_list[trade_date]
                feature_info = pd.concat(
                    [feature_info,feature_info_cell],
                    axis = 0
                )
            h = pd.HDFStore(os.path.join(RESULTS,Feature_Info+'.h5'),'w')
            h['data'] = feature_info
            h.close()
        self.get_env().add_data(feature_info,Feature_Info)
        pass

    def clean(self,is_local = False):
        def __clean(df,feature):
            for col in feature:
                df[col] = df[col].fillna(np.nanmedian(df[col]))
                df[col] = np.argsort(np.argsort(df[col]))
                df[col] =  (df[col]-np.nanmean( df[col]))/( np.nanstd(df[col])+0.0000001)
            return df

        if is_local:
            factors = pd.read_hdf(os.path.join(RESULTS,Factors_Clean_Data+'.h5'))
        else:
            factors = self.get_env().query_data(Factors_Data)
            data_index = self.get_env().query_data(Index_Data)

            feature = list(factors.columns)
            feature.remove(COM_SEC)
            feature.remove(COM_DATE)
            factors = factors.groupby(COM_DATE).apply(lambda df:__clean(df,feature))
            factors = pd.merge(
                data_index,
                factors,
                on=[COM_DATE,COM_SEC],
                how = 'left'
            )
            h = pd.HDFStore(os.path.join(RESULTS,Factors_Clean_Data+'.h5'),'w')
            h['data'] = factors
            h.close()
        self.get_env().add_data(factors,Factors_Clean_Data)