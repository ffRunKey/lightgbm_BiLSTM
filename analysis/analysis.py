import numpy as np
import pandas as pd
from constant.constant import *
from ini.ini import *
from copy import deepcopy
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
class Analysis:
    def __init__(self,env,sillping = 0.0000,commison = 0.002):
        self.__env = env
        self.sillping = sillping
        self.commison = commison

    def get_env(self):
        return self.__env

    def set_env(self,env):
        self.__env = env

    def backtest(self,n=75):
        def change_date(df):
            return df.split('.')[0]
        def change_date2(df):
            return df[0:4]+'-'+df[4:6]+'-'+df[6:8]
        y_hat = self.get_env().query_data(Y_HAT)
        y_hat[COM_DATE] = list(map(change_date,y_hat[COM_DATE].values))
        y_hat['y_hat'] = -y_hat['y_hat']
        market = self.get_env().query_data(Market_Data).get_data_serise()
        market[COM_DATE] = list(map(change_date2,market[COM_DATE]))
        y_hat = pd.merge(
            market[[COM_DATE,COM_SEC]],
            y_hat,
            on = [COM_DATE,COM_SEC],
            how = 'left'
        ).dropna()
        market = market.set_index([COM_DATE,COM_SEC])
        open =  market['open'].unstack()
        ret = open.shift(-2)/open.shift(-1)-1
        stock_list = list(ret.columns)
        stock_list.sort()
        w = y_hat.set_index([COM_DATE, COM_SEC])[Y_HAT].unstack()
        w_ = np.argsort(np.argsort(w.values, axis=1), axis=1)
        weight_values = deepcopy(w_).astype(float)
        weight_values[
            w_ < n
            ] = 1 / n
        weight_values[
            w_ >= n
            ] = 0
        weight = pd.DataFrame(data = weight_values,index = w.index,columns = w.columns)
        turnover = 1*np.sum(np.abs(weight-weight.shift(1)),axis = 1)/2
        stock_list_x = [sec+'_x' for sec in stock_list]
        stock_list_y = [sec+'_y' for sec in stock_list]
        for sec in set(stock_list)-set(weight.columns):
            weight[sec] = 0
        pct_chg = pd.merge(
            ret,
            weight,
            right_index=True,
            left_index=True,
            how = 'outer',
        ).fillna(method='ffill').fillna(method='bfill')
        turnover = pd.merge(
            pct_chg,
            pd.DataFrame(turnover),
            right_index=True,
            left_index=True,
            how='outer',
        ).fillna(0.0).rename(columns = {0:'turnover'})['turnover']
        pct_chg = pd.DataFrame(
            data = (1+ np.nansum(
                pct_chg[stock_list_x].values * pct_chg[stock_list_y].values, axis=1)
            )*(1-turnover.values*self.sillping)*(1-turnover.values*self.commison)-1,
            columns=['pct_chg'],
            index = pct_chg.index
        )
        self.get_env().add_data(pct_chg,'pct_chg')
        self.get_env().add_data(turnover,'turnover')
        pass

    def analysis(self,index_code = '000985.SH'):
        nav_pct = self.get_env().query_data('pct_chg').rename(columns={'pct_chg': 'strategy'})
        index = self.get_env().query_data(Index_Data).get_data_serise()
        index = index[
            (index[COM_DATE]>=nav_pct.index[0])&
            (index[COM_DATE]<=nav_pct.index[-1])
        ].set_index([COM_DATE])
        turnover = self.get_env().query_data('turnover')
        open =  index['open']
        ret = (open.shift(-2)/open.shift(-1)-1)
        benchmark = (1+ret).cumprod().fillna(method = 'ffill')

        nav = (1+nav_pct).cumprod()
        res = pd.DataFrame(columns = ['累计收益','年化收益','年化波动率','夏普值','最大回撤','年化换手率','年化交易成本'])
        res.loc['我的策略'] = [
            nav.iloc[-1].values[0]-1,
            np.power(nav.iloc[-1].values[0]/nav.iloc[0].values[0],252/len(nav))-1,
            np.nanstd(nav_pct)*np.sqrt(252),
            (np.power(nav.iloc[-1].values[0]/nav.iloc[0].values[0],252/len(nav))-1-0.035)/(np.nanstd(nav_pct)*np.sqrt(252)),
            1-min(nav['strategy']/ np.maximum.accumulate(nav['strategy'])),
            turnover.sum()*252/len(nav),
            np.power((1 + self.sillping) * (1 + self.commison), turnover.sum() * 252 / len(nav)) - 1
        ]
        res.loc['基准'] = [
            benchmark.iloc[-1]- 1,
            np.power(benchmark.iloc[-1] / benchmark.iloc[0], 252 / len(benchmark)) - 1,
            np.nanstd(ret) * np.sqrt(252),
            (np.power(benchmark.iloc[-1] / benchmark.iloc[0], 252 / len(benchmark)) - 1 - 0.035) / (
                        np.nanstd(ret) * np.sqrt(252)),
            1 - min(benchmark / np.maximum.accumulate(benchmark)),
            0,
            0
        ]
        benchmark.index = pd.to_datetime(benchmark.index)
        nav.index = pd.to_datetime(nav.index)
        fig, ax = plt.subplots()
        ax.plot(nav,label = '我的策略')
        ax.plot(benchmark,label = '基准')
        ax.set_xlabel('时间')
        ax.set_ylabel('净值')
        ax.set_title('收益曲线')
        ax.legend()
        plt.savefig(os.path.join(RESULTS,'res.png'))
        plt.show()
        plt.close()
        res.to_excel(os.path.join(RESULTS,'res.xlsx'))
        pass