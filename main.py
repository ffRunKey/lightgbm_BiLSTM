import pandas as pd
from ini.ini import *
from constant.constant import *
from enviorment.enviorment import Environment
from data_serise.data_serise import Daily_Data_Serise
from factors_build.factors_build import Factors_Build
import pickle
from deep_learning.deep_learning import Deep_Learning
from analysis.analysis import Analysis

if __name__ == '__main__':
    '''
        读取市场数据market_total
    '''
    market_file_name = 'market_total.h5'
    factor_build_pick_filename = 'factors_build.pkl'
    index_file_name = 'index_daily.h5'
    '''
        将市场日线行情数据加入环境类
    '''

    env  = Environment()
    dds = Daily_Data_Serise()
    dds.read_local_file(market_file_name)
    env.add_data(dds,Market_Data)

    dds2 = Daily_Data_Serise()
    dds2.read_local_file(index_file_name)
    env.add_data(dds2,Index_Data)
    # with open(os.path.join(DEV_PATH,'factors_build',factor_build_pick_filename),'rb') as f:
    #     fb = pickle.load(f)
    # factor_builds = Factors_Build(fb,env)
    # factor_builds.run(is_local=True)
    # factor_builds.feature_choice(is_local = True)
    # factor_builds.clean(is_local = True)
    dl = Deep_Learning(env)
    # dl.data_process()
    # dl.build_net_blstm()
    dl.train(islocal = True)
    analysis = Analysis(env)
    analysis.backtest()
    analysis.analysis()