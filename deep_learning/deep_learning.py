import pandas as pd
import numpy as np
from ini.ini import *
from constant.constant import *
import time
import pickle
# import keras as ks
class Deep_Learning:
    def __init__(self,env):
        self.__env = env
        self.__model = None
        pass

    def get_env(self):
        return self.__env

    def set_env(self,env):
        self.__env = env

    def get_model(self):
        return self.__model

    def set_model(self, model):
        self.__model = model

    def build_net_blstm(self):
        model = ks.Sequential()
        model.add(
            ks.layers.Bidirectional(ks.layers.LSTM(
                50
            ),input_shape=(11,10))
        )
        model.add(
            ks.layers.Dropout(0.01)
        )
        model.add(ks.layers.Dense(256))
        model.add(
            ks.layers.Dropout(0.01)
        )
        model.add(ks.layers.Dense(64))
        model.add(ks.layers.Dense(1))
        model.compile(optimizer='sgd', loss='mse')
        model.summary()
        self.set_model(model)

    def build_net_cnn(self):
        model = ks.Sequential()
        model.add(
            ks.layers.Conv2D(
                256,
                kernel_size=(5, 5),
                strides=(1, 1),
                activation='relu',
                input_shape=(11,50,1)
            )
        )
        # model.add(
        #     ks.layers.MaxPooling2D(
        #         pool_size=(2, 2),
        #         strides=(2, 2)
        #     )
        # )
        model.add(
            ks.layers.Conv2D(
                256,
                kernel_size=(3, 3),
                strides=(1, 1),
                activation='relu',
            )
        )
        model.add(
            ks.layers.MaxPooling2D(
                pool_size=(2, 2),
                strides=(2, 2)
            )
        )
        model.add(
            ks.layers.Dropout(0.01)
        )
        model.add(ks.layers.Flatten())
        model.add(ks.layers.Dense(256))
        model.add(
            ks.layers.Dropout(0.01)
        )
        model.add(ks.layers.Dense(64))
        model.add(ks.layers.Dense(1))
        model.compile(optimizer='sgd', loss='mse')
        model.summary()
        self.set_model(model)

    def train(self,islocal=False):
        if islocal:
            res = pd.read_csv(os.path.join(RESULTS, Y_HAT + '2.csv'))
        else:
            model = self.get_model()
            date_list = os.listdir(os.path.join(RESULTS,'train'))
            res = pd.DataFrame()
            for date in date_list:
                with open(os.path.join(RESULTS,'train',date),'rb') as f:
                    data = pickle.load(f)
                model.fit(
                    np.array(data['train']).reshape((len(data['train']), 11, 10)),
                    np.array(data['label']),
                    batch_size=1024,
                    epochs=400,
                )
                data_cell = pd.DataFrame(columns=[COM_SEC, COM_DATE, Y_HAT])
                data_cell[COM_SEC] = data['sec']
                data_cell[COM_DATE] = date.split('.')[0]
                data_cell[Y_HAT] = model.predict(np.array(data['predict']).reshape((len(data['predict']), 11, 10)))
                res = pd.concat(
                    [res,data_cell],
                    axis = 0
                )
                res.to_csv(os.path.join(RESULTS, Y_HAT + '2.csv'),index = False)
                # print(
                #     np.corrcoef(
                #         [
                #             predict.reshape((len(predict),)), np.array(data['label']).reshape((len(data['label'], )))
                #         ]
                #     )[0, 1]
                # )
            res.to_csv(os.path.join(RESULTS, Y_HAT + '2.csv'), index=False)
        self.get_env().add_data(res,Y_HAT)

    def data_process(self):
        '''
            数据清洗模块，将数据进行清洗
        '''
        factors = self.get_env().query_data(Factors_Clean_Data)
        factors_info = self.get_env().query_data(Feature_Info)
        trade_list = list(set(factors[COM_DATE]))
        trade_list.sort()
        trade_info_list  = list(set(factors_info[COM_DATE]))
        trade_info_list.sort()
        n = len(trade_info_list)-10
        begin_index = 10
        for date in range(n):
            t = time.time()
            data_gen = {}
            pic_list = []
            ret_list = []
            predict_list = []
            sec_list = []
            trade_date = trade_info_list[begin_index+date-1]
            factors_info_use = list(
                factors_info[
                    factors_info[COM_DATE] == trade_date
                ]['feature_name']
            )
            sec_code_list = list(set(factors[COM_SEC]))
            sec_code_list.sort()
            for i in range(9):
                print(i)
                end_date = trade_info_list[trade_info_list.index(trade_date) - i-1]
                begin_date = trade_list[trade_list.index(end_date) - 10]
                factors_use = factors[
                    (factors[COM_DATE] >= begin_date) &
                    (factors[COM_DATE] <= end_date)
                ]
                for sec in sec_code_list:
                    factors_sec = factors_use[
                        factors_use[COM_SEC]==sec
                    ]
                    pic = factors_sec[factors_info_use]
                    nan_num = np.sum(np.isnan(pic.values))
                    total_num = np.size(pic.values)
                    if(nan_num/total_num<0.4):
                        pic_list.append(
                            pic.fillna(0.0).values
                        )
                        ret_list.append(
                            factors_sec.tail(1)['ret'].fillna(0.0).values[0]
                        )
            end_date = trade_info_list[trade_info_list.index(trade_date)]
            begin_date = trade_list[trade_list.index(end_date) - 10]
            factors_use = factors[
                (factors[COM_DATE] >= begin_date) &
                (factors[COM_DATE] <= end_date)
            ]
            for sec in sec_code_list:
                factors_sec = factors_use[
                    factors_use[COM_SEC] == sec
                    ]
                pic = factors_sec[factors_info_use]
                predict_list.append(
                    pic.fillna(0.0).values
                )
                sec_list.append(
                    sec
                )
            data_gen['predict'] = predict_list
            data_gen['sec'] = sec_list
            data_gen['train'] = pic_list
            data_gen['label'] = ret_list
            with open(os.path.join(RESULTS,'train',trade_date+'.pkl'),'wb') as f:
                pickle.dump(data_gen,f)
            print(time.time()-t)