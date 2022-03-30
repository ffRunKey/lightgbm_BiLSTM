import pandas as pd
import os
from constant.constant import *
if __name__ == '__main__':
    data = pd.read_excel(
        os.path.join(DATA_BASE_PATH,'index.xlsx')
    ).rename(columns = {'OPEN':'open','HIGH':'high','LOW':'low','CLOSE':'close','VOLUME':'volume','PRE_CLOSE':'pre_close','DateTime':'trade_date'})
    data = data.set_index('trade_date')
    data.index = data.index.strftime('%Y-%m-%d')
    data = data.reset_index().rename(columns = {'index':'trade_date'})
    h = pd.HDFStore('index_daily.h5','w')
    h['data'] = data
    h.close()
    op=1