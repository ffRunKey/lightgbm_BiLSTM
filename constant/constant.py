import os

DEV_PATH = os.path.dirname(os.path.dirname(__file__))
DATA_BASE_PATH = os.path.join(DEV_PATH,'data_base')
RESULTS = os.path.join(DEV_PATH,'results')


COM_DATE = 'trade_date'
COM_SEC = 'sec_code'
COM_INDUSTRY_CODE = 'industry_code'
COM_INDEX_CODE = 'index_code'
COM_FINANCE_DATE = 'ann_date'
Y_HAT = 'y_hat'

STOCK_NUM = 'stock_num'
TRADE_NUM = 'trade_num'


H5_FILE = 'h5'
CSV_FILE = 'csv'