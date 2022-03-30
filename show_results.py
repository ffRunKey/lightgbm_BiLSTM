import pandas as pd
import os
from constant.constant import RESULTS
if __name__ == '__main__':
    factor_clean = pd.read_hdf(os.path.join(RESULTS,'feature_info.h5'))
    op=1