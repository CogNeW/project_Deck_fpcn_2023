#!/usr/bin/env python3.9

import pandas as pd
from ts_reliability import ts_reliability
import os
import warnings
warnings.filterwarnings("ignore")

subjects=['sub-DP5001',      'sub-DP5007',      'sub-DP5013',      'sub-DP5020',      'sub-DP5026',      'sub-DP5032',      'sub-DP5038',
'sub-DP5002',      'sub-DP5008',      'sub-DP5014',      'sub-DP5021',      'sub-DP5027',      'sub-DP5033',      'sub-DP5040',
'sub-DP5003',      'sub-DP5009',      'sub-DP5016',      'sub-DP5022',      'sub-DP5028',      'sub-DP5034',      'sub-DP5041',
'sub-DP5004',      'sub-DP5010',      'sub-DP5017',      'sub-DP5023',      'sub-DP5029',      'sub-DP5035',      'sub-DP5044',
'sub-DP5005',      'sub-DP5011',      'sub-DP5018',      'sub-DP5024',      'sub-DP5030',      'sub-DP5036',      'sub-DP5047',
'sub-DP5006',      'sub-DP5012',      'sub-DP5019',      'sub-DP5025',      'sub-DP5031',      'sub-DP5037',      'sub-DP5050']

out = '/path/to/out/directory/'

task = 'ses-<put_task_name_here>'


low_list = []
high_list = []
contrast_list = []


for sub in subjects:
    print('Now obtaining reliability measures for ' + sub)

    tshigh = '/path/to_ts/data/directory/' + sub + '/'+task + '/' + sub +  '_' + task + '_high_li2019_ts.txt'  
    tslow = '/path/to_ts/data/directory/'+ sub + '/'+task + '/'  + sub + '_' + task + '_low_li2019_ts.txt'

    high_path = os.path.exists(tshigh)
    low_path = os.path.exists(tslow)

    if ((high_path  == 1) and (low_path == 1)):

        print('High and low path exist for ' + sub)

        icc_low, icc_high, icc_contrast=ts_reliability(sub, tshigh,tslow )

        low_list.append(icc_low)
        high_list.append(icc_high)
        contrast_list.append(icc_contrast)
    else:
        print("Subject doesn't have appropriate files, please check their time-series data")


# convert to dataframe
df = pd.DataFrame({'ICC Low':low_list, 'ICC High': high_list, 'ICC Contrast':contrast_list})

df.to_csv(out+ task + '_ts_reliability.csv', index=None)
