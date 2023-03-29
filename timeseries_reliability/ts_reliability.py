#!/usr/bin/env python
def ts_reliability(sub, tshigh, tslow):
    '''
    1. Load in each time series txt file with numpy.loadtxt
    2. Cut each time-series txt file at 0:480 and 480:960
    3. convert numpy array to pandas df
    4. create TR column from index which is equivalent to rating occurance 
    5. pd.melt from https://rowannicholls.github.io/python/statistics/agreement/intraclass_correlation.html and put data into long format
    6. Calculate ICC for beginning and end and then average these two scores
    7. Do this for low, high, and high - low contrast for each task and rest data.
    '''



    import numpy as np
    import pandas as pd    
    import pingouin as pg


 


# High data

    high = np.loadtxt(tshigh)
    
    print('Now performing split half from subject TR count')
    print()
    
    # split half beginning
    high_begin = high[0:480,] # take only the first 480 TRs
    high_begin= high_begin[:,~np.all(high_begin == 0, axis = 0)] # remove any columns with only 0s
    high_begin_df = pd.DataFrame(high_begin).corr() # get correlation matrix from high beginning TRs
    high_begin_subtract = high_begin_df
    high_begin_df['timepoint'] = 1 # create timepoint ID
    high_begin_long = pd.melt(high_begin_df, id_vars=['timepoint'], value_vars=list(high_begin_df)[:-1])
    # end
    high_end = high[480:960,] # take only the remaining 480 TRs
    high_end= high_end[:,~np.all(high_end== 0, axis = 0)] # remove any columns with only 0s
    high_end_df = pd.DataFrame(high_end).corr() # get corr of high end TRs
    high_end_subtract = high_end_df
    high_end_df['timepoint'] = 2
    high_end_long = pd.melt(high_end_df, id_vars=['timepoint'], value_vars=list(high_end_df)[:-1])

    
    high_cor = pd.concat([high_begin_long, high_end_long])
    print("Getting reliability of High condition")
    print()
    icc_high = pg.intraclass_corr(data=high_cor, targets='timepoint', raters='variable', ratings='value')


    icc_high_val = icc_high['ICC'][2]

# Low data
    low = np.loadtxt(tslow)

    low_begin = low[0:480,] # take only the first 480 TRs
    low_begin= low_begin[:,~np.all(low_begin == 0, axis = 0)] # remove any columns with only 0s
    low_begin_df = pd.DataFrame(low_begin).corr() # get correlation matrix from low beginning TRs
    low_begin_subtract = low_begin_df
    low_begin_df['timepoint'] = 1 # create timepoint ID
    low_begin_long = pd.melt(low_begin_df, id_vars=['timepoint'], value_vars=list(low_begin_df)[:-1])
    # end
    low_end = low[480:960,] # take only the remaining 480 TRs
    low_end= low_end[:,~np.all(low_end== 0, axis = 0)] # remove any columns with only 0s
    low_end_df = pd.DataFrame(low_end).corr() # get corr of low end TRs
    low_end_subtract = low_end_df
    low_end_df['timepoint'] = 2
    low_end_long = pd.melt(low_end_df, id_vars=['timepoint'], value_vars=list(low_end_df)[:-1])

    
    low_cor = pd.concat([low_begin_long, low_end_long])

    print('Getting Reliability of Low condition')
    print()
    icc_low = pg.intraclass_corr(data=low_cor, targets='timepoint', raters='variable', ratings='value')
    icc_low_val = icc_low['ICC'][2]


    # subtraction

    ### need to:
    # 1. subtract all columns which are numeric not strings
    begin_contrast = high_begin_subtract.subtract(low_begin_subtract)

    begin_contrast['timepoint']=1
    begin_contrast_long = pd.melt(begin_contrast, id_vars=['timepoint'], value_vars=list(begin_contrast)[:-1])

    end_contrast = high_end_subtract.subtract(low_end_subtract)
    end_contrast['timepoint']=2
    end_contrast_long = pd.melt(end_contrast, id_vars=['timepoint'], value_vars=list(end_contrast)[:-1])
    contrast_cor =pd.concat([begin_contrast_long, end_contrast_long])

    print('Getting reliability of contrast, High - Low conditions')
    print()
    icc_contrast = pg.intraclass_corr(data=contrast_cor, targets='timepoint', raters='variable', ratings='value')
    icc_contrast_val = icc_contrast['ICC'][2]

    return icc_low_val, icc_high_val, icc_contrast_val
