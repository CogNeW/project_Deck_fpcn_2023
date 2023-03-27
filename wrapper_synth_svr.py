#!/usr/bin/env pyhton

# Path where features csv resides
x_vars='FPCNB_avg_output.csv'


# Path where target csv resides
y_var='shiftcost_ACC_pre.csv'

# initialize synthetic data gen variables
# Read about this process at the following links:
# 1. https://sdv.dev/blog/synthetic-clones-for-ml/
# 2. https://github.com/sdv-dev/SDV/discussions/980
# 3. https://github.com/sdv-dev/SDV/issues/222
# 4. https://sdv.dev/SDV/user_guides/evaluation/single_table_metrics.html

synth_sample_size=10000 # number of samples to create for synthetic data.
eps=10000 # set the number of epochs
batch=1000 # set the number of values to train each epoch on


import numpy as np
# Specify halving grid search params via a dictionary
grid_CVparams={'svr__kernel':['linear','rbf'] ,
          'svr__C': [1e0, 1e1, 1e2, 1e3],
              'svr__epsilon':[.001,.01,.1,1],
              'svr__gamma':np.logspace(-9, 3, 13)}


# specify the number of permutations to conduct for the permutation test.
permutations=10000


# Specify the output path
out_path='/test/'

# Import the synth svr script
from synth_svr import synth_svr

# Run
print('running synth_svr')
synth_svr(x_vars,y_var, synth_sample_size, eps, batch, grid_CVparams, permutations, out_path)
