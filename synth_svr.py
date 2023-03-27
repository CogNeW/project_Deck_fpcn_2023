#!/usr/bin/env python








def synth_svr(x_vars,y_var, synth_sample_size, eps, batch, grid_CVparams, permutations, out_path):

    """
    Creates synthetic data and runs SVR on the synthetic data and the hold
     out test set. Additionally, this function performs a permutation test 
     and compares the null to the test statistic This function also has the 
     ability to bootstrap the outcome of interest, either MSE or R2
    """


    # load dependencies
    import pandas as pd 
    import numpy as np
    from sklearn.svm import SVR
    from sklearn.model_selection import KFold, LeaveOneOut, train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    from sdv.tabular import CTGAN
    import matplotlib.pyplot as plt
    import seaborn as sns
    from time import time
    import os
    import sys
    from utils.pipeline_funcs import learning_curves, transform_data, synth, svr_selectCV, perm_svr, plot_models
    from datetime import datetime






    start1 = time()
    # append system path
    sys.path.append('/Users/neurouser1/Documents/Software/generalUse_models/')

    # Make a reports directory to store all images and outputs of SVR
    dir = 'reports'
    reports_path = os.path.join(out_path, dir)
    if not os.path.exists(reports_path):
        os.umask(0)
        os.makedirs(reports_path, mode=0o777)




    # load data
    X = pd.read_csv(x_vars)
    features_str = os.path.basename(x_vars).split('.')[0]
    print(features_str)
    y = pd.read_csv(y_var)
    targets_str = os.path.basename(y_var).split('.')[0]
    print(targets_str)
###############################################################################

    # call transformer and transform x and y
    print('Transforming data')
    X_normed, y_normed, col_names = transform_data(X, y, reports_path)
    print('Your x and y variables have been transformed to be more normal. Check ' + reports_path + ' for the resulting distributions')

###############################################################################

###############################################################################

    # Outer loop taking the best model and training and testing on all the data using the best model.
    # define cross-validation method to use
    
    ...
    # define nested cross-validation scheme
    cv_outer = KFold(n_splits=5)
    fold_n = cv_outer.get_n_splits(X_normed)
    # enumerate splits
    outer_mse = list()
    outer_r2 = list()
    perm_pvalues = list()
    outer_c = list()
    outer_kernel = list()
    outer_gamma = list()
    outer_epsilon = list()
    # initialize fold counter
    fold = 0
    for train_ix, test_ix, in cv_outer.split(X_normed):
        fold += 1 
        print('Currently running fold #:',fold)
        
        # split data into train and test sets
        
        X_train, X_test = X_normed[train_ix, :], X_normed[test_ix, :]
        y_train, y_test = y_normed[train_ix], y_normed[test_ix]


        # append x and y np arrays together
        x_y = np.append(X_train, y_train, axis=1)
        # create dataframe of x and y array and create column names
        x_y_df = pd.DataFrame(x_y, columns = col_names)


    ###############################################################################
        # create synthetic data using sdv CTGAN model
        verbosity=0 # forcing user to see this for now, to help diagnostics.
        synth_data = synth(x_y_df, eps, batch, verbosity, col_names, synth_sample_size)

    ###############################################################################
        # turn on if want report on synthetic data
        # print('Generating a quality control report for the synthetic data')
        # from utils.pipeline_funcs import synth_report

        # synth_report(x_y_df, synth_data, reports_path)


    ###############################################################################

        print("Now creating an SVR model for the synthetic data")
        # get all data from synthetic data dataframe
        x_synth = synth_data.iloc[:,:-1].to_numpy()
        y_synth = synth_data.iloc[:,-1:].to_numpy()
        # createtrain test split of synthetic data
        synth_X_train, synth_X_test, synth_y_train, synth_y_test = train_test_split(
            x_synth, y_synth, test_size=0.33)


###############################################################################


        # fit synthetic data to svr
        print('\n')
        print('\n')
        print('Now Performing model selection using Halving Grid Search CV, this could take some time')
        print('\n')
        print('\n')
        print('Halving Grid Search CV your synthetic model and obtaining the best fit')
        start = time() # time how long the CTGAN model takes to converge
        # fit CTGAN model
        synth_svr, best_SVR, hyp_score, r_squared = svr_selectCV(synth_X_train, synth_X_test, synth_y_train, synth_y_test, grid_CVparams)
        end = time()
        print("This process took:", end - start, 'seconds')

        # plot train and test curves
        print('Plotting learning curves for model selection')
        learning_curves(reports_path, synth_svr, grid_CVparams, 1, fold, targets_str)

        print('\n')
        print('\n')
        print('Appending the best test MSE and R-squared from the best models from inner Grid CV')
        outer_mse.append(hyp_score)
        outer_r2.append(r_squared)

        outer_c.append(best_SVR.C)
        outer_kernel.append(best_SVR.kernel)
        outer_epsilon.append(best_SVR.epsilon)
        outer_gamma.append(best_SVR.gamma)

    print('MSE: %.3f (%.3f)' % (np.median(outer_mse), np.std(outer_mse)))
    
    print('r-squared: %.3f (%.3f)' % (np.median(outer_r2), np.std(outer_r2)))


    filestr =  "SVR_OUTPUT_"+features_str+"-vs-"+targets_str+".txt"
    
    # write report of SVR
    now = datetime.now()
    now = now.strftime("%Y-%m-%d_%H-%M-%S")
    
    f = open(reports_path + '/' + filestr,"w+")
    f.writelines(now)
    f.writelines('\n Mean MSE: %.3f (%.3f)' % (np.median(outer_mse), np.std(outer_mse)))
    f.writelines('\n r-squared: %.3f (%.3f)' % (np.median(outer_r2), np.std(outer_r2)))
    f.close()

    # create dataframe from model metrics
    metrics_df = pd.DataFrame({'MSE array':outer_mse, 'kernel' : outer_kernel, 'C': outer_c, 'gamma': outer_gamma, 'epsilon': outer_epsilon })
    # save that dataframe to usr defined reports path path
    metrics_df.to_csv(reports_path + '/' + targets_str + "_outer_metrics.csv", index=False)


    end1 = time()
    print("This whole pipeline took:", end1 - start1, 'seconds')

    best_model = metrics_df.nsmallest(1, 'MSE array')

    print('The best test mse and model are:', best_model)

    # define test cross-validation on the original real test data.
    cv_test = LeaveOneOut()
    splits = cv_test.get_n_splits(X_normed)
    # enumerate splits
    test_mse = list()
    test_r2 = list()
    kernel = best_model['kernel'].values[0]
    c = best_model['C'].values[0]
    gamma = best_model['gamma'].values[0]
    epsilon=best_model['epsilon'].values[0]
    # fit final model for each fold
    final_SVR = SVR(
        kernel = kernel,
        C= c,
        gamma=gamma,
        epsilon=epsilon)
    print(final_SVR.get_params)
    # initialize fold counter
    fold = 0
    all_synth_x = []
    all_synth_y = []
    for train, test, in cv_test.split(X_normed):
        fold += 1
        print('Currently on fold:', fold, 'of the testing process')
        final_X_train, final_X_test = X_normed[train, :], X_normed[test, :]
        final_y_train, final_y_test = y_normed[train], y_normed[test]


        # append x and y np arrays together
        x_y_final = np.append(final_X_train, final_y_train, axis=1)
        # create dataframe of x and y array and create column names
        x_y_df_final = pd.DataFrame(x_y_final, columns = col_names)
        # create final synthetic data using LOOCV and fit
        test_synth_data = synth(x_y_df_final, eps, batch, verbosity, col_names, synth_sample_size)
        
        x_synth_final = test_synth_data.iloc[:,:-1].to_numpy()
        y_synth_final = test_synth_data.iloc[:,-1:].to_numpy()
        
        final_SVR.fit(x_synth_final, y_synth_final.ravel())

        final_predictions = final_SVR.predict(final_X_test)
        
        
        print('mean_squared_error:', mean_squared_error(final_y_test, final_predictions))
        

        print('Appending your final MSE scores to a list')
        test_mse.append(mean_squared_error(final_y_test, final_predictions))

    print('MSE: %.3f (%.3f)' % (np.median(test_mse), np.std(test_mse)))




##########################################################################
    # Permutation test against the median test mse and all data.

    median_mse = np.median(test_mse)
    x_y_normed = np.append(X_normed, y_normed, axis=1)
        # create dataframe of x and y array and create column names
    x_y_df_normed = pd.DataFrame(x_y_normed, columns = col_names)

    full_synth_data_normed = synth(x_y_df_normed, eps, batch, verbosity, col_names, synth_sample_size)
    all_synth_x = full_synth_data_normed.iloc[:,:-1].to_numpy()
    all_synth_y = full_synth_data_normed.iloc[:,-1:].to_numpy()
    p_value, obs_better_than_hyp = perm_svr(final_SVR, all_synth_x, X_normed, all_synth_y, y_normed, permutations, median_mse, reports_path, targets_str)
    print(p_value)




    
###############################################################################
    # Plot fitted models

    plot_models(X_train, X_test, y_train, y_test, final_SVR, reports_path, targets_str)


###############################################################################
    # create final files with metrics
    filestr =  "SVR_OUTPUT_"+features_str+"-vs-"+targets_str+"_final_model_metrics.txt"

    now1 = datetime.now()
    now1 = now1.strftime("%Y-%m-%d_%H-%M-%S")
    
    f2= open(reports_path + '/' + filestr,"w+")
    f2.writelines(now1)
    f2.writelines('\n Here are the model parameters for the best tested model. That is, the cross validated model which performed best.')
    f2.writelines('\n This model was used in a Leave one-out cross validation scheme to predict the outcome of interest')
    f2.writelines('\n Best kernel is: ' + str(kernel))
    f2.writelines('\n Best Cost value is: ' + str(c))
    f2.writelines('\n Best gamma value is: ' + str(gamma))
    f2.writelines('\n Best epsilon value is: '+ str(epsilon))
    f2.writelines('\n Median MSE: %.3f (%.3f) (STD)' % (np.median(test_mse), np.std(test_mse)))
    f2.writelines('\n Number of permutations better than hypothesized model: '+ str(obs_better_than_hyp))
    f2.writelines('\n Permutation test p-value: ' + str(p_value))
    f2.writelines('\n Bootstrapped confidence interval:' +  str(conf_interval))
    f2.close()


    metrics_df = pd.DataFrame({'MSE array':test_mse})


    metrics_df.to_csv(reports_path + '/' + "final_model_metrics.csv", index=False)






###############################################################################
