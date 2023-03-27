#!/usr/bin/env python


# Data transform


def transform_data(x, y, reports_path):
    """
    Transforms data using PowerTransformer for the features ('yeo-johnson') and QuantileTransformer for targets to create more normal distributions for all features and targets. Also plots all distributions after transformation.

    x ---(str) a path to the input feature set or csv
    y --- (str) a path to the input target set or csv
    reports_path --- (str) where the output report folder should be created
    
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import QuantileTransformer, PowerTransformer, StandardScaler
    x_cols = x.columns
    y_cols = y.columns
    x = x.to_numpy()
    y = y.to_numpy()
    # perform yeo-johnson trasnformation on x training  array
    scaler=StandardScaler()
    # scaler=PowerTransformer(method='yeo-johnson')
    x_normed= scaler.fit_transform(x)

    # perform Quantile transformation to normal distribution on y training array
    y_scaler = StandardScaler()
    # y_scaler = QuantileTransformer(n_quantiles=10, random_state=0, output_distribution='normal')
    y_normed= y_scaler.fit_transform(y.reshape(-1,1))


    print('Transforming your data and making distribution plots')

    # Making distribution plots for each column
    for feat, x_col in zip(range(x_normed.shape[1]), x_cols):
        sns.scatterplot(data= x_normed, x= x_normed[:,feat], y= y_normed.ravel())
        plt.xlabel(x_col)
        plt.ylabel(y_cols)
        plt.tight_layout()
        plt.savefig(reports_path + "/" + str(x_col) + '_' + str(y_cols) + '_scatterplot.svg')
        plt.clf()

    for feat, x_col in zip(range(x_normed.shape[1]), x_cols):
        sns.displot(data = x_normed, x = x_normed[:,feat], kde=True);
        plt.xlabel(x_col)
        plt.tight_layout()
        plt.savefig(reports_path + "/" + str(x_col) + '_distplot.svg')
        plt.clf()

    sns.displot(y_normed, kde=True)
    plt.xlabel(y_cols)
    plt.tight_layout()
    plt.savefig(reports_path + "/" + str(y_cols) + '_distplot.svg')
    col_names = x_cols.append(y_cols)
    
    plt.clf()

    return x_normed, y_normed, col_names
    

def synth(x_y_df, eps, batch, verbosity, col_names, synth_sample_size):
    """
    Generates synthetic data based on user specifications and reports some preliminary statistics about the quality of that synthetic data.


    x_y_df --- (pandas df) a concatenation of the X and y dataframes
    eps --- (int) of the number of epochs to fit the synthetic data GANs model
    batch --- (int) the number of rows to use when fitting the data in the GANs model
    verbosity --- (1 or 0) determines whether or not to print the loss values from fitting the GANS model
    col_names --- (list) list of column names from the dataframe
    synth_sample_size --- (int) number of synthetic data points to generate.
    
    """

    from sdv.tabular import CTGAN
    from sdv.evaluation import evaluate
    from time import time
    import warnings
    warnings.filterwarnings("ignore")
    from sklearn.preprocessing import StandardScaler
    import pandas as pd

    model = CTGAN(learn_rounding_scheme=False, epochs=eps, batch_size=batch, verbose=verbosity)

    # see [here](https://github.com/sdv-dev/SDV/discussions/980) for info on fitting the model
    start = time()
    
    print('Fitting the model to create synthetic data for train subjects, this could take a bit')
    print('\n')
    model.fit(x_y_df)
    
    print('Generating the final sample from the synthetic data')
    print('\n')
    synth_data = model.sample(num_rows=synth_sample_size)
    scaler= StandardScaler()
    scaled_synth_data = scaler.fit_transform(synth_data)
    end = time()
    
    print("This process took:", end - start, 'seconds')
    print('\n')
    
    # see (here)[https://sdv.dev/SDV/user_guides/evaluation/single_table_metrics.html] for evaluation info
    print("The Kolmogorov Smirnov normalized D statistic is:", evaluate(synth_data, x_y_df))

    print('\n')
    print('\n')
    print('You can see more about this process of evaluation of synthetic data here: https://sdv.dev/SDV/user_guides/evaluation/single_table_metrics.html')

    print('mean of synthetic data for each feature', scaled_synth_data.mean(axis=0))
    scaled_synth_data = pd.DataFrame(scaled_synth_data, columns=col_names)
    return scaled_synth_data


def synth_report(x_y_df, synth_data, reports_path):
    """
    Generate a report on the quality of the synthetic data used in each fold ---not currently used in synth_svr.py ---Still in BETA testing.

    x_y_df --- (pd df) dataframe to be used to compare to synthetic data fit
    synth_data --- (pd df) of synthetic data from the GANs model
    reports_path --- (str) path to output folder
    """

    from sdmetrics.reports.single_table import QualityReport
    from sdv.tabular import CTGAN
    from sdv.evaluation import evaluate
    import json
    import plotly


    # Generate json from original dataframe and save for later use.
    x_y_df.to_json(reports_path + '/' + 'xy_df.json')

    with open(reports_path + '/' + 'xy_df.json') as f:
        meta_data = json.load(f)

    
    # Initialize report
    report = QualityReport()
    report.generate(x_y_df, synth_data, meta_data)
    print('Overall quality similarity score of synthetic data:', report.get_score())
    print('A score closer to 1 is better')

    # creating a QC figure and saving as an html file
    print('Saving a QC image to:' , reports_path)
    fig = report.get_visualization(property_name='Column Shapes')
    # html file
    plotly.offline.plot(fig, filename=reports_path + '/' + 'synthetic_data_quality_fig.html')



# cross validated SVR and model selection for training data
def svr_selectCV(X_train, X_test, y_train, y_test, params):
    """
    Model selection SVR with CV using halving grid search CV. 

    X_train --- (numpy array) Array of synthetic data to train the SVR estimator for features
    X_test --- (numpy array) Array of real test data from the original dataset
    y_train --- (numpy array) Array of synthetic training data of targets
    y_test --- (numpy array) Array of real teset data for targets
    params --- (dictionary) Dictionary of parameters for estimator including those to be used in CV model selection.
    """
    # load dependencies
    from sklearn.svm import SVR
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV 
    from sklearn.experimental import enable_halving_search_cv # noqa
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import HalvingGridSearchCV
    from joblib import Memory
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    # create cached tmp directory which can be used to store the fit data
    location = 'cachedir'
    memory = Memory(location=location, verbose=0)
    
    # initialize ml pipeline
    pipe = Pipeline(steps=[('svr', SVR()),], memory=memory)
    # Select the optimal number of parcels with grid search
    grid_svr = HalvingGridSearchCV(pipe, params, scoring='r2', verbose=1, n_jobs=-1, cv=5)

    # fit halving grid svr
    grid_svr.fit(X_train, y_train.ravel())  # fit the best regressor


    print("Best cross validated score: %f using %s" % (grid_svr.best_score_, grid_svr.best_params_))
    # intialize best svr
    best_SVR = SVR(kernel = grid_svr.best_params_['svr__kernel'],
                   epsilon = grid_svr.best_params_['svr__epsilon'],
                   C = grid_svr.best_params_['svr__C'],
                   gamma= grid_svr.best_params_['svr__gamma'])
    # fit best svr
    best_SVR.fit(X_train, y_train.ravel())

    predictions = best_SVR.predict(X_test)
    
    print('r2:', r2_score(y_test, predictions))
    
    print('mean_squared_error:', mean_squared_error(y_test, predictions))
    
    
    hyp_score=mean_squared_error(y_test, predictions)
    r_squared= r2_score(y_test, predictions)

    return grid_svr, best_SVR, hyp_score, r_squared  



def learning_curves(reports_path, model, params, model_selection, fold_n, targets_str):
    """
    Creates a series of learning curves from the best overall model selection CV or best models in outer CV (toggle model selection on or off).

    reports_path --- (str) Path to output folder
    model --- (sklearn object) Estimator used in Hyperparameter tuning
    params --- (dict) Dictionary of parameters for estimator including those to be used in CV model selection.
    model_selection --- (int) Binary determines whether this is model selection or final model process. Should be either 0 or 1
    fold_n --- (int) Integer which dictates the number of folds to compute
    targets_str --- (str) String indicating the targets file name
    """
    # load dependencies
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    df = pd.DataFrame(model.cv_results_)
    results = ['mean_test_score',
        'mean_train_score',
        'std_test_score', 
        'std_train_score']

    # define poolev variance
    def pooled_var(stds):
        # https://en.wikipedia.org/wiki/Pooled_variance#Pooled_standard_deviation
        n = 5 # size of each group
        return np.sqrt(sum((n-1)*(stds**2))/ len(stds)*(n-1))

    fig, axes = plt.subplots(1, len(params), 
                            figsize = (5*len(params), 7))
                            # sharey='row')
    axes[0].set_ylabel("Score", fontsize=15)


    for idx, (param_name, param_range) in enumerate(params.items()):
        grouped_df = df.groupby(f'param_{param_name}')[results]\
            .agg({'mean_train_score': 'mean',
                'mean_test_score': 'mean',
                'std_train_score': pooled_var,
                'std_test_score': pooled_var})

        previous_group = df.groupby(f'param_{param_name}')[results]
        axes[idx].set_xlabel(param_name, fontsize=15)
        # axes[idx].set_ylim(0.0, 1.1)
        lw = 2
        axes[idx].plot(param_range, grouped_df['mean_train_score'], label="Training score",
                    color="darkorange", lw=lw)
        axes[idx].fill_between(param_range,grouped_df['mean_train_score'] - grouped_df['std_train_score'],
                        grouped_df['mean_train_score'] + grouped_df['std_train_score'], alpha=0.2,
                        color="darkorange", lw=lw)
        axes[idx].plot(param_range, grouped_df['mean_test_score'], label="Cross-validation score",
                    color="navy", lw=lw)
        axes[idx].fill_between(param_range, grouped_df['mean_test_score'] - grouped_df['std_test_score'],
                        grouped_df['mean_test_score'] + grouped_df['std_test_score'], alpha=0.2,
                        color="navy", lw=lw)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle('Validation curves', fontsize=15)
    fig.legend(handles, labels, loc="lower right", ncol=2, fontsize=10)
    plt.tight_layout()

    if model_selection == 1:
        plt.savefig(reports_path + '/' + str(fold_n) + '_' + targets_str + '_learning_curves_model_selection.svg')
    else:
        plt.savefig(reports_path + '/' + str(fold_n) + '_' + targets_str + '_learning_curves_final_model.svg')


# permutation test of SVR
def perm_svr(svr, synth_x, real_x, synth_y, real_y, permutations, median_mse, reports_path, targets_str):
    """
    Performs a permutation test using the original hypothesized MSE and compares that to a random null distribution.


    svr --- (sklearn object) Estimator from the overall best model
    x --- (numpy array) Array of original real features data
    y --- (numpy array) Array of original real target data
    permutations --- (int) Integer dictating the number of permutations to use (or shuffles of the target vector)
    median_mse --- (float) Float indicating the median test mse after LOOCV
    reports_path  --- (str) String indicating the output path

    
    """
    # import dependencies
    import matplotlib.pyplot as plt
    from sklearn.utils import shuffle
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import permutation_test_score
    import seaborn as sns
    import numpy as np
    import concurrent.futures

    
    print('Running permutation test')
    mse_arr = []
    
    score, perm_scores, pvalue = permutation_test_score(
        svr, real_x, real_y.ravel(), scoring="neg_mean_squared_error", cv=len(real_x), n_permutations=permutations,
         n_jobs=-1, verbose=1)

    perm_scores_pos = np.abs(perm_scores)

    # plot observed mse and null permuted mse
    plt.clf()
    sns.histplot(perm_scores_pos, bins=1000, color = 'slategrey')
    plt.axvline(median_mse, color='green', linestyle='dashed', linewidth=2)
    # plt.title(title, fontsize = 17)
    plt.xlabel('Mean Squared Error (MSE)', fontsize = 15)
    plt.ylabel('Frequency', fontsize = 15)
    plt.savefig(reports_path +'/' + targets_str + '_permutation_test.svg')

    # compute permutation test which is the proportion of models which performed better than the observed MSE
    obs_better_than_hyp = 0
    for ii in perm_scores_pos:
        if ii <= median_mse:
            obs_better_than_hyp = obs_better_than_hyp + 1 
    p_value = round(obs_better_than_hyp / permutations, 5)

    print(p_value)

        
    return p_value, obs_better_than_hyp




def plot_models(X_train, X_test, y_train, y_test, svr, reports_path, targets_str):
    """
    This function plots the training and test MSE on the same lineplots
    
    X_train --- (pandas dataframe) A dataframe with the training data
    X_test --- (pandas dataframe)
    """
    # import dependencies
    import matplotlib.pyplot as plt
    import seaborn as sns

    for feat in range(X_train.shape[1]):

            plt.clf()
            bestHypermod_predict = svr.fit(X_train, y_train.ravel()).predict(X_test)

            sns.scatterplot(X_train[:,feat ], y_train.ravel(), color='darkorange', label='SVR fit')
            plt.plot(sorted(X_test[:,feat ]), sorted(bestHypermod_predict), color='navy', label='Hypothesized Model')

            fig = plt.gcf()
            plt.draw()
            plt.tight_layout()
            fig.savefig(reports_path + '/' + str(feat)+ '_' + targets_str + '_model_plot.png')



# generate report

def summary_file(reports_path, features_str, targets_str, final_model, permutations, scores, obs, p_val, conf_int, summary_metrics):
    """
    Generates a summary text file ---not currently used in synth_svr.py


    reports_path --- (str) String indicating location of output folder
    features_str --- (str) String indicating the features basename
    targets_str --- (str) String indicating the targets basename
    final_model --- (sklearn object) Estimator of the best overall model across all CV schemes and model selection
    permutations --- (int) Integer dictating the number of permutations run for the permutation test.
    obs --- (int) Integer dictating the number of models which fit the data better than the hypothesized model
    p_val --- (float) Float indicating the p-value from the permutation test
    conf_int --- (list) list indicating the 95% confidence interval
    summary_metrics --- (list) List of possible metrics to create files for.
    """
    
    # load dependencies
    from datetime import datetime
    import numpy as np
    now = datetime.now()
    now = now.strftime("%Y-%m-%d_%H-%M-%S")
    # set output file path
    filestr =  "SVR_OUTPUT_"+features_str+"-vs-"+targets_str+".txt"
    grid_bestscore = final_model.best_score_
    grid_best_params= final_model.best_params_
    # write report file
    f= open(reports_path + '/' + filestr,"w+")
    f.writelines(now)
    f.writelines('\nBest model MSE:' + str(grid_bestscore))
    f.writelines('\nOptimal Hyperparameters across all tested models')
    f.writelines('\nBest params: ' + str(grid_best_params))
    
    f.writelines('\n')
    f.writelines('\n')
    f.writelines('\nFeature Set Filename: ' + str(features_str))
    f.writelines('\nTarget Set Filename: ' + str(targets_str))
    f.writelines('\nNumber of Permutations: ' + str(permutations))
    for score, metric in zip(scores, summary_metrics):
        f.writelines('\n' +  metric  + '=' + str(score))
    f.writelines('\nMean null (mean squared) error: ' + str(np.mean(abs(np.array(obs)))))
    f.writelines('\nNumber of models better than hypothesized model: '+ str(obs))
    f.writelines('\nHypothesized model p-value compared to the null distribution: ' + str(p_val))
    f.writelines('\nBootstrap confidence interval' + str(conf_int))
    f.close()


if __name__ == '__main__':
    print()
