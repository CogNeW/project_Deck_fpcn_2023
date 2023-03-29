def feat_imp(x, y, cv_splits, out_path):
    '''
    This function allows you to compute cross-validated random forest 
    model and further computes feature importance using permutation 
    importance. 
    x = features
    y = targets
    cv_splits = how many splits for CV process
    out_path

    '''
    print('Importing modules')

    import pandas as pd 
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.inspection import permutation_importance
    import matplotlib.pyplot as plt
    import csv

    feats = pd.read_csv(x)
    targs = pd.read_csv(y)

    print(len(feats))
    print(len(targs))

    print('initiating your model')
    
    model = RandomForestRegressor(n_jobs = -1, random_state=17)

    print('creating your Random Search CV parameters dicts')

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True]

    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}

    rf_random = RandomizedSearchCV(estimator = model, 
                               param_distributions = random_grid, 
                               n_iter = 100, cv = cv_splits, verbose=2, 
                               random_state=17, n_jobs = -1, return_train_score = True)

    # Fit the random search model
    rf_random.fit(feats, np.ravel(targs))

    print("The best params from the random search CV are: ", rf_random.best_estimator_)
    model = rf_random.best_estimator_

    print('Performing permutation importance from the best randome forest model')

    result = permutation_importance(
    model, feats, np.ravel(targs), n_repeats=1000, random_state=42, n_jobs=-1
    )
    sorted_idx = result.importances_mean.argsort()

    fig, ax = plt.subplots()
    ax.boxplot(
        result.importances[sorted_idx].T, vert=False, labels=feats.columns[sorted_idx]
    )
    ax.set_title("Permutation Importances (test set)")
    fig.tight_layout()
    plt.show()

    # Print mapped feature columns to feature importance mean and std
    print('Mean Importances from highest to lowest : ', list(zip(result.importances_mean, feats.columns)))
    print('Mean Importances from highest to lowest : ', list(zip(result.importances_std, feats.columns)))



    csvfile = out_path

    with open(csvfile, "wb") as output:
        writer = csv.DictWriter(output, fieldnames=['date', 'v'])
        writer.writerows(list(zip(result.importances_mean, feats.columns)))



    return rf_random.cv_results_, result.importances_mean, result.importances_std
