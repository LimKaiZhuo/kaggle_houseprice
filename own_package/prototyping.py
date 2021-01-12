import pandas as pd
import numpy as np
import scipy.stats
import category_encoders as ce
from xgboost import XGBRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
import pickle

from own_package.pipeline_helpers import preprocess_pipeline_1, preprocess_pipeline_2, preprocess_pipeline_3
from own_package.others import create_results_directory


def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y) - np.log1p(y0), 2)))


def lvl1_randomsearch(rawdf, results_dir, preprocess_pipeline_choice):
    x_train = rawdf.iloc[:, :-1]
    y_train = rawdf.iloc[:, -1]
    model_store = ['rf', 'et', 'xgb']
    model_object = {
        'xgb': XGBRegressor(),
        'rf': RandomForestRegressor(),
        'et': ExtraTreesRegressor()
    }
    model_param = {
        'xgb': {'xgb__n_estimators': scipy.stats.randint(150, 1000),
                'xgb__learning_rate': scipy.stats.uniform(0.01, 0.59),
                'xgb__subsample': scipy.stats.uniform(0.3, 0.6),
                'xgb__max_depth': scipy.stats.randint(1, 16),
                'xgb__colsample_bytree': scipy.stats.uniform(0.5, 0.4),
                'xgb__min_child_weight': [1, 2, 3, 4],
                'xgb__gamma': scipy.stats.expon(scale=0.05),
                },
        'rf': {"rf__max_depth": [None],
               "rf__max_features": scipy.stats.randint(1, 11),
               "rf__min_samples_split": scipy.stats.randint(2, 11),
               "rf__min_samples_leaf": scipy.stats.randint(1, 11),
               "rf__bootstrap": [False],
               "rf__n_estimators": scipy.stats.randint(10, 300),},
        'et': {"et__max_depth": [None],
               "et__max_features": scipy.stats.randint(1, 11),
               "et__min_samples_split": scipy.stats.randint(2, 11),
               "et__min_samples_leaf": scipy.stats.randint(1, 11),
               "et__bootstrap": [False],
               "et__n_estimators": scipy.stats.randint(10, 300),}
    }
    results_store = {}

    if preprocess_pipeline_choice == 1:
        preprocess_pipeline = preprocess_pipeline_1()
    elif preprocess_pipeline_choice == 2:
        preprocess_pipeline = preprocess_pipeline_2()
    elif preprocess_pipeline_choice == 3:
        preprocess_pipeline = preprocess_pipeline_3(rawdf)

    for model_name in model_store:
        model = Pipeline([
            ('preprocess', preprocess_pipeline),
            (model_name, model_object[model_name])
        ])

        clf = RandomizedSearchCV(model,
                                 param_distributions=model_param[model_name],
                                 cv=5,
                                 n_iter=100,
                                 scoring=make_scorer(rmsle, greater_is_better=False),
                                 verbose=1,
                                 n_jobs=-1)

        clf.fit(x_train, y_train)
        results_store[model_name] = clf.cv_results_

    results_dir = create_results_directory(results_dir)
    with open(f'{results_dir}/results_store.pkl', 'wb') as f:
        pickle.dump(results_store, f)


def analyze_xgb_randomsearch(results_dir):
    with open(f'{results_dir}/results_store.pkl', 'rb') as f:
        results_store = pickle.load(f)
    results_store = {k: pd.DataFrame(v).sort_values('mean_test_score', ascending=False) for k,v in results_store.items()}
    print('hi')
