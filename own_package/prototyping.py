import pandas as pd
import numpy as np
import scipy.stats
from xgboost import XGBRegressor
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, StackingRegressor
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import RandomizedSearchCV, cross_validate

from sklearn.metrics import make_scorer
import pickle

from own_package.pipeline_helpers import preprocess_pipeline_1, preprocess_pipeline_2, \
    preprocess_pipeline_3, preprocess_pipeline_4, preprocess_pipeline_5, final_est_pipeline, DebuggerTransformer
from own_package.others import create_results_directory


def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y) - np.log1p(y0), 2)))


def pp_selector(preprocess_pipeline_choice, rawdf=None):
    if preprocess_pipeline_choice == 1:
        preprocess_pipeline = preprocess_pipeline_1(rawdf)
    elif preprocess_pipeline_choice == 2:
        preprocess_pipeline = preprocess_pipeline_2(rawdf)
    elif preprocess_pipeline_choice == 3:
        preprocess_pipeline = preprocess_pipeline_3(rawdf)
    elif preprocess_pipeline_choice == 4:
        preprocess_pipeline = preprocess_pipeline_4(rawdf)
    elif preprocess_pipeline_choice == 5:
        preprocess_pipeline = preprocess_pipeline_5(rawdf)
    return preprocess_pipeline


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
               "rf__n_estimators": scipy.stats.randint(10, 300), },
        'et': {"et__max_depth": [None],
               "et__max_features": scipy.stats.randint(1, 11),
               "et__min_samples_split": scipy.stats.randint(2, 11),
               "et__min_samples_leaf": scipy.stats.randint(1, 11),
               "et__bootstrap": [False],
               "et__n_estimators": scipy.stats.randint(10, 300), }
    }
    results_store = {}

    preprocess_pipeline = pp_selector(preprocess_pipeline_choice, rawdf)

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


def lvl2_ridgecv(rawdf, results_dir, pp_choice, param_dir, passthrough, final_pp_choice=None, ):
    x_train = rawdf.iloc[:, :-1]
    y_train = rawdf.iloc[:, -1]
    model_store = ['rf', 'et', 'xgb']
    model_object = {
        'xgb': XGBRegressor(),
        'rf': RandomForestRegressor(),
        'et': ExtraTreesRegressor()
    }

    with open(param_dir, 'rb') as f:
        model_results = pickle.load(f)
    model_results = {k: pd.DataFrame(v).sort_values('mean_test_score', ascending=False) for k, v in
                     model_results.items()}
    model_object = {k: model_object[k].set_params(**{kk.split('__')[1]: vv for kk, vv in v.loc[0, 'params'].items()})
                    for k, v in model_results.items()}

    lvl1_pipeline = [
        (model_name,
         Pipeline([
             ('preprocess', pp_selector(pipeline_idx)),
             ('debugger', DebuggerTransformer(info='lvl1')),
             (model_name, model_object[model_name])
         ])
         )
        for model_name, pipeline_idx in zip(model_store, pp_choice)]

    if passthrough:
        final_est = Pipeline([
            ('final_preprocess', final_est_pipeline(feature_names=x_train.columns.tolist(),
                                                    preprocess_pipeline=pp_selector(final_pp_choice),
                                                    no_of_lvl1=len(lvl1_pipeline))),
            ('debugger', DebuggerTransformer(info='final')),
            ('final_est', RidgeCV())
        ])
    else:
        final_est = RidgeCV()

    est = StackingRegressor(estimators=lvl1_pipeline, final_estimator=final_est, passthrough=passthrough)
    score = cross_validate(est, x_train, y_train, cv=5, return_train_score=True,
                           scoring=make_scorer(rmsle, greater_is_better=False))

    results_dir = create_results_directory(results_dir)
    with open(f'{results_dir}/results_store.pkl', 'wb') as f:
        pickle.dump(score, f)


def lvl2_xgb_randomsearch(rawdf, results_dir, pp_choice, param_dir, passthrough, final_pp_choice=None):
    x_train = rawdf.iloc[:, :-1]
    y_train = rawdf.iloc[:, -1]
    model_store = ['rf', 'et', 'xgb']
    model_object = {
        'xgb': XGBRegressor(),
        'rf': RandomForestRegressor(),
        'et': ExtraTreesRegressor()
    }

    with open(param_dir, 'rb') as f:
        model_results = pickle.load(f)
    model_results = {k: pd.DataFrame(v).sort_values('mean_test_score', ascending=False) for k, v in
                     model_results.items()}
    model_object = {k: model_object[k].set_params(**{kk.split('__')[1]: vv for kk, vv in v.loc[0, 'params'].items()})
                    for k, v in model_results.items()}

    preprocess_pipeline = pp_selector(pp_choice)

    lvl1_pipeline = [
        (model_name,
         Pipeline([
             ('preprocess', preprocess_pipeline),
             (model_name, model_object[model_name])
         ])
         )
        for model_name in model_store]
    final_estimator_params = {'final_estimator__final_est__n_estimators': scipy.stats.randint(150, 1000),
                              'final_estimator__final_est__learning_rate': scipy.stats.uniform(0.01, 0.59),
                              'final_estimator__final_est__subsample': scipy.stats.uniform(0.3, 0.6),
                              'final_estimator__final_est__max_depth': scipy.stats.randint(1, 16),
                              'final_estimator__final_est__colsample_bytree': scipy.stats.uniform(0.5, 0.4),
                              'final_estimator__final_est__min_child_weight': [1, 2, 3, 4],
                              'final_estimator__final_est__gamma': scipy.stats.expon(scale=0.05),
                              }
    if passthrough:
        final_est = Pipeline([
            ('final_preprocess', final_est_pipeline(feature_names=x_train.columns.tolist(),
                                                    preprocess_pipeline=pp_selector(final_pp_choice),
                                                    no_of_lvl1=len(lvl1_pipeline))),
            ('debugger', DebuggerTransformer(info='final')),
            ('final_est', XGBRegressor())
        ])
    else:
        final_est = XGBRegressor()

    est = StackingRegressor(estimators=lvl1_pipeline, final_estimator=final_est, passthrough=passthrough)
    est = RandomizedSearchCV(est,
                             param_distributions=final_estimator_params,
                             cv=5,
                             n_iter=100,
                             scoring=make_scorer(rmsle, greater_is_better=False),
                             verbose=1,
                             n_jobs=-1)

    est.fit(x_train, y_train)
    score = {'lvl2_xgb': est.cv_results_}
    results_dir = create_results_directory(results_dir)
    with open(f'{results_dir}/results_store.pkl', 'wb') as f:
        pickle.dump(score, f)


def lvl1_generate_prediction(rawdf, x_test, result_dir, type_, preprocess_pipeline_choice):
    x_train = rawdf.iloc[:, :-1]
    y_train = rawdf.iloc[:, -1]
    if type_ == 'lvl1_randomsearch':
        model_names = ['rf', 'et', 'xgb']
        model_object = {
            'xgb': XGBRegressor(),
            'rf': RandomForestRegressor(),
            'et': ExtraTreesRegressor()
        }

        with open(f'{result_dir}/results_store.pkl', 'rb') as f:
            model_results = pickle.load(f)
        model_results = {k: pd.DataFrame(v).sort_values('mean_test_score', ascending=False) for k, v in
                         model_results.items()}

        lvl1_pipeline = [
            Pipeline([
                ('preprocess', pp_selector(preprocess_pipeline_choice)),
                (model_name, model_object[model_name])
            ]).set_params(**model_results[model_name].loc[0, 'params'])
            for model_name in model_names]

        prediction_store = [model.fit(x_train, y_train).predict(x_test) for model in lvl1_pipeline]
        sub = pd.DataFrame()
        sub['Id'] = x_test['Id']
        for prediction, model_name in zip(prediction_store, model_names):
            temp = sub.copy()
            temp['SalePrice'] = prediction
            temp.to_csv(f'{result_dir}/{model_name}_predictions.csv', index=False)


def lvl2_generate_prediction(rawdf, x_test, results_dir, lvl1_results_dir, type_, preprocess_pipeline_choice):
    x_train = rawdf.iloc[:, :-1]
    y_train = rawdf.iloc[:, -1]
    model_names = ['rf', 'et', 'xgb']
    model_object = {
        'xgb': XGBRegressor(),
        'rf': RandomForestRegressor(),
        'et': ExtraTreesRegressor()
    }

    with open(f'{lvl1_results_dir}/results_store.pkl', 'rb') as f:
        model_results = pickle.load(f)
    model_results = {k: pd.DataFrame(v).sort_values('mean_test_score', ascending=False) for k, v in
                     model_results.items()}

    lvl1_pipeline = [
        (model_name, Pipeline([
            ('preprocess', pp_selector(preprocess_pipeline_choice)),
            (model_name, model_object[model_name])
        ]).set_params(**model_results[model_name].loc[0, 'params']))
        for model_name in model_names]

    if type_ == 'lvl2_ridgecv':
        est = StackingRegressor(estimators=lvl1_pipeline, final_estimator=RidgeCV(), passthrough=False)
    elif type_ == 'lvl2_xgb':
        est = StackingRegressor(estimators=lvl1_pipeline, final_estimator=XGBRegressor(), passthrough=False)

        with open(f'{results_dir}/results_store.pkl', 'rb') as f:
            model_results = pickle.load(f)
        model_results = {k: pd.DataFrame(v).sort_values('mean_test_score', ascending=False) for k, v in
                         model_results.items()}
        est.set_params(**model_results['lvl2_xgb'].loc[0, 'params'])

    prediction = est.fit(x_train, y_train).predict(x_test)
    sub = pd.DataFrame()
    sub['Id'] = x_test['Id']
    sub['SalePrice'] = prediction
    sub.to_csv(f'{results_dir}/{type_}_pp{preprocess_pipeline_choice}_predictions.csv', index=False)
