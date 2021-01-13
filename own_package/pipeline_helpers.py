import pandas as pd
import numpy as np
import scipy.stats
import category_encoders as ce
from xgboost import XGBRegressor
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.metrics import make_scorer
import pickle

from own_package.others import create_results_directory


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.feature_names]


class DebuggerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, info=None):
        self.info = info

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


class FinalFeatureDataframe(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        self.feature_count_ = len(self.feature_names)
        return self

    def transform(self, X):
        return pd.DataFrame(X,
                            columns=[f'{x + 1}_lvl1_' for x in
                                     range(X.shape[1] - self.feature_count_)] + self.feature_names)


class PdFunctionTransformer(BaseEstimator, TransformerMixin):
    """ A DataFrame transformer providing imputation or function application

    Parameters
    ----------
    impute : Boolean, default False

    func : function that acts on an array of the form [n_elements, 1]
        if impute is True, functions must return a float number, otherwise
        an array of the form [n_elements, 1]

    """

    def __init__(self, func, impute=False):
        self.func = func
        self.impute = impute
        self.series = pd.Series()

    def transform(self, X, **transformparams):
        """ Transforms a DataFrame

        Parameters
        ----------
        X : DataFrame

        Returns
        ----------
        trans : pandas DataFrame
            Transformation of X
        """

        if self.impute:
            trans = pd.DataFrame(X).fillna(self.series).copy()
        else:
            trans = pd.DataFrame(X).apply(self.func).copy()
        return trans

    def fit(self, X, y=None, **fitparams):
        """ Fixes the values to impute or does nothing

        Parameters
        ----------
        X : pandas DataFrame
        y : not used, API requirement

        Returns
        ----------
        self
        """
        if self.impute:
            self.series = pd.DataFrame(X).apply(self.func).squeeze()
        return self


class PdWithinGroupImputerTransformer(BaseEstimator, TransformerMixin):
    """ A DataFrame transformer providing imputation or function application

    Parameters
    ----------
    impute : Boolean, default False

    func : function that acts on an array of the form [n_elements, 1]
        if impute is True, functions must return a float number, otherwise
        an array of the form [n_elements, 1]

    """

    def __init__(self, func, groupby, x_names):
        self.func = func
        self.groupby = groupby  # String
        self.x_names = x_names  # List of string

    def transform(self, X, **transformparams):
        X = X.copy()
        # Replace new unseen categories in groupby with Others_
        X.loc[~X[self.groupby].isin(self.groupby_categories), self.groupby] = 'Others_'
        trans = []
        for x in self.x_names:
            temp = X.copy()
            temp.loc[X[x].isnull(), x] = X.loc[X[x].isnull(), self.groupby].map(lambda n: self.df_map.loc[n, x])
            trans.append(temp[[x]])
        return pd.concat(trans, axis=1)

    def fit(self, X, y=None, **fitparams):
        # lambda x: x.value_counts().index[0]
        self.df_map = X[self.x_names + [self.groupby]].groupby(self.groupby).agg(
            {x: self.func for x in self.x_names})
        self.df_map = pd.concat([self.df_map, X[self.x_names].apply(self.func).to_frame(name='Others_').T], axis=0)
        self.groupby_categories = X[self.groupby].unique()
        return self


class PdTypeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, type_):
        self.type_ = type_

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.astype(self.type_)


class PdSumTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, series_name, weights=None):
        """
        series_name: name of the new summed column
        weights: weights when summing the columns together
        """
        self.series_name = series_name
        self.weights = weights

    def fit(self, X, y=None):
        if self.weights:
            assert X.shape[1] == len(self.weights)  # no. of columns = no. of weights
        return self

    def transform(self, X):
        if self.weights:
            X = X * self.weights
        return X.sum(axis=1).to_frame(self.series_name)


class Binarizer(BaseEstimator, TransformerMixin):
    def __init__(self, condition, name):
        self.condition = condition
        self.name = name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(lambda x: int(self.condition(x))).to_frame(self.name)


class GroupingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, *, min_count=0, min_freq=0.0, top_n=0):
        self.min_count = min_count
        self.min_freq = min_freq
        self.top_n = top_n

    def fit(self, X, y=None):
        self.group_name = 'Others_'
        X = X.fillna('None')  # In case there is still any nan left
        n_samples, n_features = X.shape
        counts = []
        groups = []
        other_in_keys = []
        for i in range(n_features):
            cnts = Counter(X.iloc[:, i])
            counts.append(cnts)
            if self.top_n == 0:
                self.top_n = len(cnts)
            labels_to_group = (label for rank, (label, count) in enumerate(cnts.most_common())
                               if ((count < self.min_count)
                                   or (count / n_samples < self.min_freq)
                                   or (rank >= self.top_n)
                                   )
                               )
            groups.append(np.array(sorted(set(labels_to_group))))
            other_in_keys.append(self.group_name in cnts.keys())
        self.counts_ = counts
        self.groups_ = groups
        self.other_in_keys_ = other_in_keys
        return self

    def transform(self, X):
        X_t = X.copy()
        X_t = X_t.fillna('None')
        _, n_features = X.shape
        for i in range(n_features):
            mask = np.isin(X_t.iloc[:, i], self.groups_[i])
            X_t.iloc[mask, i] = self.group_name
        return X_t


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]

        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self

    # Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)


def final_est_pipeline(feature_names, preprocess_pipeline, no_of_lvl1):
    lvl1_pred = Pipeline([
        ('create_final_df', FinalFeatureDataframe(feature_names)),
        ('lvl_1_pred', FeatureSelector([f'{x+1}_lvl1_' for x in range(no_of_lvl1)])),
    ])
    preprocess = Pipeline([
        ('create_final_df', FinalFeatureDataframe(feature_names)),
        ('final_preprocess', preprocess_pipeline),
    ])
    return FeatureUnion([
        ('lvl_1_pred', lvl1_pred),
        ('trans_features', preprocess)
    ])


def preprocess_pipeline_1(rawdf):
    # Ratio or Interval pipeline: Numeric pipeline
    numeric_columns = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtUnfSF', '1stFlrSF', '2ndFlrSF',
                       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
                       'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageArea', 'WoodDeckSF',
                       'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
                       'OverallQual', 'OverallCond']
    p_numeric = Pipeline([
        ('sel_numeric', FeatureSelector(numeric_columns)),
        ('fillna_with_0', SimpleImputer())
    ])

    # Onehot with binning infrequents
    cat_onehot_columns = ['MSZoning', 'Street', 'BldgType', 'MasVnrType', 'GarageType', 'Fence',
                          'SaleType', 'SaleCondition',

                          'LandContour', 'LotConfig', 'Neighborhood', 'Condition1', 'HouseStyle', 'RoofStyle',
                          'Exterior1st', 'Foundation', 'Electrical']

    p_cat_onehot = Pipeline([
        ('sel_cat', FeatureSelector(cat_onehot_columns)),
        ('infrequent_grouping', GroupingTransformer(min_freq=0.05)),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    # This is the only categorical column that has all numbers so must convert to string first.
    p_mssubclass_onehot = Pipeline([
        ('sel_mssubclass', FeatureSelector(['MSSubClass'])),
        ('float2int', PdTypeTransformer(str)),
        ('infrequent_grouping', GroupingTransformer(min_freq=0.05)),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    # Ordinal transformations
    map_alley = {'Grvl': 0, 'Pave': 0}  # Missing values will be filled with -1
    map_LotShape = {'IR3': 0, 'IR2': 1, 'IR1': 2, 'Reg': 3}
    map_LandSlope = {'Gtl': 0, 'Mod': 1, 'Sev': 2}
    map_po_ex = {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}
    map_no_gd = {'No': 0, 'Mn': 1, 'Av': 2, 'Gd': 3}
    map_unf_glq = {'Unf': 0, 'LwQ': 1, 'Rec': 2, 'BLQ': 3, 'ALQ': 4, 'GLQ': 5}
    map_n_y = {'N': 0, 'Y': 1}
    map_Functional = {'Typ': 0, 'Min1': 1, 'Min2': 2, 'Mod': 3, 'Maj1': 4, 'Maj2': 5, 'Sev': 6, 'Sal': 7}
    map_GarageFinish = {'Unf': 0, 'RFn': 1, 'Fin': 2}
    map_PavedDrive = {'N': 0, 'P': 1, 'Y': 2}
    map_MiscFeature = {'Shed': 0}

    ordinal_columns = ['Alley', 'LotShape', 'LandSlope', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
                       'BsmtExposure', 'BsmtFinType1', 'HeatingQC', 'CentralAir', 'KitchenQual',
                       'Functional', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond',
                       'PavedDrive', 'MiscFeature']
    ordinal_mappings = [map_alley, map_LotShape, map_LandSlope, map_po_ex, map_po_ex, map_po_ex, map_po_ex,
                        map_no_gd, map_unf_glq, map_po_ex, map_n_y, map_po_ex,
                        map_Functional, map_po_ex, map_GarageFinish, map_po_ex, map_po_ex,
                        map_PavedDrive, map_MiscFeature]

    ordinal_mappings = [{'col': col, 'mapping': mapping} for col, mapping in zip(ordinal_columns, ordinal_mappings)]

    p_ordinal = Pipeline([
        ('sel_ordinal', FeatureSelector(ordinal_columns)),
        ('ordinal', ce.OrdinalEncoder(mapping=ordinal_mappings))
    ])

    # Bin pool to yes or no only
    p_PoolArea_binary = Pipeline([
        ('sel_PoolArea', FeatureSelector('PoolArea')),
        ('binary', Binarizer(condition=lambda x: x > 0, name='PoolArea'))
    ])

    # Time information
    time_columns = ['YearBuilt', 'YearRemodAdd', 'MoSold', 'YrSold']
    p_time = Pipeline([
        ('sel_PoolArea', FeatureSelector(time_columns))
    ])

    p_preprocess = FeatureUnion([
        ('p_numeric', p_numeric),
        ('p_cat_onehot', p_cat_onehot),
        ('p_mssubclass_onehot', p_mssubclass_onehot),
        ('p_ordinal', p_ordinal),
        ('p_PoolArea_binary', p_PoolArea_binary),
        ('p_time', p_time),
    ])

    return p_preprocess


def preprocess_pipeline_2(rawdf):
    '''
    Neighborhood binning to ordinal
    :return:
    '''
    # Ratio or Interval pipeline: Numeric pipeline
    numeric_columns = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtUnfSF', '1stFlrSF', '2ndFlrSF',
                       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
                       'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageArea', 'WoodDeckSF',
                       'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
                       'OverallQual', 'OverallCond']
    p_numeric = Pipeline([
        ('sel_numeric', FeatureSelector(numeric_columns)),
        ('fillna_with_0', SimpleImputer())
    ])

    # Onehot with binning infrequents
    cat_onehot_columns = ['MSZoning', 'Street', 'BldgType', 'MasVnrType', 'GarageType', 'Fence',
                          'SaleType', 'SaleCondition',

                          'LandContour', 'LotConfig', 'Condition1', 'HouseStyle', 'RoofStyle',
                          'Exterior1st', 'Foundation', 'Electrical']

    p_cat_onehot = Pipeline([
        ('sel_cat', FeatureSelector(cat_onehot_columns)),
        ('infrequent_grouping', GroupingTransformer(min_freq=0.05)),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    # This is the only categorical column that has all numbers so must convert to string first.
    p_mssubclass_onehot = Pipeline([
        ('sel_mssubclass', FeatureSelector(['MSSubClass'])),
        ('float2int', PdTypeTransformer(str)),
        ('infrequent_grouping', GroupingTransformer(min_freq=0.05)),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    # Ordinal transformations
    map_alley = {'Grvl': 0, 'Pave': 0}  # Missing values will be filled with -1
    map_LotShape = {'IR3': 0, 'IR2': 1, 'IR1': 2, 'Reg': 3}
    map_LandSlope = {'Gtl': 0, 'Mod': 1, 'Sev': 2}
    map_po_ex = {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}
    map_no_gd = {'No': 0, 'Mn': 1, 'Av': 2, 'Gd': 3}
    map_unf_glq = {'Unf': 0, 'LwQ': 1, 'Rec': 2, 'BLQ': 3, 'ALQ': 4, 'GLQ': 5}
    map_n_y = {'N': 0, 'Y': 1}
    map_Functional = {'Typ': 0, 'Min1': 1, 'Min2': 2, 'Mod': 3, 'Maj1': 4, 'Maj2': 5, 'Sev': 6, 'Sal': 7}
    map_GarageFinish = {'Unf': 0, 'RFn': 1, 'Fin': 2}
    map_PavedDrive = {'N': 0, 'P': 1, 'Y': 2}
    map_MiscFeature = {'Shed': 0}
    name_store = [['MeadowV', 'IDOTRR', 'BrDale', 'BrkSide', 'Edwards'],
                  ['OldTown', 'Sawyer', 'Blueste', 'SWISU', 'NPkVill'],
                  ['NAmes', 'Mitchel', 'SawyerW', 'NWAmes', 'Gilbert'],
                  ['Blmngtn', 'CollgCr', 'Crawfor', 'ClearCr', 'Somerst'],
                  ['Veenker', 'Timber', 'StoneBr', 'NridgHt', 'NoRidge']]
    map_Neighborhood = {name: value for value, name_group in enumerate(name_store) for name in name_group}

    ordinal_columns = ['Alley', 'LotShape', 'LandSlope', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
                       'BsmtExposure', 'BsmtFinType1', 'HeatingQC', 'CentralAir', 'KitchenQual',
                       'Functional', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond',
                       'PavedDrive', 'MiscFeature', 'Neighborhood']
    ordinal_mappings = [map_alley, map_LotShape, map_LandSlope, map_po_ex, map_po_ex, map_po_ex, map_po_ex,
                        map_no_gd, map_unf_glq, map_po_ex, map_n_y, map_po_ex,
                        map_Functional, map_po_ex, map_GarageFinish, map_po_ex, map_po_ex,
                        map_PavedDrive, map_MiscFeature, map_Neighborhood]

    ordinal_mappings = [{'col': col, 'mapping': mapping} for col, mapping in zip(ordinal_columns, ordinal_mappings)]

    p_ordinal = Pipeline([
        ('sel_ordinal', FeatureSelector(ordinal_columns)),
        ('ordinal', ce.OrdinalEncoder(mapping=ordinal_mappings))
    ])

    # Bin pool to yes or no only
    p_PoolArea_binary = Pipeline([
        ('sel_PoolArea', FeatureSelector('PoolArea')),
        ('binary', Binarizer(condition=lambda x: x > 0, name='PoolArea'))
    ])

    # Time information
    time_columns = ['YearBuilt', 'YearRemodAdd', 'MoSold', 'YrSold']
    p_time = Pipeline([
        ('sel_PoolArea', FeatureSelector(time_columns))
    ])

    p_preprocess = FeatureUnion([
        ('p_numeric', p_numeric),
        ('p_cat_onehot', p_cat_onehot),
        ('p_mssubclass_onehot', p_mssubclass_onehot),
        ('p_ordinal', p_ordinal),
        ('p_PoolArea_binary', p_PoolArea_binary),
        ('p_time', p_time),
    ])

    return p_preprocess


def preprocess_pipeline_3(rawdf=None):
    '''
    2) Neighborhood binning to ordinal
    3) Combine bathrooms.
    :return:
    '''
    # Ratio or Interval pipeline: Numeric pipeline
    numeric_columns = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtUnfSF', '1stFlrSF', '2ndFlrSF',
                       'LowQualFinSF', 'GrLivArea',
                       'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageArea', 'WoodDeckSF',
                       'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
                       'OverallQual', 'OverallCond']
    p_numeric = Pipeline([
        ('sel_numeric', FeatureSelector(numeric_columns)),
        ('fillna_with_0', SimpleImputer())
    ])

    # Weighted sum of bathrooms
    bathroom_columns = ['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', ]
    p_Bathrooms = Pipeline([
        ('sel_Bathrooms', FeatureSelector(bathroom_columns)),
        ('fillna_with_0', PdFunctionTransformer(func=pd.Series.mean, impute=True)),
        ('weighted_sum', PdSumTransformer(series_name='TotalBathrooms', weights=[1, 0.5, 1, 0.5]))
    ])

    # Onehot with binning infrequents
    cat_onehot_columns = ['MSZoning', 'Street', 'BldgType', 'MasVnrType', 'GarageType', 'Fence',
                          'SaleType', 'SaleCondition',

                          'LandContour', 'LotConfig', 'Condition1', 'HouseStyle', 'RoofStyle',
                          'Exterior1st', 'Foundation', 'Electrical']

    p_cat_onehot = Pipeline([
        ('sel_cat', FeatureSelector(cat_onehot_columns)),
        ('imputation', PdFunctionTransformer(func=pd.Series.mode, impute=True)),
        ('infrequent_grouping', GroupingTransformer(min_freq=0.05)),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    # This is the only categorical column that has all numbers so must convert to string first.
    p_mssubclass_onehot = Pipeline([
        ('sel_mssubclass', FeatureSelector(['MSSubClass'])),
        ('float2int', PdTypeTransformer(str)),
        ('infrequent_grouping', GroupingTransformer(min_freq=0.05)),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    # Ordinal transformations
    map_alley = {'Grvl': 0, 'Pave': 0}  # Missing values will be filled with -1
    map_LotShape = {'IR3': 0, 'IR2': 1, 'IR1': 2, 'Reg': 3}
    map_LandSlope = {'Gtl': 0, 'Mod': 1, 'Sev': 2}
    map_po_ex = {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}
    map_no_gd = {'No': 0, 'Mn': 1, 'Av': 2, 'Gd': 3}
    map_unf_glq = {'Unf': 0, 'LwQ': 1, 'Rec': 2, 'BLQ': 3, 'ALQ': 4, 'GLQ': 5}
    map_n_y = {'N': 0, 'Y': 1}
    map_Functional = {'Typ': 0, 'Min1': 1, 'Min2': 2, 'Mod': 3, 'Maj1': 4, 'Maj2': 5, 'Sev': 6, 'Sal': 7}
    map_GarageFinish = {'Unf': 0, 'RFn': 1, 'Fin': 2}
    map_PavedDrive = {'N': 0, 'P': 1, 'Y': 2}
    map_MiscFeature = {'Shed': 0}
    name_store = [['MeadowV', 'IDOTRR', 'BrDale', 'BrkSide', 'Edwards'],
                  ['OldTown', 'Sawyer', 'Blueste', 'SWISU', 'NPkVill'],
                  ['NAmes', 'Mitchel', 'SawyerW', 'NWAmes', 'Gilbert'],
                  ['Blmngtn', 'CollgCr', 'Crawfor', 'ClearCr', 'Somerst'],
                  ['Veenker', 'Timber', 'StoneBr', 'NridgHt', 'NoRidge']]
    map_Neighborhood = {name: value for value, name_group in enumerate(name_store) for name in name_group}

    ordinal_columns = ['Alley', 'LotShape', 'LandSlope', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
                       'BsmtExposure', 'BsmtFinType1', 'HeatingQC', 'CentralAir', 'KitchenQual',
                       'Functional', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond',
                       'PavedDrive', 'MiscFeature', 'Neighborhood']
    ordinal_mappings = [map_alley, map_LotShape, map_LandSlope, map_po_ex, map_po_ex, map_po_ex, map_po_ex,
                        map_no_gd, map_unf_glq, map_po_ex, map_n_y, map_po_ex,
                        map_Functional, map_po_ex, map_GarageFinish, map_po_ex, map_po_ex,
                        map_PavedDrive, map_MiscFeature, map_Neighborhood]

    ordinal_mappings = [{'col': col, 'mapping': mapping} for col, mapping in zip(ordinal_columns, ordinal_mappings)]

    p_ordinal = Pipeline([
        ('sel_ordinal', FeatureSelector(ordinal_columns)),
        ('ordinal', ce.OrdinalEncoder(mapping=ordinal_mappings))
    ])

    # Bin pool to yes or no only
    p_PoolArea_binary = Pipeline([
        ('sel_PoolArea', FeatureSelector('PoolArea')),
        ('binary', Binarizer(condition=lambda x: x > 0, name='PoolArea'))
    ])

    # Time information
    time_columns = ['YearBuilt', 'YearRemodAdd', 'MoSold', 'YrSold']
    p_time = Pipeline([
        ('sel_PoolArea', FeatureSelector(time_columns))
    ])

    p_preprocess = FeatureUnion([
        ('p_numeric', p_numeric),
        ('p_Bathroom', p_Bathrooms),
        ('p_cat_onehot', p_cat_onehot),
        ('p_Mssubclass_onehot', p_mssubclass_onehot),
        ('p_ordinal', p_ordinal),
        ('p_PoolArea_binary', p_PoolArea_binary),
        ('p_time', p_time),
    ])

    return p_preprocess


def preprocess_pipeline_4(rawdf=None):
    '''
    2) Neighborhood binning to ordinal
    3) Combine bathrooms.
    4) Better imputation of LotFrontage based on groupby Neighborhood. Added new TotalSF area feature.
    :return:
    '''
    # Ratio or Interval pipeline: Numeric pipeline
    numeric_columns = ['LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtUnfSF', '1stFlrSF', '2ndFlrSF',
                       'LowQualFinSF', 'GrLivArea',
                       'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageArea', 'WoodDeckSF',
                       'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
                       'OverallQual', 'OverallCond']
    p_numeric = Pipeline([
        ('sel_numeric', FeatureSelector(numeric_columns)),
        ('fillna_with_0', SimpleImputer())
    ])

    p_LotFrontage = Pipeline([
        ('groupbyimputer',
         PdWithinGroupImputerTransformer(func=pd.Series.mean, groupby='Neighborhood', x_names=['LotFrontage']))
    ])

    # Total SF area
    sf_columns = ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']
    p_SF = Pipeline([
        ('sel_SF', FeatureSelector(sf_columns)),
        ('fillna', PdFunctionTransformer(func=pd.Series.mean, impute=True)),
        ('sum', PdSumTransformer(series_name='TotalSF', weights=[1, 1, 1]))
    ])

    # Weighted sum of bathrooms
    bathroom_columns = ['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', ]
    p_Bathrooms = Pipeline([
        ('sel_Bathrooms', FeatureSelector(bathroom_columns)),
        ('fillna_with_0', PdFunctionTransformer(func=np.min, impute=True)),
        ('weighted_sum', PdSumTransformer(series_name='TotalBathrooms', weights=[1, 0.5, 1, 0.5]))
    ])

    # Onehot with binning infrequents
    cat_onehot_columns = ['MSZoning', 'Street', 'BldgType', 'MasVnrType', 'GarageType', 'Fence',
                          'SaleType', 'SaleCondition',

                          'LandContour', 'LotConfig', 'Condition1', 'HouseStyle', 'RoofStyle',
                          'Exterior1st', 'Foundation', 'Electrical']

    p_cat_onehot = Pipeline([
        ('sel_cat', FeatureSelector(cat_onehot_columns)),
        ('imputation', PdFunctionTransformer(func=pd.Series.mode, impute=True)),
        ('infrequent_grouping', GroupingTransformer(min_freq=0.05)),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    # This is the only categorical column that has all numbers so must convert to string first.
    p_mssubclass_onehot = Pipeline([
        ('sel_mssubclass', FeatureSelector(['MSSubClass'])),
        ('float2int', PdTypeTransformer(str)),
        ('infrequent_grouping', GroupingTransformer(min_freq=0.05)),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    # Ordinal transformations
    map_alley = {'Grvl': 0, 'Pave': 0}  # Missing values will be filled with -1
    map_LotShape = {'IR3': 0, 'IR2': 1, 'IR1': 2, 'Reg': 3}
    map_LandSlope = {'Gtl': 0, 'Mod': 1, 'Sev': 2}
    map_po_ex = {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}
    map_no_gd = {'No': 0, 'Mn': 1, 'Av': 2, 'Gd': 3}
    map_unf_glq = {'Unf': 0, 'LwQ': 1, 'Rec': 2, 'BLQ': 3, 'ALQ': 4, 'GLQ': 5}
    map_n_y = {'N': 0, 'Y': 1}
    map_Functional = {'Typ': 0, 'Min1': 1, 'Min2': 2, 'Mod': 3, 'Maj1': 4, 'Maj2': 5, 'Sev': 6, 'Sal': 7}
    map_GarageFinish = {'Unf': 0, 'RFn': 1, 'Fin': 2}
    map_PavedDrive = {'N': 0, 'P': 1, 'Y': 2}
    map_MiscFeature = {'Shed': 0}
    name_store = [['MeadowV', 'IDOTRR', 'BrDale', 'BrkSide', 'Edwards'],
                  ['OldTown', 'Sawyer', 'Blueste', 'SWISU', 'NPkVill'],
                  ['NAmes', 'Mitchel', 'SawyerW', 'NWAmes', 'Gilbert'],
                  ['Blmngtn', 'CollgCr', 'Crawfor', 'ClearCr', 'Somerst'],
                  ['Veenker', 'Timber', 'StoneBr', 'NridgHt', 'NoRidge']]
    map_Neighborhood = {name: value for value, name_group in enumerate(name_store) for name in name_group}

    ordinal_columns = ['Alley', 'LotShape', 'LandSlope', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
                       'BsmtExposure', 'BsmtFinType1', 'HeatingQC', 'CentralAir', 'KitchenQual',
                       'Functional', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond',
                       'PavedDrive', 'MiscFeature', 'Neighborhood']
    ordinal_mappings = [map_alley, map_LotShape, map_LandSlope, map_po_ex, map_po_ex, map_po_ex, map_po_ex,
                        map_no_gd, map_unf_glq, map_po_ex, map_n_y, map_po_ex,
                        map_Functional, map_po_ex, map_GarageFinish, map_po_ex, map_po_ex,
                        map_PavedDrive, map_MiscFeature, map_Neighborhood]

    ordinal_mappings = [{'col': col, 'mapping': mapping} for col, mapping in zip(ordinal_columns, ordinal_mappings)]

    p_ordinal = Pipeline([
        ('sel_ordinal', FeatureSelector(ordinal_columns)),
        ('ordinal', ce.OrdinalEncoder(mapping=ordinal_mappings))
    ])

    # Bin pool to yes or no only
    p_PoolArea_binary = Pipeline([
        ('sel_PoolArea', FeatureSelector('PoolArea')),
        ('binary', Binarizer(condition=lambda x: x > 0, name='PoolArea'))
    ])

    # Time information
    time_columns = ['YearBuilt', 'YearRemodAdd', 'MoSold', 'YrSold']
    p_time = Pipeline([
        ('sel_PoolArea', FeatureSelector(time_columns))
    ])

    p_preprocess = FeatureUnion([
        ('p_numeric', p_numeric),
        ('p_LotFrontage', p_LotFrontage),
        ('p_TotalSF', p_SF),
        ('p_Bathroom', p_Bathrooms),
        ('p_cat_onehot', p_cat_onehot),
        ('p_Mssubclass_onehot', p_mssubclass_onehot),
        ('p_ordinal', p_ordinal),
        ('p_PoolArea_binary', p_PoolArea_binary),
        ('p_time', p_time),
    ])

    return p_preprocess


def preprocess_pipeline_5(rawdf=None):
    '''
    2) Neighborhood binning to ordinal
    3) Combine bathrooms.
    4) Better imputation of LotFrontage based on groupby Neighborhood. Added new TotalSF area feature.
    5) Ordinal transformation instead of one-hot
    :return:
    '''
    # Ratio or Interval pipeline: Numeric pipeline
    numeric_columns = ['LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtUnfSF', '1stFlrSF', '2ndFlrSF',
                       'LowQualFinSF', 'GrLivArea',
                       'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageArea', 'WoodDeckSF',
                       'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
                       'OverallQual', 'OverallCond']
    p_numeric = Pipeline([
        ('sel_numeric', FeatureSelector(numeric_columns)),
        ('fillna_with_0', SimpleImputer())
    ])

    p_LotFrontage = Pipeline([
        ('groupbyimputer',
         PdWithinGroupImputerTransformer(func=pd.Series.mean, groupby='Neighborhood', x_names=['LotFrontage']))
    ])

    # Total SF area
    sf_columns = ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']
    p_SF = Pipeline([
        ('sel_SF', FeatureSelector(sf_columns)),
        ('fillna', PdFunctionTransformer(func=pd.Series.mean, impute=True)),
        ('sum', PdSumTransformer(series_name='TotalSF', weights=[1, 1, 1]))
    ])

    # Weighted sum of bathrooms
    bathroom_columns = ['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', ]
    p_Bathrooms = Pipeline([
        ('sel_Bathrooms', FeatureSelector(bathroom_columns)),
        ('fillna_with_0', PdFunctionTransformer(func=np.min, impute=True)),
        ('weighted_sum', PdSumTransformer(series_name='TotalBathrooms', weights=[1, 0.5, 1, 0.5]))
    ])

    # Onehot with binning infrequents
    cat_onehot_columns = ['MSZoning', 'Street', 'BldgType', 'MasVnrType', 'GarageType', 'Fence',
                          'SaleType', 'SaleCondition',

                          'LandContour', 'LotConfig', 'Condition1', 'HouseStyle', 'RoofStyle',
                          'Exterior1st', 'Foundation', 'Electrical']

    p_cat_onehot = Pipeline([
        ('sel_cat', FeatureSelector(cat_onehot_columns)),
        ('imputation', PdFunctionTransformer(func=pd.Series.mode, impute=True)),
        ('infrequent_grouping', GroupingTransformer(min_freq=0.05)),
        ('onehot', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    # This is the only categorical column that has all numbers so must convert to string first.
    p_mssubclass_onehot = Pipeline([
        ('sel_mssubclass', FeatureSelector(['MSSubClass'])),
        ('float2int', PdTypeTransformer(str)),
        ('infrequent_grouping', GroupingTransformer(min_freq=0.05)),
        ('onehot', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    # Ordinal transformations
    map_alley = {'Grvl': 0, 'Pave': 0}  # Missing values will be filled with -1
    map_LotShape = {'IR3': 0, 'IR2': 1, 'IR1': 2, 'Reg': 3}
    map_LandSlope = {'Gtl': 0, 'Mod': 1, 'Sev': 2}
    map_po_ex = {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}
    map_no_gd = {'No': 0, 'Mn': 1, 'Av': 2, 'Gd': 3}
    map_unf_glq = {'Unf': 0, 'LwQ': 1, 'Rec': 2, 'BLQ': 3, 'ALQ': 4, 'GLQ': 5}
    map_n_y = {'N': 0, 'Y': 1}
    map_Functional = {'Typ': 0, 'Min1': 1, 'Min2': 2, 'Mod': 3, 'Maj1': 4, 'Maj2': 5, 'Sev': 6, 'Sal': 7}
    map_GarageFinish = {'Unf': 0, 'RFn': 1, 'Fin': 2}
    map_PavedDrive = {'N': 0, 'P': 1, 'Y': 2}
    map_MiscFeature = {'Shed': 0}
    name_store = [['MeadowV', 'IDOTRR', 'BrDale', 'BrkSide', 'Edwards'],
                  ['OldTown', 'Sawyer', 'Blueste', 'SWISU', 'NPkVill'],
                  ['NAmes', 'Mitchel', 'SawyerW', 'NWAmes', 'Gilbert'],
                  ['Blmngtn', 'CollgCr', 'Crawfor', 'ClearCr', 'Somerst'],
                  ['Veenker', 'Timber', 'StoneBr', 'NridgHt', 'NoRidge']]
    map_Neighborhood = {name: value for value, name_group in enumerate(name_store) for name in name_group}

    ordinal_columns = ['Alley', 'LotShape', 'LandSlope', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
                       'BsmtExposure', 'BsmtFinType1', 'HeatingQC', 'CentralAir', 'KitchenQual',
                       'Functional', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond',
                       'PavedDrive', 'MiscFeature', 'Neighborhood']
    ordinal_mappings = [map_alley, map_LotShape, map_LandSlope, map_po_ex, map_po_ex, map_po_ex, map_po_ex,
                        map_no_gd, map_unf_glq, map_po_ex, map_n_y, map_po_ex,
                        map_Functional, map_po_ex, map_GarageFinish, map_po_ex, map_po_ex,
                        map_PavedDrive, map_MiscFeature, map_Neighborhood]

    ordinal_mappings = [{'col': col, 'mapping': mapping} for col, mapping in zip(ordinal_columns, ordinal_mappings)]

    p_ordinal = Pipeline([
        ('sel_ordinal', FeatureSelector(ordinal_columns)),
        ('ordinal', ce.OrdinalEncoder(mapping=ordinal_mappings))
    ])

    # Bin pool to yes or no only
    p_PoolArea_binary = Pipeline([
        ('sel_PoolArea', FeatureSelector('PoolArea')),
        ('binary', Binarizer(condition=lambda x: x > 0, name='PoolArea'))
    ])

    # Time information
    time_columns = ['YearBuilt', 'YearRemodAdd', 'MoSold', 'YrSold']
    p_time = Pipeline([
        ('sel_PoolArea', FeatureSelector(time_columns))
    ])

    p_preprocess = FeatureUnion([
        ('p_numeric', p_numeric),
        ('p_LotFrontage', p_LotFrontage),
        ('p_TotalSF', p_SF),
        ('p_Bathroom', p_Bathrooms),
        ('p_cat_onehot', p_cat_onehot),
        ('p_Mssubclass_onehot', p_mssubclass_onehot),
        ('p_ordinal', p_ordinal),
        ('p_PoolArea_binary', p_PoolArea_binary),
        ('p_time', p_time),
    ])

    return p_preprocess
