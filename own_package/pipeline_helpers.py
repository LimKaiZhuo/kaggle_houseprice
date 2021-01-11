import pandas as pd
import numpy as np
import scipy.stats
import category_encoders as ce
from xgboost import XGBRegressor
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
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


class TypePdTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, type_):
        self.type_ = type_

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.astype(self.type_)


class Binarizer(BaseEstimator, TransformerMixin):
    def __init__(self, condition, name):
        self.condition = condition
        self.name = name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(lambda x: int(self.condition(x))).to_frame(self.name)


class GroupingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, *, min_count=0, min_freq=0.0, top_n=0, group_name='other'):
        self.min_count = min_count
        self.min_freq = min_freq
        self.top_n = top_n
        self.group_name = 'other'

    def fit(self, X, y=None):
        X = X.fillna('None')
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


def preprocess_pipeline_1():
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
        ('float2int', TypePdTransformer(str)),
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


def preprocess_pipeline_2():
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
        ('float2int', TypePdTransformer(str)),
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
