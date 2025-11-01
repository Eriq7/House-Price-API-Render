# house_price_api/train_and_save.py
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, PowerTransformer
from sklearn.compose import TransformedTargetRegressor

from xgboost import XGBRegressor
from feature_engine.imputation import MeanMedianImputer
from feature_engine.outliers import Winsorizer

from house_price_api.hp_components import (
    cast_three_cols, add_top_features, apply_ordinal_maps,
    select_notwinsor_numcol, winsor_cols, num, cat
)


# ====== 读数据（按你当前路径修改）======
train_df = pd.read_csv("train.csv", index_col='Id')
train_df = train_df.dropna(subset=['SalePrice'])

x = train_df.drop(columns=['SalePrice'])
y = train_df['SalePrice']

# ====== 组件 ======
caster       = FunctionTransformer(cast_three_cols,    validate=False)
fe           = FunctionTransformer(add_top_features,   validate=False)
ordinalizer  = FunctionTransformer(apply_ordinal_maps, validate=False)

num_winsor = Pipeline(steps=[
    ('impute', MeanMedianImputer(imputation_method='median', variables=winsor_cols)),
    ('winsor', Winsorizer(capping_method='quantiles', tail='right', fold=0.005, variables=winsor_cols))
])

num_other = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='median'))
])

cat_transformer = Pipeline(steps=[
    ('cat_null', SimpleImputer(strategy='most_frequent')),
    ('cat_OHE',  OneHotEncoder(handle_unknown='ignore'))
])

cols_transformer = ColumnTransformer(transformers=[
    ('num_winsor',     num_winsor,      winsor_cols),
    ('num_other',      num_other,       select_notwinsor_numcol),  # 传“函数名”，不是 select_notwinsor_numcol(x)
    ('cat_transformer',cat_transformer, cat)
])

# ====== 模型基参（你可替换为Optuna best_params）======
xgb_base = dict(
    device='cpu', tree_method='hist', grow_policy='lossguide',
    learning_rate=0.010305842703698576,
    n_estimators=2316,
    max_depth=3,
    min_child_weight=2,
    subsample=0.9366386702771559,
    colsample_bytree=0.7293783281971383,
    colsample_bynode=0.8514399053192538,
    reg_alpha=0.003694243702831284,
    reg_lambda=0.8468637682354315,
    gamma=0.08530116671956567,
    max_bin=169
)


xgb = XGBRegressor(**xgb_base)
y_transformer = PowerTransformer(method='box-cox')  # 目标正数→OK
reg = TransformedTargetRegressor(transformer=y_transformer, regressor=xgb)

pipe = Pipeline(steps=[
    ('caster',            caster),
    ('features_engineering', fe),
    ('ordinal_mapping',   ordinalizer),
    ('columns',           cols_transformer),
    ('reg',               reg),
])

# ====== 训练并保存 ======
pipe.fit(x, y)

os.makedirs("house_price_api/model", exist_ok=True)
joblib.dump(pipe, "house_price_api/model/house_price_xgb_pipe.pkl", compress=3)
print("OK -> house_price_api/model/house_price_xgb_pipe.pkl")
