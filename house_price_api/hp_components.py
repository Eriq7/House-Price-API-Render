# house_price_api/hp_components.py
import numpy as np
import pandas as pd
from sklearn.compose import make_column_selector as selector


# ====== 有序映射表 ======
ordinal_map = {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'NA':0}
bsmtFinType_map = {'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1,'NA':0}
bsmtExpo_map    = {'Gd':4,'Av':3,'Mn':2,'No':1,'NA':0}

ord_cols = ['ExterQual','ExterCond','BsmtQual','BsmtCond',
            'HeatingQC','KitchenQual','GarageQual','GarageCond']
fin_cols = ['BsmtFinType1','BsmtFinType2']
expo_cols = ['BsmtExposure']

ORDINAL_SPECS = [
    (ord_cols, ordinal_map),
    (fin_cols, bsmtFinType_map),
    (expo_cols, bsmtExpo_map),
]

# ====== 列选择器 & 固定winsor列 ======
num = selector(dtype_include=np.number)
cat = selector(dtype_include=["object", "string", "category"])
winsor_cols = ['GrLivArea', 'TotalBsmtSF']   # 你之前就定了这两列

def select_notwinsor_numcol(X):
    """返回：所有数值列 - winsor_cols"""
    return [c for c in num(X) if c not in winsor_cols]

# ====== FunctionTransformer 对应函数 ======
def cast_three_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ['MSSubClass', 'MoSold']:
        if c in df.columns:
            df[c] = df[c].astype('str')
    return df

def add_top_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 1) TotalSF
    if {'GrLivArea','TotalBsmtSF'}.issubset(df.columns):
        df['TotalSF'] = df['GrLivArea'].fillna(0) + df['TotalBsmtSF'].fillna(0)
    # 2) TotalBath
    for c in ['FullBath','HalfBath','BsmtFullBath','BsmtHalfBath']:
        if c not in df.columns: df[c] = 0
    df['TotalBath'] = (df['FullBath'].fillna(0)
                       + 0.5*df['HalfBath'].fillna(0)
                       + df['BsmtFullBath'].fillna(0)
                       + 0.5*df['BsmtHalfBath'].fillna(0))
    # 3) TotalPorchSF
    cols_porch = ['OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','WoodDeckSF']
    for c in cols_porch:
        if c not in df.columns: df[c] = 0
    df['TotalPorchSF'] = df[cols_porch].fillna(0).sum(axis=1)
    # 4) HouseAge & RemodAge
    if 'YearBuilt' in df.columns and 'YrSold' in df.columns:
        df['HouseAge'] = (df['YrSold'].fillna(df['YrSold'].median())
                          - df['YearBuilt'].fillna(df['YearBuilt'].median())).clip(lower=0)
    if 'YearRemodAdd' in df.columns and 'YrSold' in df.columns:
        df['RemodAge'] = (df['YrSold'].fillna(df['YrSold'].median())
                          - df['YearRemodAdd'].fillna(df['YearRemodAdd'].median())).clip(lower=0)
    # 5) log1p 面积族
    for c in ['GrLivArea','TotalBsmtSF','LotArea','LotFrontage','GarageArea','TotalSF','TotalPorchSF']:
        if c in df.columns:
            df[f'log1p_{c}'] = np.log1p(df[c])
    return df

def apply_ordinal_maps(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 先统一为字符串并填 'NA'
    for cols, _ in ORDINAL_SPECS:
        for c in cols:
            if c in df.columns:
                df[c] = df[c].astype('string').str.strip().fillna('NA')
    # 再映射为分数
    for cols, mapping in ORDINAL_SPECS:
        for c in cols:
            if c in df.columns:
                df[c] = df[c].map(mapping).fillna(0).astype('int16')
    return df

