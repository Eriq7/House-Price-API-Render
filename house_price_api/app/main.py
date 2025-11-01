# house_price_api/app/main.py
from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
import joblib, json, traceback
from pathlib import Path
from sklearn.pipeline import Pipeline

# ✅ 导入自定义组件，保证反序列化能找到函数和符号
import sys
from house_price_api.hp_components import (
    cast_three_cols,
    add_top_features,
    apply_ordinal_maps,
    select_notwinsor_numcol,
)

# ✅ 注册所有旧模型里以 "__main__" 保存的函数路径
sys.modules["__main__"].cast_three_cols = cast_three_cols
sys.modules["__main__"].add_top_features = add_top_features
sys.modules["__main__"].apply_ordinal_maps = apply_ordinal_maps
sys.modules["__main__"].select_notwinsor_numcol = select_notwinsor_numcol


import house_price_api.hp_components
# ==== 基础路径定义 ====
BASE_DIR     = Path(__file__).resolve().parents[1]
MODEL_PATH   = BASE_DIR / "model" / "house_price_xgb_pipe.pkl"
FEATURE_PATH = BASE_DIR / "model" / "feature_names_in.json"   # 可选存在


app = FastAPI(title="House Price Prediction API")


# ==== 自动识别训练列 ====
def _infer_expected_cols(model):
    """
    自动推断训练时的输入列：
      1. 尝试读取 model.feature_names_in_
      2. 尝试从 Pipeline 或 'preprocess' 步骤读取
      3. 若都无结果，退化读取 feature_names_in.json
    """
    # 1) 模型自身
    cols = getattr(model, "feature_names_in_", None)
    if cols is not None and len(cols) > 0:
        print(f"✅ EXPECTED_COLS from model.feature_names_in_: {len(cols)} cols")
        return list(cols)

    # 2) 若是 Pipeline
    if isinstance(model, Pipeline):
        cols = getattr(model, "feature_names_in_", None)
        if cols is not None and len(cols) > 0:
            print(f"✅ EXPECTED_COLS from pipeline.feature_names_in_: {len(cols)} cols")
            return list(cols)

        # 3) 若 Pipeline 内含 preprocess 步骤
        pre = model.named_steps.get("preprocess") if hasattr(model, "named_steps") else None
        if pre is not None:
            cols = getattr(pre, "feature_names_in_", None)
            if cols is not None and len(cols) > 0:
                print(f"✅ EXPECTED_COLS from preprocess.feature_names_in_: {len(cols)} cols")
                return list(cols)

    # 4) 从 JSON 读取
    try:
        with open(FEATURE_PATH, "r") as f:
            cols = json.load(f)
        if cols:
            print(f"✅ EXPECTED_COLS from feature_names_in.json: {len(cols)} cols")
            return list(cols)
    except FileNotFoundError:
        pass

    print("⚠️ Could not infer expected columns (no feature_names_in_ and no JSON).")
    return None


# ==== 启动时加载模型 ====
@app.on_event("startup")
def _load_model():
    global MODEL, EXPECTED_COLS
    MODEL = joblib.load(MODEL_PATH)
    EXPECTED_COLS = _infer_expected_cols(MODEL)


# ==== 根路径健康检测 ====
@app.get("/")
def home():
    return {"status": "running", "build": "v4-clean-retrained"}


# ==== 预测接口 ====
@app.post("/predict")
def predict(data: dict):
    try:
        df = pd.DataFrame([data])

        # 若成功加载列信息，则补齐缺失列
        if EXPECTED_COLS is not None:
            incoming = set(df.columns)
            expected = set(EXPECTED_COLS)
            missing  = list(expected - incoming)
            extra    = list(incoming - expected)

            if missing:
                for col in missing:
                    df[col] = np.nan  # 留给 pipeline 中的 Imputer 处理

            # 严格对齐顺序
            df = df.reindex(columns=EXPECTED_COLS)

            print(f"→ incoming: {len(incoming)} cols, expected: {len(expected)} cols, "
                  f"filled_missing: {len(missing)}, extra_ignored: {len(extra)}")
        else:
            raise HTTPException(
                status_code=500,
                detail=("Server has no EXPECTED_COLS. "
                        "Ensure your retrained pipeline exposes feature_names_in_.")
            )

        # 执行预测
        y = MODEL.predict(df)[0]
        return {"predicted_price": float(y)}

    except HTTPException:
        raise
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=400, detail=f"Prediction failed: {repr(e)}")
