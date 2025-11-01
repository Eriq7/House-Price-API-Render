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

# === NEW: 前端需要的响应类型（HTML）
from fastapi.responses import HTMLResponse  # === NEW

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

# === NEW: 健康检查（供 Render 探活）
@app.get("/healthz")
def healthz():  # === NEW
    return {"status": "ok", "build": "v4-clean-retrained"}  # === NEW

# === NEW: 根路径提供一个极简前端（HTML），用于粘贴 JSON 并调用 /predict
@app.get("/", include_in_schema=False, response_class=HTMLResponse)  # === NEW
def home():  # === NEW
    # 前端不强依赖具体特征；用户可直接粘贴 JSON 调用 /predict
    # 你也可以把下面的 placeholder JSON 替换成你实际的最小样例
    placeholder = {
          "MSSubClass":"60","MSZoning":"RL","LotArea":2000,"Street":"Pave",
  "LotShape":"Reg","LandContour":"Lvl","LotFrontage":65,
  "Neighborhood":"CollgCr","YearBuilt":2003,"YearRemodAdd":2003,
  "OverallQual":7,"OverallCond":5,"GrLivArea":1710,"TotalBsmtSF":856,
  "FullBath":2,"HalfBath":1,"BedroomAbvGr":3,"KitchenAbvGr":1,
  "TotRmsAbvGrd":7,"Fireplaces":0,"GarageCars":2,"GarageArea":999,
  "YrSold":1987,"MoSold":"2"
        }
    return f"""  <!-- === NEW: 简易前端 HTML 开始 === -->
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>House Price Prediction API</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif; padding: 24px; max-width: 900px; margin: auto; }}
    h1 {{ margin-bottom: 8px; }}
    textarea {{ width: 100%; height: 220px; font-family: ui-monospace, Menlo, Consolas, monospace; }}
    button {{ padding: 8px 14px; cursor: pointer; }}
    .row {{ display: flex; gap: 16px; align-items: center; flex-wrap: wrap; }}
    .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 12px; }}
    pre {{ white-space: pre-wrap; word-break: break-word; background: #111; color: #eee; padding: 12px; border-radius: 8px; }}
    .muted {{ color: #666; }}
  </style>
</head>
<body>
  <h1>House Price Prediction API</h1>
  <p class="muted">Status: running • Build: v4-clean-retrained • Try the interactive docs at <a href="/docs">/docs</a></p>

  <div class="card">
    <h3>1) Paste JSON payload for <code>/predict</code></h3>
    <textarea id="payload">{json.dumps(placeholder, indent=2)}</textarea>
    <div class="row">
      <button onclick="doPredict()">Send to /predict</button>
      <span class="muted">Content-Type: application/json</span>
    </div>
  </div>

  <div class="card" style="margin-top:16px;">
    <h3>2) Response</h3>
    <pre id="out">—</pre>
  </div>

  <script>
    async function doPredict() {{
      const out = document.getElementById('out');
      out.textContent = 'Requesting...';
      let body;
      try {{
        body = JSON.parse(document.getElementById('payload').value);
      }} catch (e) {{
        out.textContent = '❌ Invalid JSON: ' + e.message;
        return;
      }}
      try {{
        const r = await fetch('/predict', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify(body)
        }});
        const text = await r.text();
        out.textContent = text;
      }} catch (e) {{
        out.textContent = '❌ Request failed: ' + e.message;
      }}
    }}
  </script>
</body>
</html>
"""  # === NEW: 简易前端 HTML 结束 ===

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
