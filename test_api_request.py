import requests

payload = {
  "MSSubClass":"60","MSZoning":"RL","LotArea":2000,"Street":"Pave",
  "LotShape":"Reg","LandContour":"Lvl","LotFrontage":65,
  "Neighborhood":"CollgCr","YearBuilt":2003,"YearRemodAdd":2003,
  "OverallQual":7,"OverallCond":5,"GrLivArea":1710,"TotalBsmtSF":856,
  "FullBath":2,"HalfBath":1,"BedroomAbvGr":3,"KitchenAbvGr":1,
  "TotRmsAbvGrd":7,"Fireplaces":0,"GarageCars":2,"GarageArea":999,
  "YrSold":1987,"MoSold":"2"
}

r = requests.post("http://127.0.0.1:8003/predict", json=payload)
print("Status code:", r.status_code)
print("Response:", r.text)