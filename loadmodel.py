import joblib
import pandas as pd
import numpy as np

model = joblib.load("./corona.joblib")
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
test = np.array([3, 0, 4, 2], ndmin=2)
#test = np.array([ 19.69,	21.25,	130,	1203,	0.1096,	0.1599,	0.1974,	0.1279,	0.2069,	0.05999,	0.7456,	0.7869,	4.585,	94.03,	0.00615,	0.04006,	0.03832,	0.02058,	0.0225,	0.004571,	23.57,	25.53,	152.5,	1709,	0.1444,	0.4245,	0.4504,	0.243,	0.3613,	0.08758,], ndmin=2)
# well = pd.DataFrame(test)
# well = sc.transform(well)

finally_prediction = model.predict(test)
print(finally_prediction)