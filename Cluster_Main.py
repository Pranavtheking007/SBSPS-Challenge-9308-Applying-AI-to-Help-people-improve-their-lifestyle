import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

Fit_Bit = pd.read_csv("https://raw.githubusercontent.com/Pranavtheking007/IBM_FIT-BIT/main/Fit_Bit_modified.csv")