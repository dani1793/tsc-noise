# Santoshs paths and possible other configurations
import pandas as pd
import numpy as np

pd.set_option("precision", 3)  #  display precision
pd.set_option("display.max_columns", 80)
pd.set_option("display.width", 120)

data_path = "ndvi_1_30_365_all_years_for_daniyal.csv"
result_base_path = "results/draft/"
save_flag = 0
balance_flag = 0
nclass = 2
# random_states=[99, 78, 61, 16, 73,  8, 62, 27, 30, 80]
random_state = 99
cv = 10
scoring = "roc_auc"
test_size = 0.2
train_size = 1 - test_size
fill_value = -9999
decimal_precision = 5
years = np.arange(2000, 2016)
attribute_vars = [
    "new_ID",
    "year",
    "full_cl",
    "partial_cl",
    "loss",
    "area",
    "plantcode",
    "speciescod"]
