import pandas as pd
import numpy as np

### bills
df_bills = pd.read_csv('data/bills-ca.csv')

# preprocessing
df_bills['cosponsors'] = df_bills['cosponsors'].str.split(';')
df_bills['authors'] = df_bills['authors'].str.split(';')
df_bills['date'] = pd.to_datetime(df_bills['date'], format="%Y%m%d")

### sponsors
df_sponsors = pd.read_csv('data/sponsors-ca.csv')

# preprocessing
id_pos = df_sponsors['url'].str.rfind('/')[0] + 1
df_sponsors['id'] = df_sponsors['url'].str[id_pos:]
