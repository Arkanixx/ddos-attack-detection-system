import pandas as pd
import numpy as np

#loading the data without doing any preprocessing
df_raw = pd.read_csv('logs/Wednesday-21-02-2018_TrafficForML_CICFlowMeter_DDoS.csv')

print(df_raw.head())

#---------------loading the data after doing the preprocessing-------------
def clean_data(df):
   
   df = df.copy()

   #clean column names
   df.columns = df.columns.str.strip()

   #replace any infinite values
   df.replace([np.inf, -np.inf], np.nan, inplace=True)

   #Convert the timestamp column string to datetime
   df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce")

   #handle any missing values
   df.dropna(subset=["Label", "Timestamp"], inplace=True)

    #remove duplicates
   df.drop_duplicates(inplace=True)

   return df

def encode_labels(df):
   df["Label"] = df["Label"].apply(lambda x: 1 if x != "Benign" else 0 )
   return df

def save_clean_data(df, path):
   df.to_csv(path, index=False)

# ---------run preprocess pipeline--------------

df_clean = clean_data(df_raw)
df_clean = encode_labels(df_clean)

# logging dataset changes
print("Original rows:", df_raw.shape[0])
print("Cleaned rows:", df_clean.shape[0])
print("Rows removed:", df_raw.shape[0] - df_clean.shape[0])

save_clean_data(df_clean, "logs/WednesdayTraffic_cleaned.csv")

print("Cleaned dataset shape:", df_clean.shape)

# print attack vs benign count
print(df_clean["Label"].value_counts())