import pandas as pd
import numpy as np

df_raw = pd.read_csv("logs/UNSW_NB15_testing-set.csv")

print(df_raw.head())
print(df_raw["attack_cat"].value_counts())

def clean_data(df):
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Keep only Normal and DoS
    df = df[df["attack_cat"].isin(["Normal", "DoS"])]

    # Remove leakage / unnecessary columns
    cols_to_drop = ["id", "attack_cat"]
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    # Replace infinities
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Remove duplicates and missing values
    df.dropna(inplace=True)

    return df

def encode_categorical(df):
    categorical_cols = ["proto", "service", "state"]
    categorical_cols = [col for col in categorical_cols if col in df.columns]

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df

def save_clean_data(df, path):
    df.to_csv(path, index=False)

df_clean = clean_data(df_raw)
df_clean = encode_categorical(df_clean)

print("Original rows:", df_raw.shape[0])
print("Cleaned rows:", df_clean.shape[0])
print("Rows removed:", df_raw.shape[0] - df_clean.shape[0])

print(df_clean["label"].value_counts())

save_clean_data(df_clean, "logs/UNSW_NB15_DoS_test_cleaned.csv")

print("Cleaned dataset shape:", df_clean.shape)