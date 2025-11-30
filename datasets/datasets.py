from auton_survival.datasets import load_dataset as load_dsm
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from pycox import datasets
import pandas as pd
import numpy as np
import os

EPS = 1e-8

def load_dataset(dataset='SUPPORT', path = './', normalize = True, **kwargs):
    if dataset == 'GBSG':
        df = datasets.gbsg.read_df()
    elif dataset == 'METABRIC':
        df = datasets.metabric.read_df()
        df = df.rename(columns = {'x0': 'MKI67', 'x1': 'EGFR', 'x2': 'PGR', 'x3': 'ERBB2', 
                                  'x4': 'Hormone', 'x5': 'Radiotherapy', 'x6': 'Chemotherapy', 'x7': 'ER-positive', 
                                  'x8': 'Age at diagnosis'})
        df['duration'] += EPS # Avoid problem of the minimum value 0
    elif dataset == 'SYNTHETIC':
        df = datasets.rr_nl_nhp.read_df()
        df = df.drop([c for c in df.columns if 'true' in c], axis = 'columns')
    elif dataset == 'SEER':
        dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(dir, "seer", "seernfg.csv")
        df = pd.read_csv(path, dtype={3: "string"})
        df = process_seer(df)
        df['duration'] += EPS # Avoid problem of the minimum value 0
    elif dataset == 'SYNTHETIC_COMPETING':
        df = pd.read_csv('https://raw.githubusercontent.com/chl8856/DeepHit/master/sample%20data/SYNTHETIC/synthetic_comprisk.csv')
        df = df.drop(columns = ['true_time', 'true_label']).rename(columns = {'label': 'event', 'time': 'duration'})
        df['duration'] += EPS # Avoid problem of the minimum value 0
    elif dataset == 'PBC':
        dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(dir, "PBC_Data.txt")
        df = pd.read_csv(filepath,
                         sep=r"\s+",  # split on any whitespace
                         header=None)
        column_names = [
            "case_number", "duration", "status", "drug", "age_days", "sex",
            "ascites", "hepatomegaly", "spiders", "edema", "bilirubin",
            "cholesterol", "albumin", "urine_copper", "alk_phosphatase",
            "sgot", "triglycerides", "platelets", "prothrombin_time", "histologic_stage"
        ]
        df.columns = column_names
        df = df.replace(".", np.nan)
        df = df.dropna(axis=1)
        df["event"] = (df["status"] == 2)
        df = df.apply(pd.to_numeric)
        df['duration'] += EPS  # Avoid problem of the minimum value 0
    elif dataset == 'SUPPORT':
        df = datasets.support.read_df()
        df.columns = [
            'age', 'sex', 'race', 'number_comorbidities', 'diabetes', 'dementia',
            'cancer', 'mean_bp', 'heart_rate', 'respiration_rate', 'temperature',
            'white_blood_cell_count', 'serums_sodium', 'serums_creatinine', 'duration', 'event'
        ]
        df['duration'] += EPS  # Avoid problem of the minimum value 0
    else:
        return load_dsm(dataset, normalize = normalize, **kwargs)

    covariates = df.drop(['duration', 'event'], axis = 'columns')
    return StandardScaler().fit_transform(covariates.values).astype(float) if normalize else covariates.values.astype(float),\
           df['duration'].values.astype(float),\
           df['event'].values.astype(int),\
           covariates.columns

def process_seer(df):
    # Remove multiple visits
    df = df.groupby('Patient ID').first().drop(columns= ['Site recode ICD-O-3/WHO 2008']).copy()

    # Encode using dictionary to remove missing data
    df["RX Summ--Surg Prim Site (1998+)"] = (df["RX Summ--Surg Prim Site (1998+)"].replace('126', np.nan))
    df["Sequence number"] = (df["Sequence number"].replace(['88', '99'], np.nan))
    df["Regional nodes positive (1988+)"] = (df["Regional nodes positive (1988+)"].replace(['95', '96', '97', '98', '99', '126'], np.nan))
    df["Regional nodes examined (1988+)"] = (df["Regional nodes examined (1988+)"].replace(['95', '96', '97', '98', '99', '126'], np.nan))
    df = df.replace(['Blank(s)', 'Unknown'], np.nan).rename(columns = {"Survival months": "duration"})

    # Remove patients without survival time
    df = df[~df.duration.isna()]

    # Outcome
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].where(df[col].notna(), np.nan)
    df['duration'] = df['duration'].astype(float)
    df['event'] = (df["SEER cause-specific death classification"] == "Dead (attributable to this cancer dx)").astype(int) # Death

    df = df.drop(columns = ["COD to site recode"])

    # Imput and encode categorical
    ## Categorical
    categorical_col = ["Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)", "Laterality", 
        "Diagnostic Confirmation", "Histology recode - broad groupings", "Chemotherapy recode (yes, no/unk)",
        "Radiation recode", "ER Status Recode Breast Cancer (1990+)", "PR Status Recode Breast Cancer (1990+)",
        "Histologic Type ICD-O-3", "ICD-O-3 Hist/behav, malignant", "Sequence number", "RX Summ--Surg Prim Site (1998+)",
        "CS extension (2004-2015)", "CS lymph nodes (2004-2015)", "CS mets at dx (2004-2015)", "Origin recode NHIA (Hispanic, Non-Hisp)"]
    ordinal_col = ["Age recode with <1 year olds", "Grade Recode (thru 2017)", "Year of diagnosis"]
    for col in categorical_col:
        df[col] = df[col].where(df[col].notna(), np.nan)
        df[col] = df[col].astype(str)
        df[col] = df[col].replace('nan', np.nan)
    for col in ordinal_col:
        df[col] = df[col].astype(str).replace('nan', np.nan)
    valid_grades = [
        'Well differentiated; Grade I',
        'Moderately differentiated; Grade II',
        'Poorly differentiated; Grade III',
        'Undifferentiated; anaplastic; Grade IV'
    ]
    df['Grade Recode (thru 2017)'] = df['Grade Recode (thru 2017)'].where(
        df['Grade Recode (thru 2017)'].isin(valid_grades), np.nan
    )
    df['Age recode with <1 year olds'] = (
        df['Age recode with <1 year olds'].replace('00 years', '00-00 years')
    )

    imputer = SimpleImputer(strategy='most_frequent')
    enc = OrdinalEncoder()
    df_cat = pd.DataFrame(enc.fit_transform(imputer.fit_transform(df[categorical_col])), columns = categorical_col, index = df.index)
    df_ord = pd.DataFrame(imputer.fit_transform(df[ordinal_col]), columns = ordinal_col, index = df.index)

    pd.set_option('future.no_silent_downcasting', True)
    df_ord = df_ord.replace(
      {age: number
        for number, age in enumerate(['00-00 years', '01-04 years', '05-09 years', '10-14 years', '15-19 years', '20-24 years', '25-29 years',
        '30-34 years', '35-39 years', '40-44 years', '45-49 years', '50-54 years', '55-59 years', 
        '60-64 years', '65-69 years', '70-74 years', '75-79 years', '80-84 years', '85+ years'])
      }).replace({
        grade: number
        for number, grade in enumerate(['Well differentiated; Grade I', 'Moderately differentiated; Grade II',
       'Poorly differentiated; Grade III', 'Undifferentiated; anaplastic; Grade IV'])
      })
    ## Numerical
    numerical_col = ["Total number of in situ/malignant tumors for patient", "Total number of benign/borderline tumors for patient",
          "CS tumor size (2004-2015)", "Regional nodes examined (1988+)", "Regional nodes positive (1988+)"]
    imputer = SimpleImputer(strategy='mean')
    df_num = pd.DataFrame(imputer.fit_transform(df[numerical_col].astype(float)), columns = numerical_col, index = df.index)

    return pd.concat([df_cat, df_num, df_ord, df[['duration', 'event']]], axis = 1)
    
    