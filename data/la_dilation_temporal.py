#!/usr/bin/env python
# coding: utf-8

# # Syngo: LA Dilation

# In[1]:


import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

import os


# In[2]:


data_dir = 'data/dataset/Lee_Echo_Syngo'
adler = pd.read_csv(os.path.join(data_dir, 'Adler.csv'))
analytics_report = pd.read_csv(os.path.join(data_dir, 'Analytics_Report.csv'))
analytics_study = pd.read_csv(os.path.join(data_dir, 'AnalyticsStudy.csv'))
department = pd.read_csv(os.path.join(data_dir, 'Department.csv'))
field_map = pd.read_csv(os.path.join(data_dir, 'FieldMap.csv'))
measurement_type = pd.read_csv(os.path.join(data_dir, 'MeasurementType.csv'))
modalities = pd.read_csv(os.path.join(data_dir, 'Modalities.csv'))
observations = pd.read_csv(os.path.join(data_dir, 'Observations.csv'))
study_details = pd.read_csv(os.path.join(data_dir, 'StudyDetails.csv'))


# In[3]:


aws_uhn = pd.read_csv('aws/aws_uhn.csv', index_col=0)
print(aws_uhn.shape)


# In[4]:


syn_df = observations.loc[observations['Name'] == 'LA_size-ASE_obs']
syn_df = syn_df.dropna(subset=['Value'])


# In[5]:


syn_df.head()


# In[6]:


print(len(syn_df))
syn_df['Value'].value_counts()


# In[8]:


# list of values to drop
values_to_drop = ['cardiac_transplant', 'dilated']

# filter the DataFrame
syn_df = syn_df[~syn_df['Value'].isin(values_to_drop)]

mapping_binary = {
    'normal': 'normal',
    # 'dilated': 'dilated',
    'mildly_dilated': 'normal',
    'severely_dilated': 'dilated',
    'moderately_dilated': 'dilated',
}


# In[9]:


label_col = 'LA_Dilated_Binary'
syn_df[label_col] = syn_df['Value'].map(mapping_binary)


# In[10]:


print(len(syn_df))
syn_df[label_col].value_counts()


# In[11]:


# Create a boolean mask to identify matching StudyRef values
matching_mask = syn_df['StudyRef'].isin(aws_uhn['STUDY_REF'])

# Count the number of matches (True values)
number_of_matches = matching_mask.sum()

# Get the total number of unique studies in mv_obs for context
total_unique_studies = syn_df['StudyRef'].nunique()

print(f"\nTotal unique StudyRef values in syn_df: {total_unique_studies}")
print(f"Number of 'StudyRef' values from syn_df that are in aws_uhn: {number_of_matches}")
print(f"Percentage of matching studies: {(number_of_matches / total_unique_studies) * 100:.2f}%")


# In[12]:


import pandas as pd

# Assuming 'mv_obs' and 'aws_uhn' are your DataFrames loaded as in the screenshots.

# Step 1: Merge the two DataFrames
# We'll merge 'mv_obs' with 'aws_uhn' using 'StudyRef' and 'STUDY_REF' as the keys.
merged_df = pd.merge(syn_df, aws_uhn, left_on='StudyRef', right_on='STUDY_REF', how='inner')

# Step 2: Define the desired columns in the correct order
final_columns = [
    'STUDY_REF',
    's3_key',
    'Value',
    label_col,
    'PATIENT_ID',
    'STUDY_DATE',
    'STUDY_TIME',
    'DeidentifiedStudyID',
    'OriginalStudyID'
]

# # Step 3: Create the new table with only the specified columns
syn_df_labels = merged_df[final_columns]

# convert numeric date to datetime
syn_df_labels['STUDY_DATE'] = pd.to_datetime(
    syn_df_labels['STUDY_DATE'].astype(str), 
    format='%Y%m%d'
)

# now sortable by date
syn_df_labels = syn_df_labels.sort_values('STUDY_DATE')
syn_df_labels['STUDY_DATE'] = syn_df_labels['STUDY_DATE'].dt.strftime('%Y-%m-%d')


# In[13]:


print(syn_df_labels.shape)
print(syn_df_labels[label_col].value_counts())
print(syn_df_labels.columns)
display(syn_df_labels.head())


# # HeartLab: LA Dilation

# In[14]:


import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

import os


# In[15]:


data_dir = 'data/dataset'

msmt_defs = pd.read_csv(os.path.join(data_dir,'MEASUREMENT_DEFINITIONS.csv'))
msmt_formula = pd.read_csv(os.path.join(data_dir,'MEASUREMENT_FORMULA.csv'))
msmt_groups = pd.read_csv(os.path.join(data_dir,'MEASUREMENT_GROUPS.csv'))
msmt_intersects = pd.read_csv(os.path.join(data_dir,'MEASUREMENT_INTERSECTS.csv'))
msmt_lists = pd.read_csv(os.path.join(data_dir,'MEASUREMENT_LISTS.csv'))
msmt = pd.read_csv(os.path.join(data_dir,'MEASUREMENTS.csv'))

findings = pd.read_csv(os.path.join(data_dir,'ENCOADMIN_FINDINGS.csv'))
finding_groups = pd.read_csv(os.path.join(data_dir,'ENCOADMIN_FINDING_GROUPS.csv'))
finding_intersects = pd.read_csv(os.path.join(data_dir,'ENCOADMIN_FINDING_INTERSECTS.csv'))

patients = pd.read_csv(os.path.join(data_dir,'Patients_No_PHI.csv'))
reports = pd.read_csv(os.path.join(data_dir,'REPORTS.csv'))
series = pd.read_csv(os.path.join(data_dir,'SERIES.csv'))
studies = pd.read_csv(os.path.join(data_dir,'STUDIES.csv'))


# In[16]:


hl_link = pd.read_csv(os.path.join(data_dir, 'heartlab_rep_study_video.csv'))


# In[17]:


hl_link.head()


# In[18]:


aws_heartlab = pd.read_csv('aws/aws_heartlab_0806.csv', index_col=0)
print(aws_heartlab.shape)


# In[19]:


finding_intersects = finding_intersects.merge(
    findings[['ID', 'HLCODE']], 
    left_on='FIN_ID', 
    right_on='ID', 
    how='left'
).drop(columns='ID_y').rename(columns={'ID_x': 'ID'})


# In[20]:


codes = finding_intersects['HLCODE'].str.strip().str.lower()
# hl_df = finding_intersects[codes.str.contains('tricuspid valve.*regurgitation', na=False)]
hl_df = finding_intersects[codes.str.contains('la cavity size', na=False)]


# In[21]:


hl_df['HLCODE'].value_counts()


# In[22]:


hl_df['HLCODE'].unique()


# In[23]:


keep = [
    'left atrium^morphology^la cavity size^mildly dilated',
    'left atrium^morphology^la cavity size^normal',
    'left atrium^morphology^la cavity size^moderately dilated',
    'local^left atrium^morphology^la cavity size^upper normal^1561',
    # 'left atrium^morphology^la cavity size^dilated',
    'left atrium^morphology^la cavity size^severely dilated'
]

hl_df = hl_df[hl_df['HLCODE'].isin(keep)].copy()


# In[24]:


mapping_two = {
    'left atrium^morphology^la cavity size^mildly dilated': 'normal', # prev dilated
    'left atrium^morphology^la cavity size^normal': 'normal',
    'left atrium^morphology^la cavity size^moderately dilated': 'dilated',
    'local^left atrium^morphology^la cavity size^upper normal^1561': 'normal',
    # 'left atrium^morphology^la cavity size^dilated': 'dilated', # remove
    'left atrium^morphology^la cavity size^severely dilated': 'dilated'
}

hl_df[label_col] = hl_df['HLCODE'].map(mapping_two)


# In[25]:


print(hl_df[label_col].dropna().value_counts())


# In[26]:


# REP_IDs that appear in the hl_df dataframe
hl_ids = hl_df['REP_ID'].dropna().unique()
print(len(hl_ids))

# Grab columns from the hl_link dataframes using REP_IDs
hl_df_labels = hl_link[hl_link['REP_ID'].isin(hl_ids)].copy()


# In[27]:


hl_df_labels = hl_df.merge(
    hl_df_labels,
    on='REP_ID',
    how='inner',
    suffixes=('', '_drop')  # keep REP_ID as-is, add suffix to duplicates
)

# Remove the duplicate REP_ID column if created
if 'REP_ID_drop' in hl_df_labels.columns:
    hl_df_labels = hl_df_labels.drop(columns=['REP_ID_drop'])

hl_df_labels = hl_df_labels.merge(
    aws_heartlab[['OriginalStudyID', 's3_key']],  # only needed columns
    on='OriginalStudyID',
    how='left'
)

hl_df_labels = hl_df_labels.dropna(subset=['s3_key'])

# convert STUDY_DATE to datetime objects
hl_df_labels['STUDY_DATE'] = pd.to_datetime(
    hl_df_labels['STUDY_DATE'], 
    format='%m/%d/%y %H:%M:%S'  # matches "01/20/09 10:29:09"
)



# now you can sort
hl_df_labels = hl_df_labels.sort_values('STUDY_DATE')
hl_df_labels['STUDY_DATE'] = hl_df_labels['STUDY_DATE'].dt.strftime('%Y-%m-%d')

hl_df_labels = hl_df_labels.rename(
    columns={'OriginalPatientID': 'PATIENT_ID'}
)

print(hl_df_labels.columns)
print(hl_df_labels.shape)
display(hl_df_labels.head())


# # Combine

# In[28]:


# Columns you want to keep
keep_cols = [
    'STUDY_REF','REP_ID','HLCODE', 'Value', label_col ,'s3_key',  'STUDY_DATE',
    'STUDY_TIME', 'DeidentifiedStudyID', 'OriginalStudyID',
    'PATIENT_ID', 'DeidentifiedPatientID', 'SERI_ID', 'STUDY_ID',
    'STUDY_INSTANCE_UID'
]

# Combine both dataframes
uhn_df = pd.concat([syn_df_labels, hl_df_labels], ignore_index=True, sort=False)

# Drop overlapping OriginalStudyIDs (keep first occurrence)
uhn_df = uhn_df.drop_duplicates(subset='OriginalStudyID', keep='first')

# Restrict to selected columns
uhn_df = uhn_df.reindex(columns=keep_cols)

# Drop rows where there is no s3 key
uhn_df = uhn_df.dropna(subset=['s3_key'])

# Ensure STUDY_DATE is datetime before sorting
uhn_df['STUDY_DATE'] = pd.to_datetime(uhn_df['STUDY_DATE'], errors='coerce')

# Sort by STUDY_DATE (earliest first)
uhn_df = uhn_df.sort_values('STUDY_DATE')

print(uhn_df.shape)
display(uhn_df.head())


# In[29]:


import matplotlib.pyplot as plt
import pandas as pd

# Ensure STUDY_DATE is datetime
uhn_df['STUDY_DATE'] = pd.to_datetime(uhn_df['STUDY_DATE'], errors='coerce')

# Extract year
uhn_df['YEAR'] = uhn_df['STUDY_DATE'].dt.year

# Plot histogram
plt.figure(figsize=(10,5))
uhn_df['YEAR'].dropna().astype(int).hist(bins=uhn_df['YEAR'].nunique())
plt.xlabel('Year')
plt.ylabel('Number of Studies')
plt.title('Histogram of STUDY_DATE by Year')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[30]:


# Final label counts for the selected condition
uhn_df[label_col].value_counts()


# In[31]:


# uhn_df.to_csv('csv/uhn_lad_0919.csv')
uhn_df.to_csv('csv/uhn_lad_0922.csv')


# # Create Splits

# In[32]:


import pandas as pd
from pathlib import Path

all_es = pd.read_parquet('all_es_combined.parquet')


# In[33]:


# Create a new DataFrame called 'a4c_df' by filtering 'all_es'
# all_es = combined_df
a4c_videos = all_es[all_es['pred_view'] == 'a4c'].copy()

# This regex finds 'echo-study', 'echo-study-1', or 'echo-study-2',
# and then captures the sequence of characters that follows until the next slash.
regex_pattern = r'results/echo-study(?:-[12])?/([^/]+)'

# Use .str.extract() to pull out the captured group (the Study ID)
a4c_videos['DeidentifiedStudyID'] = a4c_videos['png_uri'].str.extract(regex_pattern)

a4c_videos = a4c_videos[['DeidentifiedStudyID', 'mp4_uri_corrected']].rename(
    columns={'mp4_uri_corrected': 'URI'}
)


# In[34]:


print(a4c_videos.shape)
display(a4c_videos.head())


# In[35]:


# combined_labels = pd.read_csv('csv/uhn_lad_0919.csv')
combined_labels = pd.read_csv('csv/uhn_lad_0922.csv')


# In[36]:


# Select only the necessary columns from your labels DataFrame for efficiency
labels_to_merge = combined_labels[['DeidentifiedStudyID', 'STUDY_DATE', 'PATIENT_ID', label_col]]

# Merge the two DataFrames to find the overlap and add the 'Value'
# 'how="inner"' ensures only matching DeidentifiedStudyIDs are kept
a4c_dataset = pd.merge(
    a4c_videos, 
    labels_to_merge, 
    on='DeidentifiedStudyID', 
    how='inner'
)

print(f"Found {len(a4c_dataset)} A4C videos.")
a4c_dataset.head()


# In[37]:


a4c_dataset.to_csv(f"csv/a4c_{label_col}.csv")


# In[38]:


import pandas as pd

def build_temporal_patient_splits(
    df,
    label_mapping,
    label_col,
    date_col="STUDY_DATE",
    patient_col="PATIENT_ID",
    train_frac=0.70,
    val_frac=0.15,              # test = 1 - train - val
    anchor="min",               # 'min' | 'median' | 'max' patient date to order by
    enforce_time_windows=True,  # drop rows outside each split’s time window
):
    assert 0 < train_frac < 1 and 0 < val_frac < 1 and train_frac + val_frac < 1
    d = df.copy()

    # labels → ints
    num_col = f"{label_col}_numeric"
    d[num_col] = d[label_col].map(label_mapping)

    # dates
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col, num_col, patient_col])

    # ----- patient-level temporal ordering -----
    stats = d.groupby(patient_col)[date_col].agg(["min", "median", "max"])
    anchor_series = stats[anchor].sort_values()  # one timestamp per patient, ordered
    patients = anchor_series.index.to_list()

    n = len(patients)
    cut1 = int(n * train_frac)
    cut2 = int(n * (train_frac + val_frac))

    train_p = set(patients[:cut1])
    val_p   = set(patients[cut1:cut2])
    test_p  = set(patients[cut2:])

    train_df = d[d[patient_col].isin(train_p)]
    val_df   = d[d[patient_col].isin(val_p)]
    test_df  = d[d[patient_col].isin(test_p)]

    # optional: strictly enforce temporal windows by date (no cross-window rows)
    if enforce_time_windows:
        # windows from global item-level dates
        q1 = d[date_col].quantile(train_frac)
        q2 = d[date_col].quantile(train_frac + val_frac)
        train_df = train_df[train_df[date_col] < q1]
        val_df   = val_df[(val_df[date_col] >= q1) & (val_df[date_col] < q2)]
        test_df  = test_df[test_df[date_col] >= q2]

    # safety: no patient leakage
    assert not (set(train_df[patient_col]) & set(val_df[patient_col]))
    assert not (set(train_df[patient_col]) & set(test_df[patient_col]))
    assert not (set(val_df[patient_col])   & set(test_df[patient_col]))

    return train_df, val_df, test_df


# In[39]:


print(a4c_dataset[label_col].value_counts())

# Create mapping automatically from unique labels
label_mapping = {label: idx 
                 for idx, label in enumerate(sorted(a4c_dataset[label_col].unique()))}

print(label_mapping)


# In[40]:


print(f"a4c_{label_col}_train.csv")


# In[42]:


train_df, val_df, test_df = build_temporal_patient_splits(a4c_dataset, label_mapping, label_col)

num_col = f"{label_col}_numeric"   # e.g., 'Dilated_Binary_numeric'

train_df[['URI', num_col]].to_csv(f"csv/a4c_{label_col}_train.csv", header=False, index=False, sep=' ')
val_df[['URI', num_col]].to_csv(  f"csv/a4c_{label_col}_val.csv",   header=False, index=False, sep=' ')
test_df[['URI', num_col]].to_csv( f"csv/a4c_{label_col}_test.csv",  header=False, index=False, sep=' ')


# Print split statistics  
print(f"Total samples: {len(a4c_dataset)}")  
print(f"Train: {len(train_df)} ({len(train_df)/len(a4c_dataset)*100:.1f}%)")  
print(f"Validation: {len(val_df)} ({len(val_df)/len(a4c_dataset)*100:.1f}%)")  
print(f"Test: {len(test_df)} ({len(test_df)/len(a4c_dataset)*100:.1f}%)")  

# Print label distribution for each split  
print("\nLabel distribution:")  
for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:  
    print(f"{split_name}:")  
    for label, count in split_df[num_col].value_counts().sort_index().items():  
        original_label = [k for k, v in label_mapping.items() if v == label][0]  
        print(f"  {label} ({original_label}): {count}")  


# In[43]:


import matplotlib.pyplot as plt

date_col = "STUDY_DATE"
num_col  = f"{label_col}_numeric"

def year_hist(df, title):
    years = pd.to_datetime(df[date_col], errors="coerce").dt.year.dropna().astype(int)
    counts = years.value_counts().sort_index()

    # print table
    print(f"\n{title} year counts:")
    print(counts.to_string())

    # plot one chart per split
    plt.figure(figsize=(9,4))
    counts.plot(kind="bar")
    plt.title(f"{title}: samples per year")
    plt.xlabel("Year"); plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

year_hist(train_df, "Train")
year_hist(val_df,   "Validation")
year_hist(test_df,  "Test")


# In[ ]:




