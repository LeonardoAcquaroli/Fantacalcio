#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# data manipulation
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import re
import numpy as np
import pandas as pd
import warnings 
pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore')

# files
import os
import pathlib

# plotting
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.dates as mdates
import seaborn as sns

# web app
import streamlit as st
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

####################

# Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
# Clustering
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# Prediction
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
## models
from sklearn.linear_model import LinearRegression # ols
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor
## evaluation
from sklearn.model_selection import cross_val_score, KFold
from sklearn.base import BaseEstimator, RegressorMixin

# In[ ]:
st.title("Previsioni Fantacalcio 2023/24 ⚽")
st.write("Questa web app permette di scegliere i calciatori migliori per il 2023/24 sulla base di un <span style='color: blue;'>**algoritmo che predice la media del FantaVoto**</span> per la prossima stagione. 💻", unsafe_allow_html=True)
st.write("Inoltre divide i giocatori in due gruppi (Clusters): <span style='color: red;'>**Malus takers**</span> e <span style='color: green;'>**Bonus takers**</span>, ossia giocatori che tendono ad accumulare malus o bonus, informazione, ad esempio, molto utile per valutare i centrocampisti. ➕➖", unsafe_allow_html=True)
st.write("Infine, mette a disposizione uno  <span style='color: blue;'>**strumento di analisi grafica**</span>  delle prestazioni negli anni dei giocatori di Serie A. 📈", unsafe_allow_html=True)

# In[ ]:

def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Filtra la tabella")

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filtra la tabella in base a:", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            left.write("↳")
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"valori per {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Valori per {column}",
                    _min,
                    _max,
                    (_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Valori per {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Testo o regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].str.contains(user_text_input)]

    return df


# In[ ]:


# Set the path to the directory containing the Excel files
dir_path = r"C:\Users\leoac\OneDrive - Università degli Studi di Milano\Data science\Football\Fantacalcio\Dati Fantacalcio.it\Statistiche"
# Get a list of all the files in the directory
file_list = os.listdir(dir_path)
# Create a list of dfs
df_list = []
# Loop through the files and read each one
for file_name in file_list:
    if file_name.endswith('.xlsx'):  # Check if file is a .xlsx file
        file_path = os.path.join(dir_path, file_name)
        df = pd.read_excel(file_path, skiprows=1)
        df["Anno"] = datetime.strptime("20" + re.findall("[0-9]+", (pd.read_excel(file_path, nrows=1).columns[0]))[1], "%Y")
        df_list.append(df)
data = pd.concat(df_list,axis=0,ignore_index=True)

# Change the players' name into the same format
data.Nome = data.Nome.apply(lambda x: x.lower().title())

# Combine the two different names for games to vote
data.Pv = data.Pg.combine_first(data.Pv)
data.drop("Pg", axis = "columns", inplace=True)
# Combine the two different names for Fantamedia
data.Mf = data.Mf.combine_first(data.Fm)
data.drop("Fm", axis = "columns", inplace=True)

# Drop the specific role column (Rm)
data.drop("Rm", axis = "columns", inplace=True)

# Reorder and rename columns
data = data.reindex(columns=["Id","R","Nome","Squadra","Anno","Mv","Mf","Gf","Gs","Rp","Rc","R+","R-","Ass","Amm","Esp","Au","Pv"])
data.columns = ["Id","R","Nome","Squadra","Anno","Media voto","Media FantaVoto","Gol fatti","Gol subiti","Rigori parati","Rigori calciati","Rigori segnati","Rigori sbagliati","Assist","Ammonizioni","Espulsioni","Autogol","Partite a voto"]

# Add the quotes
Qdir_path = r"C:\Users\leoac\OneDrive - Università degli Studi di Milano\Data science\Football\Fantacalcio\Dati Fantacalcio.it\Quote"
Qfile_list = os.listdir(Qdir_path)
Qdf_list = []
# Loop through the files and read each one
c = 0
for Qfile_name in Qfile_list:
    if Qfile_name .endswith('.xlsx'):  # Check if file is a .xlsx file
        Qfile_path = os.path.join(Qdir_path, Qfile_name)
        Qdf = pd.read_excel(Qfile_path, skiprows=1)
        Qdf["Anno"] = datetime.strptime("20" + re.findall("[0-9]+", (pd.read_excel(Qfile_path, nrows=1).columns[0]))[1], "%Y")
        Qdf_list.append(Qdf)
quotes = pd.concat(Qdf_list,axis=0,ignore_index=True)

# Join quotes with data
withQuotes = data.merge(quotes, how="inner", on=['Nome', 'Anno'])
withQuotes.drop([Qcol for Qcol in withQuotes.columns if Qcol.endswith("_y")], axis=1, inplace=True) # drop duplicates columns from merging
# %%
withQuotes.columns = ['Id', 'Ruolo', 'Nome', 'Squadra', 'Anno', 'Media voto','Media FantaVoto', 'Gol fatti',
                      'Gol subiti', 'Rigori parati','Rigori calciati','Rigori segnati', 'Rigori sbagliati',
                      'Assist','Ammonizioni', 'Espulsioni', 'Autogol','Partite a voto', 'RM', 'Quota attuale',
                      'Quota iniziale', 'Differenza', 'Qt.A M', 'Qt.I M', 'Diff.M', 'FVM', 'FVM M'] # change columns names
withQuotes["FantaVoto/Quota iniziale"] = withQuotes["Media FantaVoto"] / withQuotes["Quota iniziale"]
withQuotes["Voto/Quota iniziale"] = withQuotes["Media voto"] / withQuotes["Quota iniziale"]

# Clean columns names
withQuotes = withQuotes.reindex(columns=['Ruolo', 'Nome', 'Squadra', 'Anno', 'Media voto',
       'Media FantaVoto','Quota iniziale',"FantaVoto/Quota iniziale","Voto/Quota iniziale",'Gol fatti', 'Gol subiti', 'Rigori parati',
       'Rigori calciati', 'Rigori segnati', 'Rigori sbagliati', 'Assist',
       'Ammonizioni', 'Espulsioni', 'Autogol', 'Partite a voto',
       'Quota attuale', 'Differenza tra quote','Id'])
# Crowd Differenza tra quote
withQuotes["Differenza tra quote"] = withQuotes["Quota iniziale"] - withQuotes["Quota attuale"]

# In[ ]:
# Model

@st.cache_resource
def load_xgb():
    xgb_fit = xgb.Booster()
    xgb_fit.load_model(r"C:\Users\leoac\OneDrive - Università degli Studi di Milano\Data science\Football\Fantacalcio\xgb.model") # big model
    return xgb_fit

# # Prediction for 2023/24
# #### Dataframe

# In[ ]:
# Remove goalkeepers
withQuotes_noGK = withQuotes[withQuotes["Ruolo"] != "P"]

# Create 2023 df
df_2023 = withQuotes_noGK[withQuotes_noGK.Anno == datetime(2023,1,1)] # 475 players

# Create new quotes df
newQuotes_path = r"C:\Users\leoac\OneDrive - Università degli Studi di Milano\Data science\Football\Fantacalcio\Dati Fantacalcio.it\Nuove quote\Quotazioni_Fantacalcio_Stagione_2023_24.xlsx"
newQuotes = pd.read_excel(newQuotes_path, skiprows=1)
newQuotes["Anno"] = datetime.strptime("20" + re.findall("[0-9]+", (pd.read_excel(newQuotes_path, nrows=1).columns[0]))[1], "%Y")
newQuotes.columns = ['Id', 'Ruolo', 'RM', 'Nome', 'Squadra', 'Quota attuale ', 'Quota iniziale', 'Differenza', 'Qt.A M',
        'Qt.I M', 'Diff.M', 'FVM', 'FVM M', 'Anno'] # change names of the variables of interest
newQuotes.drop(['RM','Qt.A M', 'Qt.I M', 'Diff.M', 'FVM', 'FVM M'], inplace=True, axis=1) # drop useless columns
newQuotes.rename(columns={"Quota iniziale": "Quota iniziale (t+1)"}, inplace=True)
newQuotes = newQuotes[["Id", "Quota iniziale (t+1)"]] # keep only the Id for the merge and the Quota iniziale (t+1)
################# 508 players

# Merge 2023 df and new quotes
new_prediction_df = pd.merge(df_2023, newQuotes, how = "inner", on = "Id")
teams = new_prediction_df.Squadra
roles = new_prediction_df.Ruolo
names = new_prediction_df.Nome
################# 336 players

# keep only the needed columns for predictions (numerical ones and Ruolo)
new_prediction_df = new_prediction_df[['Ruolo', 'Media FantaVoto', 'FantaVoto/Quota iniziale', 'Gol fatti', 'Rigori calciati','Rigori segnati', 'Rigori sbagliati', 'Assist', 'Ammonizioni','Espulsioni', 'Autogol', 'Partite a voto', 'Quota attuale','Differenza tra quote', 'Quota iniziale (t+1)']]
### clustering_df is created keeping only the pitch-performance metrics
clustering_df = new_prediction_df[['Gol fatti', 'Rigori calciati','Rigori segnati', 'Rigori sbagliati', 'Assist', 'Ammonizioni','Espulsioni', 'Autogol', 'Partite a voto']]

### one-hot encoding for Ruolo
dummies = pd.get_dummies(new_prediction_df.Ruolo).iloc[:,0:2] # dummies for Attaccante and Centrocampista to avoid dummy trap
new_prediction_df.drop("Ruolo", axis=1, inplace=True)
new_prediction_df = pd.concat([dummies, new_prediction_df], axis=1)

### cast as float
new_prediction_df = new_prediction_df.astype('float64')

## feature augmentation
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False) # initiate polinomial transformer
X = new_prediction_df.loc[:, ~new_prediction_df.columns.isin(["Media FantaVoto (t+1)"])]
X_quad = poly.fit_transform(X) # create all the quadratic interactions
new_prediction_df = pd.DataFrame(X_quad, columns=poly.get_feature_names_out(input_features=X.columns)) # Create a new DataFrame with the quadratic interaction features

# #### Prediction

# In[ ]:
dnew = xgb.DMatrix(new_prediction_df)
new_pred_values = load_xgb().predict(dnew)
plt.hist(new_pred_values)
new_pred_names = pd.concat([names, pd.Series(new_pred_values, name='Media FantaVoto 23/24 (previsione)'), new_prediction_df["Quota iniziale (t+1)"]], axis=1)
new_pred_names.rename(columns={"Quota iniziale (t+1)": "Quota"}, inplace=True)
new_pred_names["Rapporto FantaVoto/Quota"] = new_pred_names['Media FantaVoto 23/24 (previsione)']/new_pred_names["Quota"]
new_pred_names = pd.concat([new_pred_names, teams, roles], axis=1)
# st.dataframe(new_pred_names)
new_pred_names = new_pred_names.reindex(columns=["Nome", "Squadra", "Media FantaVoto 23/24 (previsione)", "Quota", "Rapporto FantaVoto/Quota", "Ruolo"])
# st.dataframe(new_pred_names)
# In[ ]:
new_pred_names.sort_values('Media FantaVoto 23/24 (previsione)', ascending=False).head(20)
# new_pred_names.sort_values('Media FantaVoto 23/24 (previsione)', ascending=False).to_csv(r"C:\Users\leoac\OneDrive - Università degli Studi di Milano\Data science\Football\Fantacalcio\Previsioni 23-24.csv")

# # CLUSTERING
# In[ ]:
# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(clustering_df)
# In[ ]:


# Perform PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(data_scaled)

# In[ ]:


# Perform K-means clustering with the best K value
kmeans = KMeans(n_clusters=2, random_state=42)
cluster_labels = kmeans.fit_predict(pca_result)

new_pred_names = pd.concat((new_pred_names, pd.Series(cluster_labels, name="Cluster")), axis=1) # execute this line only once
new_pred_names.Cluster = new_pred_names.Cluster.map({1: "Malus taker", 0: "Bonus taker"})
st.write("## Tabella")
st.dataframe(filter_dataframe(new_pred_names))


# In[ ]:
# Spider chart

# new_pred_names.groupby("Cluster").mean()

# # Graphical analysis
st.write("## Analisi grafica")
# In[ ]:


# Create variables user selection
nome1 = st.selectbox("Giocatore 1", new_pred_names.Nome.unique(), index=8) # (use only the names of players for 2023/24) # start with Theo Hernaned [9]
nome2 = st.selectbox("Giocatore 2", new_pred_names.Nome.unique(), index=1) # (use only the names of players for 2023/24) # start with Dimarco [1]
metrica = st.selectbox("Metrica", withQuotes.columns[4:withQuotes.shape[1]-2]) # from Media voto to Quota attuale (left out Differenza tra quote)

# Mini timeseries
player1 = withQuotes[withQuotes.Nome == nome1][["Anno", metrica]]
player2 = withQuotes[withQuotes.Nome == nome2][["Anno", metrica]]

# Merged df of both players metrics
years = sorted(set(player1['Anno']) | set(player2['Anno'])) # Determine the unique years from both dataframes
merged = pd.DataFrame({'Anno': years}) # Create a new dataframe with the unique years
merged = merged.merge(player1, on='Anno', how='left') # Merge player1 data
merged = merged.merge(player2, on='Anno', how='left') # Merge player2 data
merged.columns = ['Anno', nome1, nome2] # Rename the columns

# Set "Anno" as index
merged = merged.set_index('Anno')

# Plot
## Create a figure and axes
fig, ax = plt.subplots()
ax.plot(merged.index, merged[nome1], label=nome1, marker = '.')
ax.plot(merged.index, merged[nome2], label=nome2, marker = '.')
date_form = mdates.DateFormatter("%Y") # Set x-axis tick formatter to display only the year
ax.xaxis.set_major_formatter(date_form)
ax.set_xlabel('Anno')
ax.set_ylabel(metrica)
#ax.set_ylim(bottom=min(withQuotes[metrica]), top=max(withQuotes[metrica]))

ax.legend()
fig.tight_layout()
## Display the chart using Streamlit
st.pyplot(fig)

