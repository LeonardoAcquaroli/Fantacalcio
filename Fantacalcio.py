#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# data manipulation
import pandas as pd
import warnings 
pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore')

# plotting
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# web app
import streamlit as st
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

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
new_pred_names = pd.read_csv(r"https://raw.githubusercontent.com/LeonardoAcquaroli/Fantacalcio/main/Dataframes/Previsioni_23-24.csv", sep=";")

# %%
st.write("## Tabella")
st.dataframe(filter_dataframe(new_pred_names))


# In[ ]:
# Spider chart

# new_pred_names.groupby("Cluster").mean()

# # Graphical analysis
st.write("## Analisi grafica")
# In[ ]:
# load df
withQuotes = pd.read_csv(r"https://raw.githubusercontent.com/LeonardoAcquaroli/Fantacalcio/main/Dataframes/Dataframe%20con%20quote.csv", sep=";")

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

