import streamlit as st
from PIL import Image 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from scipy import stats
from autogluon.tabular import TabularDataset, TabularPredictor
from quickda.explore_data import *
from quickda.explore_numeric import *
from quickda.explore_categoric import *
from quickda.explore_numeric_categoric import *
from quickda.clean_data import *
from quickda.explore_time_series import *

st.set_option('deprecation.showPyplotGlobalUse', False)
st.title('Detección de fracturas de las vértebras cervicales en radiografías')

DATA_TRAIN = ('./train.csv')

st.markdown(" **Situación problemática** En Estados Unidos cada año se da una alta tasa de fracturas de columna lo que resulta en más de 15 mil lesiones de la médula espinal; además el sitio más común de fractura se da en la columna cervical. La mayoría de fracturas se da en adultos mayores y suelen ser las más difíciles de visualizar debido a una enfermedad degenerativa y una superpuesta. Detectar dichas fracturas hoy en día se logra mediante tomografías computarizadas, en lugar de rayos X como se realizaba hace algunos años. El encontrar rápidamente la ubicación de cualquier fractura vertebral es esencial para prevenir el deterioro neurológico y el trauma por parálisis. ")
st.info("El National Emergency X-Radiography Utilization Study (NEXUS) y Canadian Cervical Rules (CCR) son criterios clínicos establecidos correctamente para la exclusión de lesiones clínicamente significativas de la columna cervical con una sensibilidad cercana al 100 %. ")

if st.button("Vértebras Cervicales"):
    img=Image.open('images/vertebras-cerv-1.jpeg')
    st.image(img,width=700, caption="Vértebras cervicales")
    st.markdown("Dentro de este Dataset se puede identificar fracturas en tomografías computarizadas de la columna cervical (cuello) tanto a nivel de una sola vértebra como de todo el paciente. La detección y determinación rápidas de la ubicación de cualquier fractura vertebral es esencial para prevenir el deterioro neurológico y la parálisis después de un traumatismo.")


st.sidebar.markdown("## Side Panel" 
	)
st.sidebar.markdown("Use this panel to explore the dataset and create own viz.")


@st.cache(persist=True, show_spinner=True)
# Load  the Data 
def load_data(nrows):
	#Parse date and Time 
    df = pd.read_csv(DATA_TRAIN, nrows = nrows)
    lowercase = lambda x:str(x).lower()
    df.rename(lowercase, axis='columns',inplace=True)
    return df

st.info(
    " ***Variables del dataset***"
    )

data_load_state = st.text('Cargando la data...')
df = load_data(10000)
data_load_state.text("¡Datos cargados!")

if st.checkbox('Mostrar los datos de entrenamiento'):
    st.subheader('Datos de entrenamiento')
    st.write(df)

st.title('Quick  Explore')
st.sidebar.subheader(' Quick  Explore')
st.markdown("Tick the box on the side panel to explore the dataset.")
if st.sidebar.checkbox('Basic info'):
    if st.sidebar.checkbox('Dataset Quick Look'):
        st.subheader('Dataset Quick Look:')
        st.write(df.head())
    if st.sidebar.checkbox("Show Columns"):
        st.subheader('Show Columns List')
        all_columns = df.columns.to_list()
        st.write(all_columns)
    # if st.sidebar.checkbox('Column Names'):
    #     st.subheader('Column Names')
    #     st.write(df.columns())
    if st.sidebar.checkbox('Statistical Description'):
        st.subheader('Statistical Data Descripition')
        st.write(df.describe())
    if st.sidebar.checkbox('Missing Values?'):
        st.subheader('Missing values')
        st.write(df.isnull().sum())

st.title('Create Own Visualization')
st.markdown("Tick the box on the side panel to create your own Visualization.")
st.sidebar.subheader('Create Own Visualization')
if st.sidebar.checkbox('Graphics'):
    if st.sidebar.checkbox('Count Plot'):
        st.subheader('Count Plot')
        st.info("If error, please adjust column name on side panel.")
        column_count_plot = st.sidebar.selectbox("Choose a column to plot count. Try Selecting Sex ",df.columns)
        hue_opt = st.sidebar.selectbox("Optional categorical variables (countplot hue). Try Selecting Species ",df.columns.insert(0,None))
        # if st.checkbox('Plot Countplot'):
        fig = sns.countplot(x=column_count_plot,data=df,hue=hue_opt)
        st.pyplot()
            
            
    if st.sidebar.checkbox('Histogram | Distplot'):
        st.subheader('Histogram | Distplot')
        st.info("If error, please adjust column name on side panel.")
        # if st.checkbox('Dist plot'):
        column_dist_plot = st.sidebar.selectbox("Optional categorical variables (countplot hue). Try Selecting Body Mass",df.columns)
        fig = sns.distplot(df[column_dist_plot])
            # fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
        st.pyplot()
            
            
    if st.sidebar.checkbox('Heatmap'):
        st.subheader('HeatMap')
        fig = sns.heatmap(df.corr(),annot=True, annot_kws={"size": 9}, linewidths=1.5)
        st.pyplot()

    if st.sidebar.checkbox('Boxplot'):
        st.subheader('Boxplot')
        st.info("If error, please adjust column name on side panel.")
        column_box_plot_X = st.sidebar.selectbox("X (Choose a column). Try Selecting island:",df.columns.insert(0,None))
        column_box_plot_Y = st.sidebar.selectbox("Y (Choose a column - only numerical). Try Selecting Body Mass",df.columns)
        hue_box_opt = st.sidebar.selectbox("Optional categorical variables (boxplot hue)",df.columns.insert(0,None))
        # if st.checkbox('Plot Boxplot'):
        fig = sns.boxplot(x=column_box_plot_X, y=column_box_plot_Y,data=df,palette="Set3")
        st.pyplot()
    if st.sidebar.checkbox('Pairplot'):
        st.subheader('Pairplot')
        hue_pp_opt = st.sidebar.selectbox("Optional categorical variables (pairplot hue)",df.columns.insert(0,None))
        st.info("This action may take a while.")
        fig = sns.pairplot(df,palette="coolwarm")
        st.pyplot()



# # for the heatmap
# df_col = pd.concat([df], axis=1)
#     df5.columns = ['month', 'price_kcl', 'change_kcl']
#     df6.columns = ['month_fosfat', 'price_fosfat', 'change_fosfat']
#     df7.columns = ['month_bb', 'price_bara', 'change_bb']
#     df8.columns = ['month_urea', 'price_urea', 'change_urea']
#     df9.columns = ['month_npk', 'price_npk', 'change_npk']
#     df_col = pd.concat([df5, df6,df7,df8,df9], axis=1)
#     df5.columns = ['month', 'price_kcl', 'change_kcl']
#     df6.columns = ['month_fosfat', 'price_fosfat', 'change_fosfat']
#     df7.columns = ['month_bb', 'price_bara', 'change_bb']
#     df8.columns = ['month_urea', 'price_urea', 'change_urea']
#     df9.columns = ['month_npk', 'price_npk', 'change_npk']
#     df_col = df_col.set_index('month')
#     df_corr = df_col.corr()
#     st.write(df_corr)
#     plt.matshow(df_col.corr())
# fig, ax = plt.subplots()
# sns.heatmap(df_col.corr(), ax=ax)
# st.write(fig)

# if st.button("Viz by Tableau Community"):
#     img=Image.open('images/D1.png')
#     st.subheader("Viz by [Scott Renfree-Tuck](https://public.tableau.com/profile/scott.renfree.tuck#!/vizhome/MakeoverMondayW282020-PalmersPenguins/Dashboard2)")
#     st.image(img,width=950)
#     st.subheader("Viz by  [Agata Ketterick](https://public.tableau.com/profile/agata1619#!/vizhome/PalmerPenguinsMakeoverMonday2020_28/Penguins)")
#     img2=Image.open('images/D2.png')
#     st.image(img2,width=950)
#     img3=Image.open('images/D3.png')
#     st.subheader("Viz by [Swati Dave](https://public.tableau.com/profile/swati.dave#!/vizhome/PenguinParadox/PenguinStory)")
#     st.image(img3,width=950)



st.sidebar.info("Self Exploratory Visualization on palmerpenguins - Brought To you By [Mala Deep](https://github.com/maladeep)  ")

