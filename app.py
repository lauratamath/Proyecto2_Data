from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
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

# Title
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title('Detección de fracturas de las vértebras cervicales en radiografías')

# Carga de datos
DATA_TRAIN = ('./train.csv')
@st.cache(persist=True, show_spinner=True)
# Load  the Data 
def load_data(nrows):
	#Parse date and Time 
    df = pd.read_csv(DATA_TRAIN, nrows = nrows)
    lowercase = lambda x:str(x).lower()
    df.rename(lowercase, axis='columns',inplace=True)
    return df

# Descripción del problema
st.markdown(" **Situación problemática** En Estados Unidos cada año se da una alta tasa de fracturas de columna lo que resulta en más de 15 mil lesiones de la médula espinal; además el sitio más común de fractura se da en la columna cervical. La mayoría de fracturas se da en adultos mayores y suelen ser las más difíciles de visualizar debido a una enfermedad degenerativa y una superpuesta. Detectar dichas fracturas hoy en día se logra mediante tomografías computarizadas, en lugar de rayos X como se realizaba hace algunos años. El encontrar rápidamente la ubicación de cualquier fractura vertebral es esencial para prevenir el deterioro neurológico y el trauma por parálisis. ")
st.markdown("El National Emergency X-Radiography Utilization Study (NEXUS) y Canadian Cervical Rules (CCR) son criterios clínicos establecidos correctamente para la exclusión de lesiones clínicamente significativas de la columna cervical con una sensibilidad cercana al 100 %. ")
if st.button("Vértebras Cervicales"):
    img=Image.open('images/vertebras-cerv-1.jpeg')
    st.image(img,width=700, caption="Vértebras cervicales")
    st.markdown("Dentro de este Dataset se puede identificar fracturas en tomografías computarizadas de la columna cervical (cuello) tanto a nivel de una sola vértebra como de todo el paciente. La detección y determinación rápidas de la ubicación de cualquier fractura vertebral es esencial para prevenir el deterioro neurológico y la parálisis después de un traumatismo.")

# side panel para seleccionar el tipo de análisis
st.sidebar.markdown("## Side Panel")
st.sidebar.markdown("Use este panel para conocer sobre el dataset, análisis exploratorio y resultados obtenidos. Así también, puede modificar los parámetros para explorar los modelos usados.")

# mostramos el dataset crudo
st.info(" ***Variables del dataset***")
data_load_state = st.text('Cargando la data...')
df = load_data(10000)
data_load_state.text("¡Datos cargados!")

if st.checkbox('Apache aquí para mostrar los datos de entrenamiento'):
    st.subheader('Datos de entrenamiento')
    st.write(df)

# ------> Exploración de datos
st.title("Exploración de datos")
st.sidebar.subheader('Explore el dataset')
st.markdown("Apache en el ***sidebar*** para conocer más sobre el dataset")

# exploramos el dataset
if st.sidebar.checkbox('Información del dataset'):
    # Mostramos cómo viene el dataset
    if st.sidebar.checkbox('Ver las primeras 5 filas'):
        st.subheader('Vista previa del dataset:')
        st.write(df.head(5))

    # Mostramos las columnas del dataset
    if st.sidebar.checkbox("Columnas del dataset"):
        st.subheader('Variables del dataset')
        st.markdown('- StudyInstanceUID: el ID del estudio')
        st.markdown('- patient_overall: Indica si alguna de las vértebras está fracturada')
        st.markdown('- C1: Si la vértebra C1 está fracturada.')
        st.markdown('- C2: Si la vértebra C2 está fracturada.')
        st.markdown('- C3: Si la vértebra C3 está fracturada.')
        st.markdown('- C4: Si la vértebra C4 está fracturada.')
        st.markdown('- C5: Si la vértebra C5 está fracturada.')
        st.markdown('- C6: Si la vértebra C6 está fracturada.')
        st.markdown('- C7: Si la vértebra C7 está fracturada.')
        all_columns = df.columns.to_list()
        st.write(all_columns)
        st.info('Total Columns: {}'.format(len(all_columns)))
    # Mostramos el tamaño del dataset
    if st.sidebar.checkbox("Ver el tamaño del dataset"):
        st.subheader('Tamaño del dataset')
        st.write(df.shape)
    # Mostrar estadísticas del dataset
    if st.sidebar.checkbox("Mostrar estadísticas"):
        st.subheader('Estadísticas del dataset')
        st.write(df.describe())

# ------> Análisis exploratorio
st.title('Análisis Exploratorio de Datos')
st.sidebar.subheader('Información del análisis exploratorio')
st.markdown("Apache en el ***sidebar*** para ver el análisis exploratorio de datos")

# análisis exploratorio
if st.sidebar.checkbox('Análisis Exploratorio de Datos'):
    # Descripción de las variables
    if st.sidebar.checkbox('Descripción de las variables'):
        st.subheader('Descripción de las variables')
        st.write(df.describe())

    # Mostramos el tipo de datos
    if st.sidebar.checkbox("Ver tipos de datos"):
        st.subheader('Tipos de datos')
        st.write(df.dtypes)

    # Mostramos la cantidad de datos nulos
    if st.sidebar.checkbox("Ver datos nulos"):
        st.subheader('Datos nulos')
        st.write(df.isnull().sum())
        st.subheader('Gráfico de datos nulos')
        st.write(sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis'))
        st.pyplot()

    # Mostramos la distribución de los datos
    if st.sidebar.checkbox('Ver distribución de los datos'):
        st.subheader('Distribución de los datos por variable')
        st.info('Seleccione una variable de vértebra en X y la variable patient_overall en Y')
        all_columns = df.columns.to_list()
        primary_col = st.selectbox('Seleccione la variable', all_columns)
        selected_columns = st.multiselect('Seleccione las variables', all_columns)
        if st.button('Generar gráfico'):
            st.success('Generando gráfico de distribución de datos por variable')
            if selected_columns:
                vc_plot = df.groupby(primary_col)[selected_columns].count()
            else:
                vc_plot = df[primary_col].value_counts()
            st.write(vc_plot.plot(kind='bar'))
            st.pyplot()

    # Mostramos kurtosis y skewness
    if st.sidebar.checkbox('Ver kurtosis y skewness'):
        st.subheader('Kurtosis')
        st.write(df.kurtosis())
        # graficamos kurtosis
        st.subheader('Gráfica de kurtosis')
        st.write(sns.distplot(df.kurtosis()))
        st.pyplot()

        st.subheader('Skewness')
        st.write(df.skew())
        # graficamos skewness
        st.subheader('Gráfica de skewness')
        st.write(sns.distplot(df.skew()))
        st.pyplot()

    # Mostramos la correlación entre las variables
    if st.sidebar.checkbox('Ver correlación entre las variables'):
        st.subheader('Correlación entre las variables')
        st.write(df.corr())
        st.subheader('Gráfica de correlación')
        st.write(sns.heatmap(df.corr(), annot=True))
        st.pyplot()
        st.subheader('Gráfica de correlación con pairplot')
        st.write(sns.pairplot(df))
        st.pyplot()

# ------> Modelado
st.title('Modelado')
st.sidebar.subheader('Información del modelado')
st.markdown("Apache en el ***sidebar*** para ver el modelado")

# Entrenamiento y modelado
if st.sidebar.checkbox('Modelado'):
    data_load_state = st.text('Entrenando el modelo...')
    df = load_data(10000)
    data_load_state.text("¡Modelo entrenado!")
    
    # Dividimos el dataset en train y test
    st.subheader('Dividir el dataset en train y test')  
    st.info('Dividiendo los datos en un 80% para entrenamiento y 20% para pruebas')      

    # Dividimos el dataset en train y test
    train = pd.read_csv('train.csv')
    X_train, X_test = train_test_split(train, test_size=0.2, random_state=0)
    # Mostramos los datos de X train
    st.subheader('Datos de X train')
    st.write(X_train)
    # Mostramos los datos de X test
    st.subheader('Datos de X test')
    st.write(X_test)

    # Usamos Predictor de TabularPredictor
    predictor = TabularPredictor(label="patient_overall", 
                        problem_type = 'regression', 
                        eval_metric = 'r2').fit(train_data = X_train, time_limit = 200, presets = "best_quality")
    # Mostramos los resultados
    st.subheader('Resultados del modelado')
    predictor.fit_summary()
    predictor.leaderboard()

    test_data = TabularDataset('train.csv')
    # shot the test data
    st.subheader('Datos de test')
    st.write(test_data.head())

    testinInput = test_data.drop(columns=['patient_overall'])
    y_pred = predictor.predict(testinInput)
    st.write('y_pred: ', y_pred.shape)
    st.write(y_pred)

    predictor.leaderboard(test_data, silent=True)
    y_test = test_data['patient_overall']
    perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)

    # Primeros resultados
    if st.sidebar.checkbox('Matriz de confusión'):
        st.subheader('Matriz de confusión')
        st.write(sns.heatmap(confusion_matrix(y_test, y_pred), annot=True))
        st.pyplot()

    # Segundos resultados
    if st.sidebar.checkbox('Errores de tipo I y II'):

        trueNeg = 0
        falsePos = 0
        falseNeg = 0
        truePos = 0

        for pred in range(len(y_pred)):
            actual = y_test[pred]
            suppo = round(y_pred[pred])
            if suppo == actual:
                if suppo == 0:
                    trueNeg += 1    
                else:
                    truePos += 1
            else:
                if suppo == 0:
                    falseNeg += 1
                else:
                    falsePos += 1
        st.write('Real positive -- Real negative: ')
        st.write('Predicted positive',truePos, falsePos)
        st.write('Predicted negative',falseNeg, trueNeg)

        precision = truePos / (truePos + falsePos)
        recall = truePos / (truePos + falseNeg)
        f1 = 2 * (precision * recall) / (precision + recall)    
        st.subheader('Results: ')
        st.write('Precision: ', precision)
        st.write('Recall: ', recall)
        st.write('Score: ', f1)

    # Terceros resultados: bayesian optimization
    if st.sidebar.checkbox('Redes bayesianas'):
        X = DATA_TRAIN.drop(columns=['StudyInstanceUID'])
        y = DATA_TRAIN['patient_overall']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # importing the required libraries
        from sklearn.naive_bayes import GaussianNB
        gnb = GaussianNB()
        # fitting the model
        gnb.fit(X_train, y_train)

        # making predictions on the testing set
        y_pred = gnb.predict(X_test)

        # comparing actual response values (y_test) with predicted response values (y_pred)
        from sklearn import metrics
        st.write("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)
        st.write("Gaussian Naive Bayes model precision(in %):", metrics.precision_score(y_test, y_pred, average='weighted')*100)
        st.write("Gaussian Naive Bayes model recall(in %):", metrics.recall_score(y_test, y_pred, average='weighted')*100)
        st.write("Gaussian Naive Bayes model f1-score(in %):", metrics.f1_score(y_test, y_pred, average='weighted')*100)

        st.sidebar.info("Data Science - Proyecto 2")

    # Cuartos resultados: ...
    if st.sidebar.checkbox('Último modelo'):
        st.subheader('...')
        st.write('Aqui va el modelo faltante')

# ------> Hallazgos y conclusiones
st.title('Hallazgos y conclusiones')
st.sidebar.subheader('Información de hallazgos y conclusiones')
st.markdown("Apache en el ***sidebar*** para ver los hallazgos y conclusiones")

# Hallazgos y conclusiones
if st.sidebar.checkbox('Hallazgos y conclusiones'):
    st.subheader('Hallazgos y conclusiones')
    st.write('Aqui van los hallazgos y conclusiones')

# ------> Referencias
st.title('Referencias')
st.sidebar.subheader('Información de referencias')
st.markdown("Apache en el ***sidebar*** para ver las referencias")

# Referencias
if st.sidebar.checkbox('Referencias'):
    st.subheader('Referencias')
    st.write('Fernández Galán S. Redes Bayesianas temporales: aplicaciones médicas e industriales. Tesis doctoral. Universidad Nacional de Educación a Distancia [Internet]. Madrid 2002. Disponible en: http://www.cisiad.uned.es/tesis/tesis-seve.pdf.') 
    st.write('Kim, Jung Hee. Department of Internal Medicine, Seoul National University College of Medicine, 101 Daehak-ro, Jongno-gu, Seoul 03080, Korea Tel: +82-2-2072-4839, Fax: +82-2-2072-7246 https://e-enm.org/upload/pdf/enm-2022-1461.pdf http://scielo.sld.cu/scielo.php?pid=s1561-31942018000300014&script=sci_arttext&tlng=pt ')
    st.write('Lopera, Johan. (25/05/2021). Trauma de columna cervical: aproximación por imágenes. Perlas clínicas:  https://www.perlasclinicas.medicinaudea.co/salud-del-adulto-y-el-anciano/trauma-de-columna-cervical-aproximacion-por-imagenes ')
    st.write('Lugo S, Maldonado G, Murata Ch. Inteligencia artificial para asistir el diagnóstico clínico en medicina. Revista Alergia México 2014 [Internet];61:110-120. Available from: http://revistaalergia.mx/ojs/index.php/ram/article/download/33/46. ')
    st.write('Saha, S. (2018) A comprehensive guide to convolutional neural networks  https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53 ')