# import sys
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import torch
# from scipy import stats
# from autogluon.tabular import TabularDataset, TabularPredictor
# from quickda.explore_data import *
# from quickda.explore_numeric import *
# from quickda.explore_categoric import *
# from quickda.explore_numeric_categoric import *
# from quickda.clean_data import *
# from quickda.explore_time_series import *


# '''
# Train.csv
# - StudyInstanceUID: el ID del estudio
# - patient_overall: Indica si alguna de las vértebras está fracturada
# - C1: Si la vértebra C1 está fracturada.
# - C2: Si la vértebra C2 está fracturada.
# - C3: Si la vértebra C3 está fracturada.
# - C4: Si la vértebra C4 está fracturada.
# - C5: Si la vértebra C5 está fracturada.
# - C6: Si la vértebra C6 está fracturada.
# - C7: Si la vértebra C7 está fracturada.

# Test.csv
# - row_id: el identificador de la fila
# - StudyInstanceUID: el ID del estudio.
# - Predicción_tipo: cuál de las ocho columnas de destino necesita una predicción en esta fila.
# '''

# train = pd.read_csv('./train.csv', encoding='utf8')
# train.describe()

# """## Análisis exploratorio"""

# def analisis_exploratorio():
#   train.head(10)
#   explore(train)
#   train.info()

#   quan_vars = ['C1','C2','C3','C4','C5','C6','C7']

#   quan_df = train[quan_vars].replace('Ignorado', -1).fillna(-1)

#   # Ver si hay valores nulos "NULL"
#   sns.heatmap(train.isnull(), yticklabels = False, cbar = False, cmap="Blues")

#   train.isnull().sum()

#   train[['StudyInstanceUID', 'patient_overall', 'C1', 'C2', 'C3', 'C4','C5','C6','C7']].hist(bins = 30, figsize = (12, 12), color = '#7FCDD3')

#   for var in quan_vars:
#     serie = quan_df[quan_df[var] > 0][var]
#     display(serie.describe())
#     sns.displot(quan_df[var], kde=True)
#     print('\033[1m' + var + '\033[0m' + ': Kurtosis:', stats.kurtosis(serie), 'Skewness:', stats.skew(serie), '\n')

#   k = 10 #number of variables for heatmap
#   corrmat = quan_df.corr()
#   cm = np.corrcoef(corrmat.values.T)
#   sns.set(font_scale=1.25)
#   hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=quan_vars, xticklabels=quan_vars)
#   plt.show()

#   sns.pairplot(train)

# #Prediciendo patient_overall, que indica si alguna de las vértebras está fracturada
# df_outcome = train.groupby(by='patient_overall').mean()
# df_outcome

# # Utilizar la calidad pre-determinada "best_quality" y la métrica "accuracy"
# # Dividiendo los datos en un 80% para entrenamiento y 20% para pruebas

# from sklearn.model_selection import train_test_split
# X_entreno, X_prueba = train_test_split(train, test_size=0.2, random_state=0)

# X_entreno

# X_prueba

# predictor = TabularPredictor(label="patient_overall", 
#                              problem_type = 'regression', 
#                              eval_metric = 'r2').fit(train_data = X_entreno, time_limit = 200, presets = "best_quality")

# predictor.fit_summary()

# """Evaluando el rendimiento de los modelos entrenados, a través de la graficación del tablero de líderes "leaderboard", e indicar el mejor de los modelos"""

# predictor.leaderboard()

# test_data = TabularDataset('./train.csv')
# test_data

# testinInput = test_data.drop(columns=["patient_overall"])
# y_pred = predictor.predict(testinInput)
# y_pred

# predictor.leaderboard(test_data, silent=True)

# """Matriz de confusión"""

# y_test = test_data["patient_overall"] 
# perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)

# trueNeg = 0
# falsePos = 0
# falseNeg = 0
# truePos = 0

# for pred in range(len(y_pred)):
#   actual = y_test[pred]
#   suppo = round(y_pred[pred])
#   if suppo == actual:
#     if suppo == 0:
#       trueNeg += 1
#     else:
#       truePos += 1
#   else:
#     if suppo == 0:
#       falseNeg += 1
#     else:
#       falsePos += 1

# print('-----------------------------------------------------------------')
# print('\t\t\t|\tReal Positive \tReal Negative\t|')
# print('-----------------------------------------------------------------')
# print(f'Predicted Positive\t|\t{truePos}\t\t{falsePos}\t\t|')
# print(f'Predicted Negative\t|\t{falseNeg}\t\t{trueNeg}\t\t|')

# precision = truePos / (truePos + falsePos)
# recall = truePos / (truePos + falseNeg)
# score = (2 * precision * recall) / (precision + recall)

# print('Results:')
# print(f'\tPrecision: {precision}')
# print(f'\tRecall: {recall}')
# print(f'\tScore: {score}')

# """## Usamos el modelo de redes bayesianas"""

# # splitting X and y into training and testing sets
# from sklearn.model_selection import train_test_split

# X = train.drop(columns=["StudyInstanceUID"])
# y = train["patient_overall"]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# # training the model on training set
# from sklearn.naive_bayes import GaussianNB
# gnb = GaussianNB()
# gnb.fit(X_train, y_train)

# # making predictions on the testing set
# y_pred = gnb.predict(X_test)

# # comparing actual response values (y_test) with predicted response values (y_pred)
# from sklearn import metrics
# print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)
# print("Score; ", gnb.score(X_test, y_test))