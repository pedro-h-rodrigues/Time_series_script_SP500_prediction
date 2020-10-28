# O objetivo desse script é prever o valor de fechamento do índice SP500 da bolsa
# americana utilizando um modelo de regressão linear

# Carregando o dataset com dados e dividindo-o entra dados
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

sphist = pd.read_csv('sphist.csv')
sphist['Date'] = pd.to_datetime(sphist['Date'])
sphist = sphist.sort_values('Date')

# Calculando o preco médio dos ultimos 5 e últimos 365 dias com iterrows
    
for i, r in sphist.iterrows():
    if r['Date'] < datetime(1951,1,3): #Descartando linhas para as quais não há 365 dias de histórico
        sphist.loc[i,'avg_past365'] = np.nan
        sphist.loc[i,'avg_past5'] = np.nan
    else:
        avg_365 = np.mean(sphist.loc[i+365:i+1,'Close'])
        sphist.loc[i,'avg_past365'] = avg_365
        
        avg_5 = np.mean(sphist.loc[i+5:i+1,'Close'])
        sphist.loc[i,'avg_past5'] = avg_5

# Acima nos utilizamos iterrows para iterar linha a linha 
# Refazendo o processo utilizando o método rolling ao invés de iterar linha a linha
   
sphist['avg_past5_roll'] = sphist['Close'].rolling(5).mean()
sphist['avg_past365_roll'] = sphist['Close'].rolling(365).mean() 

# Deslocando uma linha para baixo pois o rolling usa a própria linha
sphist['avg_past5_roll'] = sphist['avg_past5_roll'].shift(1)          
sphist['avg_past365_roll'] = sphist['avg_past365_roll'].shift(1)

'''Note que o resultado das colunas usando o método .rolling é o mesmo que
 utilizando iterrows. De agora em diante vamos utilizar apenas o rolling.
 Vamos também apagar as colunas que fizemos com o método iterrows'''

sphist = sphist.drop(columns = ['avg_past5','avg_past365'])
   
# Calculando os desvios padrão dos últimos 5 e dos últimos 365 dias
sphist['std_past5'] = sphist['Close'].rolling(5).std()           
sphist['std_past5'] = sphist['std_past5'].shift(1)

sphist['std_past365'] = sphist['Close'].rolling(365).std()           
sphist['std_past365'] = sphist['std_past365'].shift(1)

#Calculando os ratios entre os últimos 5 dias e os últimos 365 dias

sphist['ratio_avg'] = sphist['avg_past5_roll']/sphist['avg_past365_roll']
sphist['ratio_std'] = sphist['std_past5']/sphist['std_past365']

# Elimando as colunas com valores vazios
sphist = sphist.dropna(axis = 0)


# Treinando um modelo de regressão linear com base nos ratios calculados

def train_test_plot (dataset, features_list):
    # Dividindo o dataset entre dataset de treino (antes de 2013) e de teste (após 2013)
    cut_date = datetime(2013,1,1)
    train_set = dataset[dataset['Date'] < cut_date]
    test_set = dataset[dataset['Date'] > cut_date]    

    #treinando o modelo
    model = LinearRegression()
    model.fit(train_set[features_list],train_set[['Close']])
    
    #prevendo e calculando o root mean squared error do modelo
    prediction = model.predict(test_set[features_list])
    rmse = mean_squared_error(test_set['Close'],prediction)**0.5
          
    #plotando o valor previsto
    fig1 = plt.figure(figsize = (18,10))
    ax1 = fig1.add_subplot(1,1,1)
    ax1.plot(test_set['Date'],test_set['Close'],color = 'blue',label = "Actual closing value")
    ax1.plot(test_set['Date'],prediction,color = 'red', label = "Predicted closing value")
    title = "Actual and predicted values of SP500 - RMSE = {}".format(round(rmse,2))
    ax1.set_title(title)
    ax1.legend(loc = 'upper left')
    plt.show()

train_test_plot(sphist, ['avg_past5_roll','std_past5','ratio_avg','ratio_std'])