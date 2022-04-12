
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from talib import MA_Type
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
import pandas as pd
from sklearn.svm import SVC
import keras
from pandas_datareader import data

ibov =data.DataReader('^BVSP','yahoo','2010-01-01','2022-03-09')
dolar = data.DataReader('USDBRL=X','yahoo','2010-01-01','2022-03-09')
cdi = pd.read_csv('CDI.csv', sep=',')
sp500 = pd.read_csv('SP500.csv', sep=',')

ibov.dtypes
cdi.dtypes
dolar.dtypes
sp500.dtypes

ibov.drop(['Adj Close'], axis=1, inplace=True)
ibov.reset_index(inplace=True)
ibov.columns = ['Data','ibov_maxima','ibov_minima','ibov_abertura','ibov_ultimo','vol']


sp500['Último']=sp500['Último'].str.replace('.','').str.replace(',','.').astype(float)
sp500.drop(['Abertura','Máxima','Mínima','Var%','Vol.'], axis=1, inplace = True)
sp500['Data'] = pd.to_datetime(sp500.Data, dayfirst = True)
sp500.columns = ['Data','sp500_ultimo']
sp500 = sp500.sort_values(by = ['Data'])

cdi['Último']=cdi['Último'].str.replace(',','.').astype(float)
cdi.drop(['Abertura','Máxima','Mínima','Var%','Vol.'], axis=1, inplace = True)
cdi['Data'] = pd.to_datetime(cdi.Data, dayfirst = True)
cdi.columns = ['Data','cdi_ultimo']
cdi = cdi.sort_values(by = ['Data'])

dolar.drop(['High','Low','Open','Volume', 'Adj Close'], axis=1, inplace = True)
dolar.reset_index(inplace=True)
dolar.columns = ['Data','dolar_ultimo']

ibov.describe()
ibov.dtypes
ibov.info()
ibov.isnull().sum()

sp500.describe()
sp500.dtypes
sp500.info()
sp500.isnull().sum()

dolar.describe()
dolar.dtypes
dolar.info()
dolar.isnull().sum()

cdi.describe()
cdi.dtypes
cdi.info()
cdi.isnull().sum()

ibov['vol'].replace(0, np.mean(ibov['vol']), inplace=True)

base = pd.merge(ibov, pd.merge(dolar, pd.merge(sp500, cdi, how = 'left', 
                on = 'Data'), how = 'left', on = 'Data'),
                how = 'left',on = 'Data')

i = 0
for i in range(0, len(base)):
    if (np.isnan(base['sp500_ultimo'][i]) or 
    np.isnan(base['cdi_ultimo'][i]) or np.isnan(base['dolar_ultimo'][i])):
       base['sp500_ultimo'][i] = base['sp500_ultimo'][i-1]
       
       base['dolar_ultimo'][i] = base['dolar_ultimo'][i-1]
      
       base['cdi_ultimo'][i] = base['cdi_ultimo'][i-1]
       
    i+=1
i = 0

#Criação dos Indicadores Tecnicos
import talib as ta
#Medias Moveis Bovespa
base['mm7']=ta.MA(base['ibov_ultimo'], timeperiod = 7)
base.mm7 = base.mm7.shift(periods=1)
base['mm21']=ta.MA(base['ibov_ultimo'], timeperiod = 21)
base.mm21 = base.mm21.shift(periods=1)
#Media móvel - periodo 14 - dolar
base['dmm14']=ta.MA(base['dolar_ultimo'], timeperiod=14)
base.dmm14 = base.dmm14.shift(periods=1)
#Media móvel - periodo 14 - sp500
base['sp14']=ta.MA(base['sp500_ultimo'], timeperiod=14)
base.sp14 = base.sp14.shift(periods=1)
#Media móvel - periodo 14 - cdi
base['cdi14']=ta.MA(base['cdi_ultimo'], timeperiod=14)
base.cdi14 = base.cdi14.shift(periods=1)   
#IFR 14
base['ifr14'] = ta.RSI(base['ibov_ultimo'], timeperiod=14)               
#IFR 7
base['ifr7'] = ta.RSI(base['ibov_ultimo'], timeperiod=7)  
#ROC 
base['ROC'] = ta.ROC(base['ibov_ultimo'], timeperiod=12)
#OBV
base['OBV'] = ta.OBV(base['ibov_ultimo'], ibov['vol'])

      
base.isnull().sum()
base.drop(base.index[range(0, 21)], inplace=True)
base.set_index('Data', inplace=True)

base[['ibov_maxima', 'ibov_minima','ibov_abertura','ibov_ultimo', 'vol']].describe()
base[['dolar_ultimo', 'sp500_ultimo','cdi_ultimo']].describe()
base.dtypes
base.info()
base.isnull().sum()


plt.plot(base.ibov_ultimo)
plt.plot(base.dolar_ultimo)
plt.plot(base.sp500_ultimo)
plt.plot(base.cdi_ultimo)


#Boxplots

base.boxplot(['ibov_maxima', 'ibov_minima','ibov_abertura','ibov_ultimo'])
base.boxplot(['sp500_ultimo'])
base.boxplot(['cdi_ultimo'])
base.boxplot(['dolar_ultimo'])
base.boxplot(['vol'])

#Histograma
import seaborn as sn
sn.distplot(base['ibov_ultimo'], kde=False)
sn.distplot(base['dolar_ultimo'], kde=False)
sn.distplot(base['sp500_ultimo'], kde=False)
sn.distplot(base['cdi_ultimo'])

#Correlação com Ibovespa
correlation_ibov = base.corr()[['ibov_ultimo']]
plot = sn.heatmap(correlation_ibov, annot = True)  

#Definindo a tendencia em até 30 dias.

i = 0
j = 1
alvo = np.zeros(len(base)-30)

for i in range(0,len(base)-30):
    while j<31:
        if ((base["ibov_ultimo"][i]-base["ibov_ultimo"][i+j])
            /(base["ibov_ultimo"][i])) >= 0.08:
            alvo[i]=1
            
            
        if ((base["ibov_ultimo"][i]-base["ibov_ultimo"][i+j])
            /(base["ibov_ultimo"][i])) <= -0.08:
            alvo[i]=2
            
        
        j+=1
    i+=1
    j=1

alvo= pd.get_dummies(alvo)
alvo.columns = ['neutro','ganho', 'perda']
alvo.ganho.sum()/len(alvo)
alvo.perda.sum()/len(alvo)
alvo.neutro.sum()/len(alvo)
alvo.sum()
alvo.perda.sum()

base = base[0:2961]
base.reset_index(inplace=True)
base = pd.concat([base, alvo], axis=1)
base = base.sort_values(by = ['Data'])
base.set_index('Data', inplace = True)
base.isnull().sum()  


#Normalizando

from sklearn.preprocessing import MinMaxScaler
normalizador = MinMaxScaler(feature_range=(0,1))
base_normalizada = normalizador.fit_transform(base) 
base_normalizada = pd.DataFrame(base_normalizada, columns = ['ibov_maxima',
                                                             'ibov_minima',
                                                             'ibov_abertura',
                                                             'ibov_ultimo',
                                                             'vol','mm7',
                                                             'mm21','ifr14',
                                                             'irf7','ROC',
                                                             'OBV',
                                                             'dolar_ultimo',
                                                             'dmm14',
                                                             'sp500_ultimo',
                                                             'sp14',
                                                             'cdi_ultimo', 
                                                             'cdi14', 
                                                             'neutro',
                                                             'ganho', 'perda'])

 

  
#Frequencia de ganhos reais
base['ganho'].sum()/(len(base))
base['perda'].sum()/(len(base))
plt.plot(base['ibov_ultimo'])
plt.plot(ibov['ibov_ultimo'])
plt.plot(base_normalizada['ibov_ultimo'])

from sklearn.model_selection import train_test_split
# Separaçao da base de treinamento e de teste
entrada = base_normalizada.drop(['neutro','ganho','perda'],axis=1)
saida = base_normalizada[['neutro','ganho','perda']]
entrada, saida = np.array(entrada), np.array(saida)
#entrada = np.reshape(entrada, (entrada.shape[0], entrada.shape[1], 1))
x_treino, x_teste, y_treino, y_teste = train_test_split(entrada, saida, 
                                                        test_size=0.2, 
                                                        random_state = 30)


# Redes Neurais

from keras.models import Sequential
from keras.layers import Dense, Dropout
modelo_RNA = Sequential()
modelo_RNA.add(Dense(units = 200, input_dim = 17, activation = 'relu'))
modelo_RNA.add(Dropout(0.2))

modelo_RNA.add(Dense(units = 100, activation = 'relu'))
modelo_RNA.add(Dropout(0.2))

modelo_RNA.add(Dense(units = 50, activation = 'relu'))
modelo_RNA.add(Dropout(0.2))

modelo_RNA.add(Dense(units = 25, activation = 'relu'))

modelo_RNA.add(Dense(units = 12,  activation = 'relu'))

modelo_RNA.add(Dense(units = 6,  activation = 'relu'))

modelo_RNA.add(Dense(units = 3, activation = 'softmax'))

modelo_RNA.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
                  metrics = ['categorical_accuracy'])
modelo_RNA.fit(x_treino, y_treino, epochs = 1000, batch_size = 32)

previsao_RNA = modelo_RNA.predict(x_teste)

previsao_RNA = previsao_RNA > 0.5
previsao_RNA = previsao_RNA.astype(float)
print('Accuracy de ', accuracy_score(previsao_RNA, y_teste))
resultados_RNA = (classification_report(y_teste, previsao_RNA))

y_teste2 = [np.argmax(t) for t in y_teste]
previsao_RNA = [np.argmax(t) for t in previsao_RNA]

matriz_RNA = confusion_matrix(previsao_RNA, y_teste2)

#Arvore de Decisão
from sklearn.ensemble import RandomForestClassifier
modelo_TR = DecisionTreeClassifier(criterion='gini', random_state=50)
modelo_TR.fit(x_treino, y_treino)

previsao_TR= modelo_TR.predict(x_teste)
print('Accuracy de ', accuracy_score(previsao_TR, y_teste))
resultados_TR = (classification_report(y_teste, previsao_TR))

previsao_TR = [np.argmax(t) for t in previsao_TR]
matriz_TR = confusion_matrix(previsao_TR, y_teste2)




# Random Forest

modelo_RF = RandomForestClassifier(n_estimators = 39, criterion='entropy', random_state=1)
modelo_RF.fit(x_treino, y_treino)
previsao_RF= modelo_RF.predict(x_teste)
  
print('Accuracy de ', accuracy_score(previsao_RF, y_teste))
resultados_RF = (classification_report(y_teste, previsao_RF))
previsao_RF = [np.argmax(t) for t in previsao_RF]
matriz_RF = confusion_matrix(previsao_RF, y_teste2)


#KNN
from sklearn.neighbors import KNeighborsClassifier
modelo_KNN = KNeighborsClassifier(n_neighbors=1)
modelo_KNN.fit(x_treino, y_treino)

previsao_KNN= modelo_KNN.predict(x_teste)

print('Accuracy de ', accuracy_score(previsao_KNN, y_teste))
resultados_KNN = (classification_report(y_teste, previsao_KNN))

previsao_KNN = [np.argmax(t) for t in previsao_KNN]
matriz_KNN = confusion_matrix(previsao_KNN, y_teste2)

#SVM

y_treino_SVM = [np.argmax(t) for t in y_treino]
from sklearn.svm import SVC
modelo_svm = SVC(kernel='rbf',C=100, gamma=3.5)
modelo_svm.fit(x_treino,y_treino_SVM)

previsao_svm= modelo_svm.predict(x_teste)
matriz_svm = confusion_matrix(previsao_svm, y_teste2)
previsao_svm1 = pd.get_dummies(previsao_svm)
print('Accuracy de ', accuracy_score(previsao_svm1, y_teste))
resultados_SVM = (classification_report(y_teste, previsao_svm1))

#Combinação Redes Neurais + Arvore + KNN
j=0
x=0
y=0
z=0
previsao_combo = np.zeros(len(y_teste))
while j < len(x_teste):
    if previsao_RNA[j] == 0:
        x +=1
    if previsao_RNA[j] == 1:
        y += 1
    if previsao_RNA[j] == 2:
        z+=1
    if previsao_RF[j] == 0:
        x +=1
    if previsao_RF[j] == 1:
        y += 1
    if previsao_RF[j] == 2:
        z+=1
    if previsao_svm[j] == 0:
        x +=1
    if previsao_svm[j] == 1:
        y += 1
    if previsao_svm[j] == 2:
        z+=1
    if x>1:
        previsao_combo[j] = 0
    if y>1:
        previsao_combo[j] = 1
    if z>1:
        previsao_combo[j] = 2
    if z==x==y:
        previsao_combo[j]=previsao_RF[j]
        
    x=0
    y=0
    z=0
    j+=1
   

matriz_combo = confusion_matrix(previsao_combo, y_teste2)
previsao_combo = pd.get_dummies(previsao_combo)
print('Accuracy de ', accuracy_score(previsao_combo, y_teste))
resultados_combo = (classification_report(y_teste, previsao_combo))
