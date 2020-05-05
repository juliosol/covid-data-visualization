from keras.layers import Dense, Input, Activation, ReLU
from keras import models
from keras.models import Sequential
from keras.optimizers import Adam, SGD

import pandas as pd
import os
from pathlib import Path
import numpy as np

import plotly_express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

#model = Sequential()
#model.add(Dense(100, input_dim=1, activation='relu'))
#model.add(Dense(100, activation='relu'))
#model.add(Dense(100, activation='relu'))
#model.add(Dense(100, activation='relu'))
#model.add(Dense(100, activation='relu'))
#model.add(Dense(100, activation='relu'))
#model.add(Dense(100, activation='relu'))

#model.add(Dense(1, activation='relu'))

Visible=  Input(shape=(1,))

Dense_l1 = Dense(100, name='Dense_l1')(Visible)
ReLU_l1 = ReLU(name='ReLU_l1')(Dense_l1)

Dense_l2 = Dense(100, name='Dense_l2')(ReLU_l1)
ReLU_l2 = ReLU(name='ReLU_l2')(Dense_l2)

Dense_l3 = Dense(100, name='Dense_l3')(ReLU_l2)
ReLU_l3 = ReLU(name='ReLU_l3')(Dense_l3)

Dense_l4 = Dense(100, name='Dense_l4')(ReLU_l3)
ReLU_l4 = ReLU(name='ReLU_l4')(Dense_l4)

Dense_l5 = Dense(100, name='Dense_l5')(ReLU_l4)
ReLU_l5 = ReLU(name='ReLU_l5')(Dense_l5)

Dense_l6 = Dense(100, name='Dense_l6')(ReLU_l5)
ReLU_l6 = ReLU(name='ReLU_l6')(Dense_l6)

Dense_l7 = Dense(100, name='Dense_l7')(ReLU_l6)
ReLU_l7 = ReLU(name='ReLU_l7')(Dense_l7)

Output = Dense(1, name='Output')(ReLU_l7)
ReLU_Output = ReLU(name='ReLU_Output')(Output)

model = models.Model(inputs=Visible, outputs=ReLU_Output)

model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001), metrics=['accuracy'])


curr_folder = Path(os.getcwd())
data_folder = os.path.join(curr_folder, 'COVID-19', 'csse_covid_19_data', 'csse_covid_19_time_series')

covid_19_confirmed_orig = pd.read_csv(os.path.join(data_folder, 'time_series_covid19_confirmed_global.csv'))

covid_19_confirmed_orig.columns = ['Province', 'Country', 'Lat', 'Long'] + pd.to_datetime(covid_19_confirmed_orig.columns[4:]).strftime('%m/%d/%Y').to_list()

covid_19_confirmed_country = covid_19_confirmed_orig.copy().groupby('Country').sum()

covid_19_confirmed = covid_19_confirmed_country.copy().drop(['Lat', 'Long'],axis=1).sum().to_frame().reset_index()
covid_19_confirmed.columns = ['Date', 'World Confirmed']
covid_19_confirmed['Date'] = pd.to_datetime(covid_19_confirmed['Date'])
print(covid_19_confirmed)

y = np.log10(covid_19_confirmed.loc[:,'World Confirmed'].to_numpy().astype('float32'))
x = np.arange(1,len(y)+1)

epochs = 10000

model.fit(x.reshape(y.shape[0],1), y.reshape(y.shape[0],1), epochs=epochs, batch_size=10)

model.save('worldwide_confirmed_model.h5')

model = models.load_model('worldwide_confirmed_model.h5')

prediction_days = 10

temp_data = covid_19_confirmed.loc[:,'World Confirmed']
data = np.power(10, model.predict(np.arange(1, len(temp_data) + prediction_days + 1)))

#temp_data = covid_19_confirmed_orig.copy()
#temp_data2 = covid_19_confirmed_orig.copy().groupby('Country').sum()

#print(data)

lakh = 100000

f = plt.figure(figsize=(15,10))
ax = f.add_subplot(111)

date = np.arange(0,len(temp_data))

marker_style =  dict(linewidth=3, linestyle='-', marker='o',markersize=7, markerfacecolor='#ffffff')

plt.plot(date,temp_data/lakh,"-.",color="darkcyan",**marker_style, label="Actual Curve")

date_pred = np.arange(0,len(data))
plt.plot(date_pred,data/lakh,"-.",color="orangered",label="Predicted Curve")

plt.show()

