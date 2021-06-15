import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

#ACTIVATING VIRTUAL ENV AND RUNNING CODE IN TERMINAL
#.venv2\scripts\activate
#python ./main.py


#default variables
default_ticker = 'SPY'
default_days = 60

#getting user ticker input
print('SPY, QQQ, IWM, DIA, AAPL, FB are all examples of tickers')
print('***enter NONE if you want to use the default ticker***')
company = input("Enter a company ticker: ")
company = company.upper()
print ('you entered', company)
if company == 'NONE' :
    company = default_ticker


#getting amount of days the users wants to look into the past
print('RECOMMENDED NUMBER OF DAYS: 60')
print('***enter 0 if you want to use the default number of days***')
prediction_days = int(input("Enter how many days you want the neural network to look into the past: "))
print ('you entered', prediction_days)
if prediction_days == 0 :
    prediction_days = default_days



# loading data 
#company = 'AAPL'

start = dt.datetime(2013,1,1)
end = dt.datetime(2021,1,1)

data = web.DataReader(company, 'yahoo', start, end)

# prepare data

scalar = MinMaxScaler(feature_range=(0,1))
scaled_data = scalar.fit_transform(data['Close'].values.reshape(-1,1))

#looking back days 
#prediction_days = 60

x_train = []
y_train = []

for x in range(prediction_days, len(scaled_data)): 
    x_train.append(scaled_data[x - prediction_days:x, 0 ])
    y_train.append(scaled_data[x,0])

x_train, y_train = np.array(x_train), np.array(y_train)

x_train= np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#building the neural network model 
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1)) #prediction of the next price 

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train,y_train, epochs=25, batch_size=32)

'''
testing the model accuracy on existing data
'''
#loading test data 
test_start = dt.datetime(2020,1,1)
test_end = dt.datetime.now()

test_data = web.DataReader(company, 'yahoo', test_start, test_end)

actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data - prediction_days):].values
model_inputs = model_inputs.reshape(-1,1)

model_inputs = scalar.transform(model_inputs)

# make predictions on test data 

x_test = []

for x in range(prediction_days, len(model_inputs)) :
    x_test.append(model_inputs[x-prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scalar.inverse_transform(predicted_prices)

#plotting the test predictions 

plt.plot(actual_prices, color="black", label=f"Actual {company} Price")
plt.plot(predicted_prices, color="green", label=f"Predicted {company} Price")
plt.title(f"{company} Share Price")
plt.xlabel('Time')
plt.ylabel(f'{company} Share Price')
plt.legend()
plt.show()

#predicting the future days

real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs + 1), 0]]

real_data =  np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

#print(scalar.inverse_transform(real_data))
prediction = model.predict(real_data)
prediction = scalar.inverse_transform(prediction)
print(f"PREDICTED PRICE FOR NEXT OPEN TRADING DAY: {prediction}")

