import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
import pickle
from datetime import datetime, timedelta

scaler = MinMaxScaler(feature_range=(0, 1))
def dataset():
    df = pd.read_csv("Dataset.csv")
    data = df[["Tanggal", "IDR"]]
    data['Tanggal'] = pd.to_datetime(data['Tanggal'], format='%d/%m/%Y')
    data['IDR'] = [float(x.replace(".", "").replace(",", ".")) for x in data['IDR']]
    return data

def split_data(n):
    data = dataset()
    idr_values = data["IDR"].values.reshape(-1, 1)
    idr_normalized = scaler.fit_transform(idr_values)
    data["IDR"] = idr_normalized
    train_size = int(len(data) * float(f"0.{n}"))
    data_train, data_test = data.iloc[:train_size], data.iloc[train_size:]
    return data_train, data_test

def create_timesteps(data, timesteps):
    data_timesteps = np.array([[j for j in data[i:i+timesteps]] for i in range(0, len(data)-timesteps+1)])[:,:,0]
    return data_timesteps

def create_X_y(data, timesteps):
    X = data[:, :timesteps-1]
    y = data[:, [timesteps-1]]
    return X, y

def svr(n, t, c, gamma, epsilon, degree, kernel):
    split = split_data(n)
    data_train = split[0]
    data_test = split[1]
    train = data_train["IDR"].to_numpy().reshape(-1,1)
    test = data_test["IDR"].to_numpy().reshape(-1,1)
    timesteps = t
    train_data_timesteps = create_timesteps(train, timesteps)
    test_data_timesteps = create_timesteps(test, timesteps)
    X_train, y_train = create_X_y(train_data_timesteps, timesteps)
    X_test, y_test = create_X_y(test_data_timesteps, timesteps)
    model = SVR(kernel= kernel, degree= degree, C= c, gamma= gamma, epsilon= epsilon)
    model.fit(X_train, y_train[:,0])
    y_pred = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    mape = mean_absolute_percentage_error(y_test, y_pred)

    return mape*100, y_test, y_pred

def prediksi(datas):
    new_data = np.array(datas)
    with open("model.pkl", 'rb') as model_file:
        loaded_model = pickle.load(model_file)
    new_data_normalized = scaler.fit_transform(new_data.reshape(-1, 1))
    new_data_timesteps = create_timesteps(new_data_normalized, 3)
    predictions = loaded_model.predict(new_data_timesteps)
    predictions_original_scale = scaler.inverse_transform(predictions.reshape(-1, 1))
    return int(predictions_original_scale[-1])

def day():
    temp_day = []
    today = datetime.now()
    tomorrow = today + timedelta(days=1)
    five_days_ago = today - timedelta(days=5)
    date_range = [five_days_ago + timedelta(days=i) for i in range(6)]
    formatted_dates = [date.strftime('%d/%m/%Y') for date in date_range]
    for formatted_date in formatted_dates:
        temp_day.append(formatted_date)
    return temp_day, tomorrow.strftime('%d/%m/%Y')


    


