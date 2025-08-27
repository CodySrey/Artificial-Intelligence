import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer

#Loads data
def loaddata(file_path): #Loads Data and Creates Data Features
    data = pd.read_csv(file_path)
    data.fillna(0, inplace=True) #NaN spots in csv gets turned into a 0
    #Uses columns from csv to be features and target for the LSTM
    features = data[["AADT_2023", "Truck%_2023", "AADT_2022", "Truck%_2022", "AADT_2021", "Truck%_2021", "AADT_2020", "Truck%_2020", "AADT_2019", "Truck%_2019", "AADT_2018", "Truck%_2018", "AADT_2017", "Truck%_2017", "AADT_2016", "Truck%_2016", "AADT_2015", "Truck%_2015", "AADT_2014", "Truck%_2014"]]
    target = data[["AADT_2023", "Truck%_2023"]]
    data["Traffic_State"] = pd.cut( #Add traffic state classification for Logistic Regression
        data["AADT_2023"], bins=[-1, 5000, 15000, np.inf], labels=[0, 1, 2]
    )
    return features, target, data[["Traffic_State"]]

#Prepare data for LSTM model
def predata(data, target, seq_length=4): #Preps the Data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(target)
    X, y = [], []
    for i in range(seq_length, len(scaled_data)):
        X.append(data.iloc[i - seq_length:i].values)
        y.append(scaled_data[i])
    X, y = np.array(X), np.array(y)
    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]
    return X_train, y_train, X_test, y_test, scaler

#Define LSTM model
def lstmmodel(input_shape, output_units): #output_units is a parameter
    model = Sequential(
        [LSTM(units=50, return_sequences=False, input_shape=input_shape), Dense(units=output_units)]  #output_units defines Dense
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

#Train the LSTM model
def lstmtrain(model, X_train, y_train, X_test, y_test, epochs=20, batch_size=32): #Trains the LSTM
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1)
    return history

#Evaluate the LSTM model
def lstmeval(model, X_test, y_test, scaler):
    y_pred = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred.reshape(-1, y_pred.shape[1])) #Makes sure proper inverse transformation
    y_test = scaler.inverse_transform(y_test.reshape(-1, y_test.shape[1]))
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("Mean Absolute Error: ", mae)
    print("Root Mean Squared Error: ", rmse)

    #Fix plotting logic
    plt.figure(figsize=(10, 6))
    plt.plot(y_test[:, 0], label='Traffic Volume', color='black', marker='x')  #Plot the AADT values
    plt.plot(y_pred[:, 0], label='Predicted Traffic', color='red', marker='o')  #Plot the AADT predictions 
    plt.title("LSTM Model Prediction")
    plt.xlabel("Data Index")
    plt.ylabel("Traffic Volume")
    plt.legend()
    plt.grid()
    plt.show()

#Logistic Regression Training
def lrtrain(data): #Logistic Regression training
    scaler = StandardScaler()
    imputer = SimpleImputer(strategy="mean") #Show missing values with the mean
    X = imputer.fit_transform(data[["AADT_2022", "Truck%_2022", "AADT_2021", "Truck%_2021"]])
    y = data["Traffic_State"].fillna(method='ffill') #Fills NaN values in the target where there could be empty spots
    train_size = int(0.8 * len(X))
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    #Plot predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(y_test)), y_test, color='black', label='History', alpha=0.6)
    plt.scatter(range(len(y_pred)), y_pred, color='red', label='Prediction', alpha=0.6)
    plt.title("Logistic Regression: Traffic State Classification")
    plt.xlabel("Data Index")
    plt.ylabel("Traffic State")
    plt.legend()
    plt.grid()
    plt.show()

    return model, scaler

def main(file_path):  #Main Function
    features, target, traffic_state = loaddata(file_path)

    #Logistic Regression for traffic state classification
    print("Logistic Regression Model for Traffic State Classification training in progress.")
    logistic_model, logistic_scaler = lrtrain(features.assign(Traffic_State=traffic_state))
    X_train, y_train, X_test, y_test, lstm_scaler = predata(features, target) #LTSM for traffic prediction
    print("\nLSTM Model for Traffic Volume Prediction training in progress.")
    buildlstmmodel = lstmmodel((X_train.shape[1], X_train.shape[2]), target.shape[1]) #Pass target.shape[1]
    history = lstmtrain(buildlstmmodel, X_train, y_train, X_test, y_test)
    
    #Training history Graph
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title("LSTM Training History")
    plt.grid()
    plt.show()

    #Evaluates LTsM model
    print("\nEvaluating LSTM Model on Test Dataset.")
    lstmeval(buildlstmmodel, X_test, y_test, lstm_scaler)

#File path: Adjust to where your dataset is located
if __name__ == "__main__":
    main("C:/Users/Mucka/Documents/AI/Final Edit/Trafficdata.csv")