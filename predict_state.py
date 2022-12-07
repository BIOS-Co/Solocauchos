import numpy as np
import matplotlib.pyplot as plt
from utils import *
from scipy.signal import resample
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Connect to database:


def main():

    conn = create_server_connection('localhost', 'admin', 'password', 'Solocauchos')

    labels = {
    1: 'Sensor desconectado',
    2: 'Standby con interferencia',
    3: 'Standby SIN interferencia',
    4: 'Trabajando',
    5: 'Trabajando + Standby',
    6: 'Trabajando + Anomalia',
    7: 'Standby + Anomalia'}

    # Get the database head table

    query = "select * from pulsos_iot_vibration_encabezado;"
    col_names = get_col_names(conn, 'pulsos_iot_vibration_encabezado')
    encabezado = create_df(conn, query, col_names)


    # Do the query to obtain the latest signal

    
    sensores = ['w', 'x', 'y', 'z']
    col_names = get_col_names(conn, 'pulsos_iot_vibration_w')

    minuto = encabezado['cod_pulso'].iloc[-1]
    total_length = 432000
    data = {}
    for sensor in sensores:
        query = f"select * from pulsos_iot_vibration_{sensor} where cod_pulso_encabezado = {minuto};"
        
        x = np.array(create_df(conn, query, col_names)['ciclo'].tolist())
        x = np.pad(x, (0,total_length-x.shape[0]), 'constant', constant_values = 0)


        data[sensor] = x
    
    df = pd.DataFrame(data, columns=sensores)

    signals = np.array(df)

    # Load the model

    model = pickle.load(open('modelo.sav', 'rb'))

    # Do the prediction using the model

    # Preprocessing:

    # Sampling:

    X_sampled = np.zeros((1,12000, signals.shape[1]))

    X_sampled[0,:,0] = resample(signals[:,0], 12000)
    X_sampled[0,:,1] = signals[:12000,1]
    X_sampled[0,:,2] = signals[:12000,2]
    X_sampled[0,:,3] = signals[:12000,3]

    X_rms = np.sqrt(np.mean(X_sampled**2, axis = 1))

    y_pred = model.predict(X_rms)[0]

    print(labels[y_pred])
    print(X_rms)
    plt.plot(X_sampled[0])
    plt.show()

if __name__ == "__main__":
    main()