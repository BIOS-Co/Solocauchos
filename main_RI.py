import numpy as np
import matplotlib.pyplot as plt
from utils import *
from scipy.signal import resample
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import time

# Connect to database:


def main():

    conn = create_server_connection('localhost', 'root', 'pamd101216', 'iot')

    state_labels = {
    0: "Sensor Desconectado",
    1: "Motor en Standby",
    2: "Motor trabajando"
    }

    # nombre del campo: resultado_ia en la tabla encabezado

    # Load the models

    model_state = pickle.load(open('modelo_estado.sav', 'rb'))



    # Get the database head table

    query = "select * from pulsos_iot_vibration_encabezado;"
    col_names = get_col_names(conn, 'pulsos_iot_vibration_encabezado')
    encabezado = create_df(conn, query, col_names)


    # Do the query to obtain the latest signal

    minuto = encabezado['cod_pulso'].iloc[-1] - 1
    signals = get_signals(minuto, conn)


    # Do the state prediction using the model

    # Preprocessing:

    X_rms = np.sqrt(np.mean(signals[:,-100:,:]**2, axis = 1))

    # Prediction:

    last_state_pred = model_state.predict(X_rms)[0]

    first_state_pred = model_state.predict(np.sqrt(np.mean(signals[:,:100,:]**2, axis = 1)))

    # Do the anomaly detection according to the signal type:


    if last_state_pred == first_state_pred:
        if last_state_pred == 0:
            anomaly_pred = 0
        
        elif last_state_pred == 1:
            anomalies = np.sum(np.abs(signals)>0.03)
            if anomalies > 300:
                anomaly_pred = 1
            else:
                anomaly_pred = 0
        
        elif last_state_pred == 2:
            anomalies = np.sum(np.abs(signals)>0.6)
            if anomalies > 50:
                anomaly_pred = 1
            else:
                anonaly_pred = 0    

    else:
        anomaly_pred = 0

    # Display results:

    print(state_labels[last_state_pred])
    if anomaly_pred == 0:
        print("Señal sin anomalía")
    
    else:
        print("Señal con anomalía")
    
    plt.plot(signals[0])
    plt.show()

    # Insert the prediction result in the database:


    query = f"""
    UPDATE pulsos_iot_vibration_encabezado
    SET resultado_ia = {last_state_pred},
    WHERE cod_pulso = {minuto};
    """

    execute_query(conn, query)
    print(f"Pulso analizado: {minuto}")


    query_anomalia = f"""
    UPDATE pulsos_iot_vibration_encabezado
    SET anomalia = {anomaly_pred},
    WHERE cod_pulso = {minuto};
    """

    execute_query(conn, query_anomalia)
    print(f"Pulso analizado: {minuto}")

        

if __name__ == "__main__":
    main()