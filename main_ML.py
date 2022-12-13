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

    conn = create_server_connection('localhost', 'admin', 'password', 'Solocauchos')

    state_labels = {
    0: "Sensor Desconectado",
    1: "Motor en Standby",
    2: "Motor trabajando"
    }

    # nombre del campo: resultado_ia en la tabla encabezado

    # Load the models

    model_state = pickle.load(open('modelo_estado.sav', 'rb'))
    model_standby = pickle.load(open('anomalia_standby_fft.sav', 'rb'))
    model_trabajando = pickle.load(open('modelo_trabajando.sav', 'rb'))


    while(True):

         # Get the database head table

        query = "select * from pulsos_iot_vibration_encabezado;"
        col_names = get_col_names(conn, 'pulsos_iot_vibration_encabezado')
        encabezado = create_df(conn, query, col_names)


        # Do the query to obtain the latest signal

        minuto = encabezado['cod_pulso'].iloc[-1] - 1

        signals = get_signals(minuto, conn)


        # Do the state prediction using the model

        # Preprocessing:

        X_rms = np.sqrt(np.mean(signals[:,:100,:]**2, axis = 1))

        # Prediction:

        state_pred = model_state.predict(X_rms)[0]


        # Do the anomaly detection according to the signal type:

        if state_pred == 0:
            anomaly_pred = 0
        
        elif state_pred == 1:
            print("Signals shape: ", signals.shape)
            X_mfcc = feature_extraction_fft(signals)
            print("Features shape: ", X_mfcc.shape)
            X_mfcc = StandardScaler().fit_transform(X_mfcc)
            anomaly_pred = model_standby.predict(X_mfcc)[0]

        elif state_pred == 2:
            X_mfcc = feature_extraction_fft(signals)
            print("Features shape: ", X_mfcc.shape)
            X_mfcc = StandardScaler().fit_transform(X_mfcc)
            anomaly_pred = model_trabajando.predict(X_mfcc)[0]


        # Display results:

        print(state_labels[state_pred])
        if anomaly_pred == 0:
            print("Señal sin anomalía")
        
        else:
            print("Señal con anomalía")
       
        # plt.plot(signals[0])
        # plt.show()

        # Insert the prediction result in the database:


        query = f"""
        UPDATE pulsos_iot_vibration_encabezado
        SET resultado_ia = {state_pred},
        WHERE cod_pulso = {minuto};
        """

        # execute_query(conn, query)
        # print(f"Pulso analizado: {minuto}")



        query_anomalia = f"""
        UPDATE pulsos_iot_vibration_encabezado
        SET anomalia = {anomaly_pred},
        WHERE cod_pulso = {minuto};
        """

        # execute_query(conn, query_anomalia)
        # print(f"Pulso analizado: {minuto}")

        time.sleep(90)

if __name__ == "__main__":
    main()