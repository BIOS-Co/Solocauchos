import numpy as np
import matplotlib.pyplot as plt
from utils import *
from scipy.signal import resample
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Connect to database:


def main():

    conn = create_server_connection(
        'localhost', 'admin', 'password', 'Solocauchos')

    labels = {
        0: "Sensor Desconectado",
        1: "Motor en Standby",
        2: "Motor trabajando"
    }

    # nombre del campo: resultado_ia en la tabla encabezado

    # Get the database head table

    query = "select * from pulsos_iot_vibration_encabezado;"
    col_names = get_col_names(conn, 'pulsos_iot_vibration_encabezado')
    encabezado = create_df(conn, query, col_names)

    # Do the query to obtain the latest signal

    sensores = ['w', 'x', 'y', 'z']
    col_names = get_col_names(conn, 'pulsos_iot_vibration_w')

    minuto = encabezado['cod_pulso'].iloc[-1] - 10
    total_length = 432000
    data = {}
    signals = np.zeros((1, 12000, 4))

    for i, sensor in enumerate(sensores):
        query = f"select * from pulsos_iot_vibration_{sensor} where cod_pulso_encabezado = {minuto};"

        x = np.array(create_df(conn, query, col_names)['ciclo'].tolist())
        x = resample(x, 12000)

        signals[0, :, i] = x

    # Load the model

    model = pickle.load(open('modelo.sav', 'rb'))

    # Do the prediction using the model

    # Preprocessing:

    X_rms = np.sqrt(np.mean(signals**2, axis=1))

    # Prediction:

    y_pred = model.predict(X_rms)[0]

    # Display results:

    print(labels[y_pred])
    print(X_rms)
    plt.plot(signals[0])
    plt.show()

    # Insert the prediction result in the database:

    query = f"""
    UPDATE pulsos_iot_vibration_encabezado
    SET resultado_ia = {y_pred},
    WHERE cod_pulso = {minuto};
    """

    # execute_query(conn, query)
    # print(f"Pulso analizado: {minuto}")


if __name__ == "__main__":
    main()
