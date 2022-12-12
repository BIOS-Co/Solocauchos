import mysql.connector
from mysql.connector import Error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import resample
from librosa.feature import mfcc

def create_server_connection(host_name, user_name, user_password, db_name):
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=user_password,
            database = db_name
        )
        print("MySQL Database connection successful")
    except Error as err:
        print(f"Error: '{err}'")

    return connection



def execute_query(connection, query):
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        connection.commit()
        print("Query successful")
    except Error as err:
        print(f"Error: '{err}'")


def read_query(connection, query):
    cursor = connection.cursor()
    result = None
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        return result
    except Error as err:
        print(f"Error: '{err}'")


def get_col_names(connection, table):
    results = read_query(connection, f"SHOW COLUMNS FROM {table};") 
    col_names = []

    for result in results:
        col_names.append(result[0])
    
    return col_names



def create_df(connection, query, col_names):
    results = read_query(connection, query)
    from_db = []


    for result in results:
        result = list(result)
        from_db.append(result)


    columns = col_names
    df = pd.DataFrame(from_db, columns=columns)

    return df

def get_signals(minuto, conn):

    sensores = ['w', 'x', 'y', 'z']
    col_names = get_col_names(conn, 'pulsos_iot_vibration_w')

    total_length = 432000
    data = {}
    signals = np.zeros((1,12000,4))

    for i,sensor in enumerate(sensores):
        query = f"select * from pulsos_iot_vibration_{sensor} where cod_pulso_encabezado = {minuto};"
        
        x = np.array(create_df(conn, query, col_names)['ciclo'].tolist())
        x = resample(x, 12000)


        signals[0,:,i] = x

    return signals

def feature_extraction(X_raw):
    x_i = []
    for i in range(X_raw.shape[2]):
        X_mfcc = mfcc(X_raw[0,:,i], n_mfcc = 5)
        print(f"mfcc of signal {i}: {X_mfcc.shape}")
        x_i.append(X_mfcc.reshape(-1))
    
    x_i = np.array(x_i).reshape(-1)
    x_i = np.expand_dims(x_i, axis = 0)
    return x_i