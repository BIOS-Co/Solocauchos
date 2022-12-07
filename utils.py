import mysql.connector
from mysql.connector import Error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

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