'''
Usage:
    read_db.py [Options]


Options:
--host_name=<str>     host name. ex: localhost
--user_name=<str>     user name to authenticate in the db
--user_password=<str> user passwordd
--db_name=<str>       name of the schema
--save_path=<str>     path to the folder to save the exported data

'''
from utils import *
from docopt import docopt
from docopt import docopt
import os

doc = '''
Usage:
    read_db.py <host_name> <user_name> <user_password> <db_name> <save_path>

Options:
--host_name=<str>     host name. ex: localhost
--user_name=<str>     user name to authenticate in the db
--user_password=<str> user passwordd
--db_name=<str>       name of the schema
--save_path=<str>     path to the folder to save the exported data

'''


def main(host_name, user_name, user_password, db_name, save_path):

    
    # Create the output directories in case they dont exist yet

    for i in range(1,7):
        if  not (os.path.exists(os.path.join(save_path, str(i)))):
            os.mkdir(os.path.join(save_path, str(i)))
    
    # Connect to the database:

    conn = create_server_connection(host_name, user_name, user_password, db_name)

    # Lectura del encabezado:

    query = "select * from pulsos_iot_vibration_encabezado;"
    col_names = get_col_names(conn, 'pulsos_iot_vibration_encabezado')
    encabezado = create_df(conn, query, col_names)

    sensores = ['w', 'x', 'y', 'z']

    col_names = get_col_names(conn, 'pulsos_iot_vibration_w')


    query = f"select * from pulsos_iot_vibration_w where cod_pulso_encabezado = 1;"
            
    w = create_df(conn, query, col_names)['ciclo'].tolist()

    total_length = len(w)

    for minuto in encabezado['cod_pulso']:
        
        print(f"Reading minute {minuto}")
        data = {}
        for sensor in sensores:
            query = f"select * from pulsos_iot_vibration_{sensor} where cod_pulso_encabezado = {minuto};"
            
            x = np.array(create_df(conn,query, col_names)['ciclo'].tolist())
            x = np.pad(x, (0,total_length-x.shape[0]), 'constant', constant_values = 0)


            data[sensor] = x
        
        df = pd.DataFrame(data, columns=sensores)

        print(df.shape)
        label = encabezado['cod_etiqueta'].iloc[minuto-1]
        fecha = encabezado['fecha_cliente'].iloc[minuto-1]
        full_path = os.path.join(save_path,str(label),fecha.strftime("%m-%d-%Y %H:%M:%S")+'.csv')
        df.to_csv(full_path)
        
        print(f"Label: {label}")
        print(f"Saved fecha {fecha}")




if __name__ == '__main__':
    arguments = docopt(doc)
    

    host_name  = arguments['<host_name>']
    user_name = arguments['<user_name>']
    user_password = arguments['<user_password>']
    db_name = arguments['<db_name>']
    save_path = arguments['<save_path>']

    main(host_name, user_name, user_password, db_name, save_path)

