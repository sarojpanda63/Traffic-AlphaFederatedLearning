from Model import *
from PythonUtils import *
import subprocess

def run_local_model(model):
    process1 = subprocess.Popen(["python", "Client1.py",model])
    process2 = subprocess.Popen(["python", "Client2.py",model])
    process3 = subprocess.Popen(["python", "Client3.py",model])
    process4 = subprocess.Popen(["python", "Client4.py",model])
    process5 = subprocess.Popen(["python", "Client5.py",model])
    process1.wait()
    process2.wait()
    process3.wait()
    process4.wait()
    process5.wait()

def get_the_weightage_of_each_client(lookback,test_size):
    s = lookback+test_size
    c_path = os.getcwd()
    client1_space = c_path + '\\data\\client1_space\\'
    client2_space = c_path + '\\data\\client2_space\\'
    client3_space = c_path + '\\data\\client3_space\\'
    client4_space = c_path + '\\data\\client4_space\\'
    client5_space = c_path + '\\data\\client5_space\\'
    df1 = load_data(client1_space, 'LOSAng')[s:]
    df2= load_data(client2_space, 'NYCMng')[s:]
    df3= load_data(client3_space, 'SNVAng')[s:]
    df4= load_data(client4_space, 'STTLng')[s:]
    df5= load_data(client5_space, 'WASHng')[s:]
    lst=[len(df1), len(df2), len(df3), len(df4), len(df5)]
    norm_lst = normalize_list(lst)
    return norm_lst

if __name__ == '__main__':
    initialize_global_model_01('server_model.h5')
    server_to_clients('server_model.h5')
    run_local_model('server_model.h5')
    weithages_list = get_the_weightage_of_each_client(70, 100)
    alpha = 3
    factor = 1000
    iteration = 10

    for i in list(range(1,iteration+1)):
        print('Additional epoch....',i,'/',iteration)
        clients_to_server()
        get_weighted_average_of_model_and_set_weight_of_server_model_and_save_server_weights(weithages_list,alpha,factor)
        server_to_clients('server_weights.h5')
        run_local_model('server_weights.h5')