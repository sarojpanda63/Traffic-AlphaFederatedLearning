import os
import sys
from Model import *
from PythonUtils import *
from tensorflow.python.keras.models import load_model

if __name__ == '__main__':
    filename = str(sys.argv[1])
    c_path = os.getcwd()
    client4_space = c_path + '\\data\\client4_space\\'
    df = load_data(client4_space,  'STTLng')
    train_x, train_y, test_x, test_y = make_rnn_data('STTLng',df,  .7,70)
    val_x = test_x[:-100]
    val_y = test_y[:-100]
    test_x = test_x[-100:]
    test_y = test_y[-100:]
    # print(train_x.shape, val_x.shape, test_x.shape, train_y.shape, val_y.shape, test_y.shape)
    if os.path.exists(client4_space + 'client4_model.h5') and os.path.exists(client4_space + filename):
        cm = load_model(client4_space + 'client4_model.h5')
        cm.load_weights(client4_space + filename)
        find_mean_squared_error(cm, '4', test_y, test_x)
        history = train_model_without_callback(cm, 1, 256, train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y)
        cm.save_weights(client4_space + 'client4_weights.h5')
        cm.save(client4_space + 'client4_model.h5')
        update_and_save_the_learning_curve(client4_space, history, 'loss','STTLng')

    else:
        m = load_model(client4_space + filename)
        history = train_model_without_callback(m, 1, 256, train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y)
        m.save_weights(client4_space + 'client4_weights.h5')
        m.save(client4_space + 'client4_model.h5')
        update_and_save_the_learning_curve(client4_space, history, 'loss','STTLng')
    delete_file(client4_space + filename)
