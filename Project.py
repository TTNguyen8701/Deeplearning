#Setup Enviroment
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Main source
import numpy as np
import pandas as pd
from CNN import CNN_Train, CNN_Test, CNN_ModelCreate

def bin2int(bit_array):
    bit_string = ''.join(i.astype(str) for i in bit_array)  # "1011"
    integer_value = int(bit_string, 2)           # Chuyển sang số nguyên
    return integer_value

def main():
    CNN_ModelCreate()

    # a = 5642
    # df = pd.DataFrame(np.squeeze(x_test[a]))
    # df.to_csv('test.txt', sep='\t')
    # temp = [round(i, 5) for i in CNN_Test(np.expand_dims(x_test[a], axis=0))[0]]
    # print(temp)

    # CNN_Train(x_train, x_test, y_train, y_test)

if __name__ == "__main__":
    main()
