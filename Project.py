#Setup Enviroment
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Main source
import numpy as np
import pandas as pd
from CNN import CNN_Train, CNN_Test, CNN_ModelCreate, UpdateDatasetFromEmnist

def main():
    # UpdateDatasetFromEmnist()

    # CNN_ModelCreate()

    # CNN_Train(x_train, x_test, y_train, y_test)

    loaded_data = np.load('dataset.npz')
    images = loaded_data['images']
    i = np.random.randint(0, len(images) + 1)
    kq = CNN_Test(np.expand_dims(images[i], axis=0))
    print('Kết quả: ', kq)

if __name__ == "__main__":
    main()
