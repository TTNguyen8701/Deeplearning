import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_datasets as tfds

import pandas as pd
import numpy as np

labels_to_unicode = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',  # 0-9
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',  # A-Z
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'   # a-z
]

def pad_array(array, target_length, value=0):
    while len(array) < target_length:
        array.insert(0, value)  # Thêm phần tử vào đầu mảng
    return np.array(array)

def char2bitarray(char):
     # Mã hóa ký tự thành bytes theo chuẩn UTF-8
    byte_array = char.encode('utf-8')
    
    # Chuyển từng byte thành chuỗi nhị phân 8 bit
    bit_array = []
    for byte in byte_array:
        # Chuyển byte thành chuỗi nhị phân 8 bit
        bit_rep = format(byte, '08b')  # '08b' đảm bảo có 8 bit
        bit_array.extend([int(bit) for bit in bit_rep])
    
    return np.array(pad_array(bit_array, 32))

def UpdateDatasetFromEmnist():
    emnist_data = tfds.load('emnist/byclass', as_supervised=True)
    train_data, test_data = emnist_data['train'], emnist_data['test']
    dataset = train_data.concatenate(test_data)
    dataset = dataset.shuffle(1000)
    
    images = []
    labels = []
    for image, label in dataset:
        image = np.array(image.numpy())
        image = np.squeeze(image)
        image = np.transpose(image)
        image = np.expand_dims(image, axis=-1)
        
        label = label.numpy()
        label = char2bitarray(labels_to_unicode[label])

        images.append(image)
        labels.append(label)
    
    # Lưu mảng vào file CSV
    np.savez('dataset.npz', images=images, labels=labels)
    
def CNN_ModelCreate():
    # Truy xuất các mảng từ file `.npz`
    loaded_data = np.load('dataset.npz')

    # Lấy các mảng đã lưu
    images = loaded_data['images']
    labels = loaded_data['labels']

    model = models.Sequential(
        [
            layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='tanh'),
            layers.Dropout(rate=0.5),
            layers.AveragePooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(units=64, activation='tanh'),
            layers.Dropout(rate=0.2),
            layers.Dense(units=32, activation='sigmoid'),
        ]
    )

    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

    history = model.fit(images, labels, epochs=10, batch_size=64, shuffle=1000)

    # Đánh giá mô hình
    #loss, accuracy = model.evaluate(ds_test)

    # Lưu mô hình
    model.save('ModelDataset.keras')

def CNN_Train():
    
    # Thêm chiều cho dữ liệu đầu vào
    y_train = np.array([(pad_array(list(bin(y)[2:]), 4)) for y in y_train], dtype=np.float32)
    y_test = np.array([(pad_array(list(bin(y)[2:]), 4)) for y in y_test], dtype=np.float32)
    #y_train = keras.utils.to_categorical(y_train, num_classes=10)
    #y_test = keras.utils.to_categorical(y_test, num_classes=10)

    model = models.load_model('ModelDataset.keras')

    history = model.fit(x_train, y_train, epochs=10, batch_size=32)

    # Đánh giá mô hình

    # Dự đoán
    #predictions = model.predict(x_new)

    # Lưu mô hình
    model.save('ModelDataset.keras')

def CNN_Test(x_new):
    # Tải mô hình
    df = pd.DataFrame(np.squeeze(x_new))
    df.to_csv('test.txt', sep='\t')

    model = models.load_model('ModelDataset.keras')
    predictions = model.predict(x_new, verbose=0)

    bit_array = np.round(np.array(predictions), decimals=0)[0].astype(int)
    print('predictions: ', ''.join(map(str, bit_array)))

    byte_list = []
    for i in range(0, 4):
        byte = int(''.join(map(str, bit_array[i*8:i*8+8])), base=2)
        byte_list.append(byte)
    
    # Chuyển đổi danh sách byte thành đối tượng bytes
    byte_data = bytes(byte_list)
    
    # Giải mã chuỗi byte thành ký tự UTF-8
    kq = byte_data.decode('utf-8')

    return kq

    