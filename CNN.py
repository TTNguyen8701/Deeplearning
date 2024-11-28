import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_datasets as tfds
import numpy as np

labels_to_unicode = tf.constant([
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',  # 0-9
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',  # A-Z
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'   # a-z
], dtype=tf.string)

def pad_array(array, target_length, value=0):
    while len(array) < target_length:
        array.insert(0, value)  # Thêm phần tử vào đầu mảng
    return np.array(array)

def char2bitarray(char):
    if not isinstance(char, tf.Tensor):
        char = tf.convert_to_tensor(char, dtype=tf.string)

    if char.shape.rank == 0:  # Nếu char là scalar
        char = tf.expand_dims(char, 0)  # Thêm chiều để tensor có shape (1,)

    # Chuyển đổi mỗi byte trong mã UTF-8 thành chuỗi bit
    byte_array = tf.strings.unicode_encode(char, 'UTF-8')  # Chuyển ký tự thành mã byte
    print(byte_array)
    bit_array = tf.bitcast(byte_array, tf.int8)  # Chuyển mã byte thành bit array

    # bit_array = pad_array(bit_array, 4*8)

    return bit_array 

def preprocess(image, label):
    image = tf.transpose(image)
    image = tf.expand_dims(image, axis=-1)

    label = tf.cast(label, dtype=tf.int32)
    #label = tf.gather(labels_to_unicode, label)
    label = char2bitarray(label)
    # label = tf.convert_to_tensor(label, dtype=tf.int64)
    # label = tf.reshape(label, (1, 32))

    return image, label

def CNN_ModelCreate():
    ds_train , ds_test = tfds.load('emnist/byclass', split=['train', 'test'], as_supervised=True)
    
    ds_train = ds_train.map(preprocess)
    ds_test = ds_test.map(preprocess)

    model = models.Sequential(
        [
            layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='tanh'),
            layers.Dropout(rate=0.5),
            layers.AveragePooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(units=64, activation='tanh'),
            layers.Dropout(rate=0.2),
            layers.Dense(units=4*8, activation='sigmoid'),
        ]
    )

    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

    history = model.fit(ds_train, epochs=10, batch_size=32, shuffle=1000)

    # Đánh giá mô hình
    loss, accuracy = model.evaluate(ds_test)
    print(f"Loss:\t {loss}")
    print(f"Accuracy:\t {accuracy}")

    # Lưu mô hình
    model.save('ModelDataset.keras')

def CNN_Train(x_train, x_test, y_train, y_test):
    
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
    model = models.load_model('ModelDataset.keras')
    predictions = model.predict(x_new, verbose=0)
    return predictions

    