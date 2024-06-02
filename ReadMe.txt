Các bước để sử dụng Py2C
bước 1: nếu dùng cho các model thông thường thì hãy vào file Py2C.py phần init. Sau đó, hãy thay đổi resnet=False. Còn nếu sử dụng cho resnet thì hãy chỉnh resnet=True và chỉnh các biến num_layers_in_block(số layer trong 1 basic block),num_layers_before_block(số layer trước khi vào basic block đầu tiên), num_blocks(số block trong resnet) theo model của bạn.
bước 2: ở main.py tại dòng: pyc_lib = Py2C(".....h5"), hãy điền vào phần .... file h5 của bạn
bước 3: chạy main.py
Bước 4: code sẽ xuất ra các đoạn code cpp và header file. Sau đó cho đoạn code được xuất ra vào project c++ để chạy(thay đổi hằng numberofpicture = số lượng input, hằng d = dimension của input.
********LƯU Ý: 

~code c++ xuất ra chỉ có thể chạy trên visual studio code.


Các layer và Configuration của nó có thể dùng (Áp dụng cho cả Conv2d và conv1d)
+ convolution  layer (kernel_size, strides padding, activation relu)
+ maxpooling và AveragePooling (stride , poolsize)
+ Flatten (Không Configuration)
+ batchnorm (Không Configuration) ở cả lớp FC và convolutional
+ activation layer (Chỉ dùng được Relu) ở cả lớp FC và convolutional
+ dense(chỉ dùng được Relu)


~ không hỗ trợ model được viết bằng model.add
**************************************************************************
			KHÔNG HỖ TRỢ MODEL KIỂU NÀY
model_m = Sequential()
model_m.add(Reshape((TIME_PERIODS, num_sensors), input_shape=(input_shape,)))
model_m.add(Conv1D(100, 10, activation='relu', input_shape=(TIME_PERIODS, num_sensors)))
model_m.add(Conv1D(100, 10, activation='relu'))
model_m.add(MaxPooling1D(3))
model_m.add(Conv1D(160, 10, activation='relu'))
model_m.add(Conv1D(160, 10, activation='relu'))
model_m.add(GlobalAveragePooling1D())
model_m.add(Dropout(0.5))
model_m.add(Dense(num_classes, activation='softmax'))
KHÔNG HỖ TRỢ MODEL KIỂU NÀY
***************************************************************************
			CHỈ HỖ TRỢ MODEL KIỂU NÀY
def Lenet15(shape=(32,32,3),classes=10):
    x_input = tf.keras.layers.Input(shape)

    x=Conv2D(20, kernel_size=(5, 5), padding='valid', activation='relu', input_shape=(32, 32, 3))(x_input)
    x=BatchNormalization(axis=3)(x)
    x=MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    x=Flatten()(x)
    x=Dense(20, activation='relu')(x)
    x=BatchNormalization(scale=False)(x)
    x=Activation('relu')(x)
    x=Dropout(rate=0.7)(x)
    x=Dense(10, activation='softmax')(x)
    model = tf.keras.models.Model(inputs = x_input, outputs = x, name = "Lenet15")
    return model


