from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras import regularizers

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten


class ModelClass(object):

    def __init__(self):
        super(ModelClass, self).__init__()

    def build_model_D4(self,
                      max_len=30000,
                      input1=1024,
                      input2=1024,
                      input3=1024,
                      input4=1,
                      dropout1=0.6,
                      dropout2=0.6,
                      dropout3=0.6,
                      activation1="tanh",
                      activation2="tanh",
                      activation3="tanh",
                      activation4="sigmoid",
                      optimizer="adam",
                      loss="binary_crossentropy"):

        '''
            Neural network composed by 4 densely connected layers

        :return: compiled model
        '''
        print(" max_len=", max_len,
              " input1=", input1,
              " input2=", input2,
              " input3=", input3,
              " input4=", input4,
              " dropout1=", dropout1,
              " dropout2=", dropout2,
              " dropout3=", dropout3)

        print(" activation1=", activation1,
              " activation2=", activation2,
              " activation3=", activation3,
              " activation4=", activation4,
              " optimizer=", optimizer,
              " loss=", loss)

        model = Sequential()
        model.add(Dense(input1,
                        activation=activation1,
                        kernel_regularizer=regularizers.l2(0.01),
                        input_shape=(max_len,)))
        model.add(Dropout(dropout1))
        model.add(Dense(input2,
                        activation=activation2,
                        kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dropout(dropout2))
        model.add(Dense(input3,
                        activation=activation3,
                        kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dropout(dropout3))
        model.add(Dense(input4, activation=activation4))
        model.summary()

        model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=['accuracy'],
                      )
        return model

    def build_model_CNN_LSTM_D(self,
                      max_features=20000,
                      maxlen=100,
                      embedding_size=128,
                      kernel_size=5,
                      filters=128,
                      pool_size=4,
                      lstm_output_size=70):
        '''
            Neural network composed by CNN and LSTM layers
        :param max_features:
        :param maxlen:
        :param embedding_size:
        :param kernel_size:
        :param filters:
        :param pool_size:
        :param lstm_output_size:
        :return:
        '''

        print("Building model CNN_LSTM_D")

        print("\n max_features=", max_features,
            "\n maxlen=", maxlen,
            "\n embedding_size=", embedding_size,
            "\n kernel_size=", kernel_size,
            "\n filters=", filters,
            "\n pool_size=", pool_size,
            "\n lstm_output_size=", lstm_output_size)

        model = Sequential()
        model.add(Embedding(max_features,
                            embedding_size,
                            input_length=maxlen))
        model.add(Dropout(0.7))
        model.add(Conv1D(filters,
                         kernel_size,
                         padding='valid',
                         activation='relu',
                         kernel_regularizer=regularizers.l2(0.05)
                         ))
        model.add(Dropout(0.7))
        model.add(MaxPooling1D(pool_size=pool_size))
        model.add(Dropout(0.7))
        model.add(LSTM(lstm_output_size, kernel_regularizer=regularizers.l2(0.05)))
        model.add(Dropout(0.7))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.summary()
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model

    def build_model_CNN2(self,
                      max_features=100000,
                      embedding_size=256,
                      kernel_size1=2,
                      kernel_size2=2,
                      filters1=128,
                      filters2=128,
                      pool_size=2,
                      vocab_size=10000):
        print("creating model 0")

        print("max_features=", max_features,
              "embedding_size=", embedding_size,
              "kernel_size1=", kernel_size1,
              "kernel_size2=", kernel_size2,
              "filters1=", filters1,
              "filters2=", filters2,
              "pool_size=", pool_size,
              "vocab_size=", vocab_size
              )
        model = Sequential()
        model.add(Embedding(max_features,
                            embedding_size,
                            input_length=vocab_size))
        model.add(Conv1D(filters1,
                         kernel_size1,
                         activation='tanh',
                         input_shape=(None, vocab_size),
                         padding='valid',
                         strides=2))
        model.add(Dropout(0.5))
        model.add(Conv1D(filters2,
                         kernel_size2,
                         activation='tanh',
                         input_shape=(None, vocab_size)))
        model.add(Dropout(0.5))
        model.add(MaxPooling1D(pool_size=pool_size))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        return model

    def build_model_LSTM2(self,
                          max_features=30000,
                          embedding_size=128,
                          max_len=300,
                          lstm_output_size1=256,
                          lstm_output_size2=128,
                          dropout1=0.8,
                          dropout2=0.7,
                          reg_l2_1=0.04,
                          reg_l2_2=0.02):

        print("model LSTM")

        print("\n max_features=", max_features,
              "\n embedding_size=", embedding_size,
              "\n max_len=", max_len,
              "\n lstm_output_size1=", lstm_output_size1,
              "\n lstm_output_size2=", lstm_output_size2,
              "\n dropout1=", dropout1,
              "\n dropout2=", dropout2,
              "\n reg_l2_1=", reg_l2_1,
              "\n reg_l2_2=", reg_l2_2)

        model = Sequential()
        model.add(Embedding(max_features,
                            embedding_size,
                            input_length=max_len))
        model.add(LSTM(lstm_output_size1,
                       kernel_regularizer=regularizers.l2(reg_l2_1),
                       return_sequences=True))
        model.add(Dropout(dropout1))
        model.add(LSTM(lstm_output_size2,
                       kernel_regularizer=regularizers.l2(reg_l2_2)))
        model.add(Dropout(dropout2))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.summary()

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model

    def build_model_CNN_EmbMatrix(self,
                        max_features=100000,
                        embedding_size=512,
                        kernel_size1=3,
                        kernel_size2=4,
                        filters1=256,
                        filters2=256,
                        pool_size=2,
                        max_len=300,
                        embedded_matrix=None):
        print("creating model 0")

        print(" max_features=", max_features,
              "\n embedding_size=", embedding_size,
              "\n kernel_size1=", kernel_size1,
              "\n kernel_size2=", kernel_size2,
              "\n filters1=", filters1,
              "\n filters2=", filters2,
              "\n pool_size=", pool_size,
              "\n max_len=", max_len
              )
        model = Sequential()
        model.add(Embedding(max_features,
                            embedding_size,
                            weights=[embedded_matrix],
                            input_length=max_len)
                  )
        model.add(Dropout(0.5))
        model.add(Conv1D(filters1,
                         kernel_size1,
                         activation='tanh',
                         kernel_regularizer=regularizers.l2(0.15),
                         input_shape=(None, max_len),
                         padding='valid')
                  )
        model.add(Dropout(0.8))
        model.add(Conv1D(filters2,
                         kernel_size2,
                         kernel_regularizer=regularizers.l2(0.15),
                         activation='tanh',
                         input_shape=(None, max_len),
                         padding='valid'
                         )
                  )
        model.add(Dropout(0.8))
        model.add(MaxPooling1D(pool_size=pool_size))
        model.add(Dropout(0.6))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        return model
