from import_data import dataset_2 as ds2
from KerasModelClass import ModelClass
from KerasPreprocessingClass import PreprocessingClass
from KerasClassifiersClass import ClassifiersClass
from keras.callbacks import TensorBoard


def test_1():
    print("[test 1]")

    batch_size = 64
    epochs = 5
    max_len = 300
    vocab_size = 50000
    labels = 2

    df = ds2.get_data()

    preproc = PreprocessingClass(df)

    classifiers = ClassifiersClass(df=df)

    models = ModelClass()

    data = preproc.preprocessing_data_tfidf(vocab_size=vocab_size)

    model = models.build_model_D4(max_len=vocab_size, input4=labels)

    callbacks = [
        TensorBoard(
            log_dir='my_log_dir_d4',
            histogram_freq=1,
        )
    ]

    classifiers.classifier_simple(callbacks=callbacks,
                                  dataset=data,
                                  model=model,
                                  epochs=epochs,
                                  max_len=max_len,
                                  batch_size=batch_size)


def test_2():
    print("[test 2]")
    max_len = 30000
    vocab_size = 30000
    labels = 2

    df = ds2.get_data()

    preproc = PreprocessingClass(df)

    models = ModelClass()

    classifiers = ClassifiersClass(df=df)

    data = preproc.preprocessing_data_tfidf(vocab_size=vocab_size)

    model = models.build_model_D4(max_len=max_len, input4=labels)

    classifiers.classifier_KFold(model=model,
                                 preproc_data=data,
                                 max_len=max_len,
                                 epochs=3,
                                 k=2)


def test_3():

    print("[test 3]")
    batch_size = 64
    epochs = 5
    max_len = 300

    df = ds2.get_data()

    preproc = PreprocessingClass(df)

    classifiers = ClassifiersClass(df=df)

    dataset = preproc.processing_data_texts_to_sequences(max_len=max_len)

    callbacks = [
        TensorBoard(
            log_dir='my_log_dir_cnn3',
            histogram_freq=1,
            # embeddings_freq=1
        )
    ]

    classifiers.classifier_Glove(callbacks=callbacks,
                                 dataset=dataset,
                                 max_len=max_len,
                                 epochs=epochs,
                                 batch_size=batch_size)


def test_4():
    print("[test 4]")
    batch_size = 64
    epochs = 5
    max_len = 300
    vocab_size = 50000

    df = ds2.get_data()

    preproc = PreprocessingClass(df)
    classifiers = ClassifiersClass(df=df)
    models = ModelClass()

    dataset = preproc.preprocessing_data_one_hot(max_len=max_len, vocab_size=vocab_size)

    model = models.build_model_LSTM2(embedding_size=256)


    # callbacks = [
    #     TensorBoard(
    #         log_dir='my_log_dir_lstm',
    #         histogram_freq=1,
    #         # embeddings_freq=1
    #     )
    # ]

    classifiers.classifier_simple(model=model,
                                  max_len=max_len,
                                  dataset=dataset,
                                  epochs=epochs,
                                  batch_size=batch_size,
                                  vocab_size=vocab_size
                                  )


def test_5():
    print("[test 5]")
    batch_size = 64
    epochs = 5
    max_len = 512
    vocab_size = 70000

    df = ds2.get_data()

    preproc = PreprocessingClass(df)
    classifiers = ClassifiersClass(df=df)
    models = ModelClass()

    data = preproc.preprocessing_data_hashing_trick(vocab_size=vocab_size,
                                                    max_len=max_len)
    model = models.build_model_CNN_LSTM_D(max_features=vocab_size,
                                          maxlen=max_len,
                                          embedding_size=512)

    # callbacks = [
    #     TensorBoard(
    #         log_dir='my_log_dir_cnn_lstm',
    #         histogram_freq=1,
    #     )
    # ]

    classifiers.classifier_simple(dataset=data,
                                  # callbacks=callbacks,
                                  model=model,
                                  epochs=epochs,
                                  batch_size=batch_size,
                                  max_len=max_len,
                                  vocab_size=vocab_size)




if __name__ == '__main__':
    test_5()
