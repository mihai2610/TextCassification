from gensim.models import KeyedVectors
import KerasPreprocessingClass as CPC
import KerasModelClass as KMC
import numpy as np
from Utils import plotter


class ClassifiersClass(object):

    def __init__(self, df):
        super(ClassifiersClass, self).__init__()
        self.df = df
        self.preproc_methods = CPC.PreprocessingClass(df=df)
        self.my_models = KMC.ModelClass()

    def classifier_simple(self,
                          callbacks=None,
                          dataset=None,
                          model=None,
                          vocab_size=10000,
                          max_len=1000,
                          epochs=5,
                          batch_size=64):

        if dataset is None:
            x_train, x_test, y_train, y_test = \
                self.preproc_methods.\
                    preprocessing_data_hashing_trick(vocab_size=vocab_size,
                                                     max_len=max_len)
        else:
            x_train, x_test, y_train, y_test = dataset

        if model is None:
            model = self.my_models.build_model_CNN2(vocab_size=vocab_size)
        else:
            model = model
        print('Fitting  model...')

        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            callbacks=callbacks,
                            validation_split=0.2)

        print('Evaluate model...')

        mse, acc = model.evaluate(x_test, y_test)

        print("err = ", mse, "acc = ", acc)

        plotter.plot_model_history(history)

    def classifier_KFold(self,
                         model=None,
                         preproc_data=None,
                         max_len=300,
                         k=3,
                         epochs=5,
                         batch_size=64):

        if preproc_data is None:
            x_train, x_test, y_train, y_test = self.preproc_methods.processing_data_texts_to_sequences(max_len=max_len)
        else:
            x_train, x_test, y_train, y_test = preproc_data

        print('Building model...')

        num_val_samples = int(len(x_train) / k)

        if model is None:
            model = self.my_models.build_model_CNN2()
        else:
            model = model

        print('Training model...')

        all_scores = []
        history = []
        for i in range(k):
            val_data = x_train[i * num_val_samples: (i + 1) * num_val_samples]
            val_targets = y_train[i * num_val_samples: (i + 1) * num_val_samples]

            partial_train_data = np.concatenate(
                [x_train[:i * num_val_samples], x_train[(i + 1) * num_val_samples:]],
                axis=0)

            partial_train_targets = np.concatenate(
                [y_train[:i * num_val_samples], y_train[(i + 1) * num_val_samples:]],
                axis=0)

            local_history = model.fit(partial_train_data,
                                      partial_train_targets,
                                      epochs=epochs,
                                      batch_size=batch_size,
                                      validation_split=0.2
                                      )

            history.append(local_history)

            val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=1)
            all_scores.append(np.mean(val_mae))

        val_mse, val_mae = model.evaluate(x_test, y_test, verbose=1)
        all_scores.append(np.mean(val_mae))

        print('all score:', all_scores)
        print('Test accuracy:', np.mean(all_scores))

        plotter.plot_model_history_list(history)

    def classifier_Glove(self,
                         callbacks=None,
                         dataset=None,
                         model=None,
                         vocab_size=10000,
                         max_len=1000,
                         epochs=5,
                         batch_size=64):
        '''
        :param callbacks:
        :param dataset: dataset, need to provide instance for Tokenizer
        :param model: neural network model
        :param vocab_size: size of the vocabulary
        :param max_len: encoding matrix length
        :param epochs: number of epochs
        :param batch_size: size of the batch
        :return:
        '''

        print('Preprocessing data...')

        if dataset is None:
            x_train, x_test, y_train, y_test, t = self.preproc_methods\
                .processing_data_texts_to_sequences(max_len=max_len)
        else:
            x_train, x_test, y_train, y_test, t = dataset

        embedded_model = KeyedVectors \
            .load_word2vec_format('..\GoogleNews-vectors-negative300.bin\GoogleNews-vectors-negative300.bin',
                                  binary=True)

        embedded_matrix = np.zeros((len(t.word_index) + 1, max_len))

        for word, i in t.word_index.items():
            embedding_vector = None
            if word in embedded_model.vocab:
                embedding_vector = embedded_model.get_vector(word)

            if embedding_vector is not None:
                embedded_matrix[i] = embedding_vector

        if model is None:
            model = self.my_models.build_model_CNN_EmbMatrix(
                embedded_matrix=embedded_matrix,
                embedding_size=max_len,
                max_len=max_len,
                max_features=len(t.word_index)+1
                )


        history = model.fit(x_train,
                            y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            callbacks=callbacks,
                            validation_split=0.2
                            )

        print('Evaluate model...')
        mse = model.evaluate(x_test, y_test)

        print("err = ", mse)

        plotter.plot_model_history(history)
