from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelBinarizer
from keras.utils import to_categorical
import re
from nltk.corpus import stopwords
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import one_hot, hashing_trick, text_to_word_sequence


class PreprocessingClass(object):
    def __init__(self, df):
        super(PreprocessingClass, self).__init__()
        self.df = df

    def preprocessing_data_hashing_trick(self,
                                         vocab_size=5000,
                                         max_len=100):
        print('Preprocessing data...')

        y = self.df['label'].astype('U')
        data_f = self.df['text'].astype('U')

        data_f = self.clear_text(data_f)
        data = []
        for xt in data_f:
            xt = ' '.join(text_to_word_sequence(xt))
            data.append(hashing_trick(xt.lower(), vocab_size, hash_function='md5'))

        x_train, x_test, y_train, y_test = train_test_split(
            data,
            y,
            test_size=0.2,
            random_state=255)
        x_train = sequence.pad_sequences(x_train, maxlen=max_len)
        x_test = sequence.pad_sequences(x_test, maxlen=max_len)

        print('x_train shape:', x_train.shape)
        print('x_test shape:', x_test.shape)

        return x_train, x_test, y_train, y_test

    def preprocessing_data_one_hot(self,
                                   vocab_size=5000,
                                   max_len=300):
        print('Preprocessing data...')

        y = self.df['label'].astype('U')
        data_f = self.df['text'].astype('U')

        data_f = self.clear_text(data_f)

        data = []
        for xt in data_f:
            xt = ' '.join(text_to_word_sequence(xt.lower()))
            data.append(one_hot(xt.lower(), vocab_size))

        x_train, x_test, y_train, y_test = train_test_split(
            data,
            y,
            test_size=0.2,
            random_state=255)
        x_train = sequence.pad_sequences(x_train, maxlen=max_len)
        x_test = sequence.pad_sequences(x_test, maxlen=max_len)

        print('x_train shape:', x_train.shape)
        print('x_test shape:', x_test.shape)

        return x_train, x_test, y_train, y_test

    def preprocessing_data_tfidf(self,
                                 vocab_size=30000):
        print('Preprocessing data...')

        train_size = int(len(self.df) * .8)

        train_posts = self.df['text'][:train_size].astype('U')
        train_tags = to_categorical(self.df['label'][:train_size].astype('U'))

        test_posts = self.df['text'][train_size:].astype('U')
        test_tags = to_categorical(self.df['label'][train_size:].astype('U'))

        tokenizer = Tokenizer(num_words=vocab_size)
        tokenizer.fit_on_texts(train_posts)

        x_train = tokenizer.texts_to_matrix(train_posts, mode='tfidf')
        x_test = tokenizer.texts_to_matrix(test_posts, mode='tfidf')

        print('x_train shape:', x_train.shape)
        print('x_test shape:', x_test.shape)

        encoder = LabelBinarizer()
        encoder.fit(train_tags)

        y_train = encoder.transform(train_tags)
        y_test = encoder.transform(test_tags)

        return x_train, x_test, y_train, y_test

    def processing_data_texts_to_sequences(self, max_len=300):

        print('Preprocessing data...')
        print("max_len=", max_len)

        y = self.df['label'].astype('U')
        data_f = self.df['text'].astype('U')

        data_f = self.clear_text(data_f)

        t = Tokenizer()
        t.fit_on_texts(data_f)

        encoded_data = t.texts_to_sequences(data_f)

        x_train, x_test, y_train, y_test = train_test_split(
            encoded_data,
            y,
            test_size=0.2,
            random_state=125,
            shuffle=True)

        x_train = sequence.pad_sequences(x_train, maxlen=max_len)
        x_test = sequence.pad_sequences(x_test, maxlen=max_len)

        print('x_train shape:', x_train.shape)
        print('x_test shape:', x_test.shape)

        return x_train, x_test, y_train, y_test, t


    def clear_text(self, text):
        stop_words = set(stopwords.words('english'))

        cleaned_text = []
        for seq in text:
            seq = re.sub(r'\W+', ' ', seq)
            seq = text_to_word_sequence(seq)
            filtered_sentence = [w for w in seq if w not in stop_words]
            cleaned_text.append(" ".join(filtered_sentence))
        return cleaned_text
