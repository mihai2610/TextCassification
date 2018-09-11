import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn import tree
from sklearn.tree import export_graphviz
from sklearn.metrics import mean_squared_error
from import_data import dataset_2 as ds2


def count_vectorizer(df):

    print("preprocessing data with count_vectorizer ...")

    y = df['label']
    data = df['text'].astype('U')
    x_train, x_test, y_train, y_test = train_test_split(
        data, y, test_size=0.33, random_state=55)

    count_vectorizer = CountVectorizer(analyzer='word',
                                       stop_words='english',
                                       ngram_range=(1, 1))

    count_train = count_vectorizer.fit_transform(x_train)
    # normalizer = Normalizer().fit(count_train)
    # _count_train = normalizer.transform(count_train)

    count_test = count_vectorizer.transform(x_test)
    # normalizer = Normalizer().fit(count_test)
    # _count_test = normalizer.transform(count_test)

    return count_train, y_train, count_test, y_test


def tfidf_vectorizer(df):

    print("preprocessing data with tfidf_vectorizer ...")

    y = df['label']
    data = df['text'].astype('U')
    x_train, x_test, y_train, y_test = train_test_split(
        data, y, test_size=0.33, random_state=55)

    tfidf_vectorizer = HashingVectorizer(analyzer='word',
                                         stop_words='english',
                                         ngram_range=(1, 2),
                                         norm='l2')

    tfidf_train = tfidf_vectorizer.fit_transform(x_train)

    tfidf_test = tfidf_vectorizer.transform(x_test)

    return tfidf_train, y_train, tfidf_test, y_test


def hashing_vectorizer(df):

    print("preprocessing data with hashing_vectorizer ...")

    y = df['label']
    data = df['text'].astype('U')
    x_train, x_test, y_train, y_test = train_test_split(
        data, y, test_size=0.33, random_state=55)

    hash_vectorizer = HashingVectorizer(analyzer='word',
                                        stop_words='english',
                                        ngram_range=(1, 2),
                                        norm='l2')

    hash_train = hash_vectorizer.fit_transform(x_train)

    hash_test = hash_vectorizer.transform(x_test)

    return hash_train, y_train, hash_test, y_test


def multinomial_nb_classifier(x_train, y_train, x_test, y_test):
    classifier = MultinomialNB()
    classifier.fit(x_train, y_train)
    pred = classifier.predict(x_test)

    accuracy = metrics.accuracy_score(y_test, pred)
    confusion = metrics.confusion_matrix(y_test, pred, labels=[0, 1])
    msre = mean_squared_error(y_test, pred)

    print("multinomial_nb_classifier")
    print("accuracy = ", str(accuracy))
    print("confusion = ", confusion)
    print("msre = ", msre)



def k_neighbors_classifier(x_train, y_train, x_test, y_test):
    classifier = KNeighborsClassifier(n_neighbors=7)
    classifier.fit(x_train, y_train)
    pred = classifier.predict(x_test)

    accuracy = metrics.accuracy_score(y_test, pred)
    confusion = metrics.confusion_matrix(y_test, pred, labels=[0, 1])
    msre = mean_squared_error(y_test, pred)

    print("k_n_eighbors_classifier")
    print("accuracy = ", str(accuracy))
    print("confusion = ", confusion)
    print("msre = ", msre)


def linear_svc_classifier(x_train, y_train, x_test, y_test):
    linear_svc = LinearSVC()

    classifier = CalibratedClassifierCV(linear_svc,
                                        method='sigmoid',
                                        cv=3)
    classifier.fit(x_train, y_train)
    pred = classifier.predict(x_test)

    accuracy = metrics.accuracy_score(y_test, pred)
    confusion = metrics.confusion_matrix(y_test, pred, labels=[0, 1])
    msre = mean_squared_error(y_test, pred)

    print("linear_svc_classifier")
    print("accuracy = ", str(accuracy))
    print("confusion = ", confusion)
    print("msre = ", msre)


def random_forest_classifier(x_train, y_train, x_test, y_test):
    classifier = RandomForestClassifier(bootstrap=True,
        class_weight=None, criterion='gini',
        max_depth=12, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=30, n_jobs=4,
            oob_score=False, random_state=1, verbose=2, warm_start=False)

    classifier.fit(x_train, y_train)
    pred = classifier.predict(x_test)

    accuracy = metrics.accuracy_score(y_test, pred)
    confusion = metrics.confusion_matrix(y_test, pred, labels=[0, 1])
    msre = mean_squared_error(y_test, pred)

    print("random_forest_classifier")
    print("accuracy = ", str(accuracy))
    print("confusion = ", confusion)
    print("msre = ", msre)


def sgd_classifier(x_train, y_train, x_test, y_test):
    classifier = SGDClassifier(loss="hinge", penalty="l2",
                               shuffle=True, random_state=255)
    classifier.fit(x_train, y_train)
    pred = classifier.predict(x_test)

    print(pred)

    accuracy = metrics.accuracy_score(y_test, pred)
    confusion = metrics.confusion_matrix(y_test, pred, labels=[0, 1])
    msre = mean_squared_error(y_test, pred)

    print("sgd_classifier")
    print("accuracy = ", str(accuracy))
    print("confusion = ", confusion)
    print("msre = ", msre)



def decision_tree_classifier(x_train, y_train, x_test, y_test):
    classifier = tree.DecisionTreeClassifier(random_state=0)

    classifier.fit(x_train, y_train)
    pred = classifier.predict(x_test)

    accuracy = metrics.accuracy_score(y_test, pred)
    confusion = metrics.confusion_matrix(y_test, pred, labels=[0, 1])
    msre = mean_squared_error(y_test, pred)

    print("decision_tree_classifier")
    print("accuracy = ", str(accuracy))
    print("confusion = ", confusion)
    print("msre = ", msre)


    # export_graphviz(classifier, out_file="tree.dot")


def main():
    print("Loading data ...")
    text = ds2.get_data()

    x_train, y_train, x_test, y_test = count_vectorizer(text)

    # x_train, y_train, x_test, y_test = tfidf_vectorizer(text)

    # x_train, y_train, x_test, y_test = hashing_vectorizer(text)

    multinomial_nb_classifier(x_train, y_train, x_test, y_test)
    k_neighbors_classifier(x_train, y_train, x_test, y_test)
    linear_svc_classifier(x_train, y_train, x_test, y_test)
    random_forest_classifier(x_train, y_train, x_test, y_test)
    sgd_classifier(x_train, y_train, x_test, y_test)
    decision_tree_classifier(x_train, y_train, x_test, y_test)


if __name__ == '__main__':
    main()
