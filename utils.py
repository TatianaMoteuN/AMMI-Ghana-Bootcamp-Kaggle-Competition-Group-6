import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
import gensim
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

from scipy.sparse import hstack

TESTING_MODE = True


def load_data(train_file_path="data/train.csv", test_file_path = "data/test.csv", is_train=True):
    if is_train:
        return pd.read_csv(train_file_path)
    else:
        train_data = pd.read_csv(train_file_path)
        test_data = pd.read_csv(test_file_path)
        data = pd.concat([train_data, test_data], sort=False)
        data = data.drop("index", axis=1)
        return data


def one_hot_encoding(dataframe, columns):
    return pd.get_dummies(dataframe[columns])


def get_count_vectorizer(dataframe, columns):
    transforms = []
    for col in columns:
        cv = CountVectorizer()
        dataframe[col] = dataframe[col].apply(lambda word: word.lower())
        cv.fit(dataframe)
        transform = cv.transform(dataframe)
        transforms.append(transform)
    return transforms


def load_embeddings_model(path):
    return gensim.models.word2vec.Word2Vec.load(path)


def vectorise(sentance, w2v_model):
    vecs = []
    for word in sentance.lower().split():
        try:
            vecs.append(w2v_model[word])
        except Exception as e:
            print(e)
            vecs.append(np.zeros((300, 1)))
    return sum(vecs)


def text_to_embeddings(data, w2v_model):
    assert isinstance(data, pd.Series)
    data = data.apply(lambda sentence: vectorise(sentence, w2v_model))
    vectors = pd.DataFrame()
    for i in range(300):
        vectors["vec_dim_" + str(i)] = data.apply(lambda v: v[i])

def rmse(y_test, y_pred):
    return np.sqrt(metrics.mean_squared_error(y_test, y_pred))


def export_output_file(y_pred, file_name="out.csv", ids=None):
    if ids is None:
        ids = pd.Series(np.arange(y_pred.shape[0]), name="id")

    result = pd.DataFrame()
    result['id'] = ids
    result["price"] = y_pred
    result.to_csv(file_name, index=False)


def retrain_embeddings_model(wv_model_path, text):
    model = Word2Vec(size=300, min_count=1)
    model.build_vocab(text)
    total_examples = model.corpus_count
    pretrained_model = KeyedVectors.load_word2vec_format(wv_model_path, binary=True)
    model.build_vocab([list(pretrained_model.vocab.keys())], update=True)
    model.intersect_word2vec_format(wv_model_path, binary=True, lockf=1.0)
    model.train(text, total_examples=total_examples, epochs=model.iter)
    model.save("data/updated_model")
    return model


def train_kfold(model, X, y, k=5, n_jobs= 8):
    kfold = KFold(k, True, 1)
    scores = []
    fold_count = 0
    for train, test in kfold.split(X):
        print("Training fold number : ", fold_count)
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        model = model.fit(X_train, y_train, n_jobs=n_jobs)
        y_pred = model.predict(X_test)
        scores.append(rmse(y_test, y_pred))
        print("RMSE for fold number {0} = {1}".format(fold_count, scores[fold_count]))
        fold_count += 1

    return model, scores


def train_grid_search(model, X, y, params):
    grid_search = GridSearchCV(model, params,)
    grid_search.fit(X, y)
    return grid_search


def remove_nan(data):
    def substitute(value):
        if str(value) == "nan":
            return "missing"
        else:
            return value
    for col in data.columns:
        data[col] = data[col].apply(lambda v: substitute(v))
    return data