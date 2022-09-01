import pickle
import sys

from sqlalchemy import create_engine
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import re
import nltk

nltk.download(['punkt', 'wordnet', 'stopwords', 'omw-1.4'])


def load_data(database_filepath):
    """
    loads the csv from the db
    :param database_filepath: db file path
    :type database_filepath: str
    :return: X, Y, and y columns
    :rtype: pd.DataFrame, pd.DataFrame, list
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('SELECT * FROM messages', engine)
    data_cols = ['id', 'message', 'original', 'genre']
    X = df[data_cols]
    Y = df.drop(data_cols, axis=1)
    return X, Y, Y.columns


def tokenize(text):
    """
    A function that tokenizes a given text
    :param text: text to tokenize
    :type text: str
    :return: list of tokens
    :rtype: list
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    stemmed = [PorterStemmer().stem(word) for word in tokens]
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words("english")
    clean_tokens = [lemmatizer.lemmatize(word) for word in stemmed if word not in stop_words]
    return clean_tokens


def build_model():
    """
    A function that build a ml pipeline
    :return: GridSearchCV that incldues a pipeline and parameters
    :rtype: GridSearchCV
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('moc', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'moc__estimator__n_estimators': [20, 30, 40],
        'moc__estimator__min_samples_split': [2, 4, 6]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    A function that evaluates a model and prints the result,
    including F1, precision and recall
    :param model: the model
    :type model: Pipeline
    :param X_test: X_test
    :type X_test: pd.DataFrame
    :param Y_test: Y_test
    :type Y_test: pd.DataFrame
    :param category_names: category names
    :type category_names: list
    """
    y_pred = model.predict(X_test['message'])
    y_pred = pd.DataFrame(y_pred, columns=category_names)
    for col in Y_test.columns:
        print(col)
        print(classification_report(Y_test[col], y_pred[col]))


def save_model(model, model_filepath):
    """
    a function that saves the model into a pickle file
    :param model: the model to save
    :type model:  Piepline
    :param model_filepath: filepath of the model
    :type model_filepath: str
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train['message'], Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()