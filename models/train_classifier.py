import sys
import nltk
nltk.download(['punkt', 'wordnet'])
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
#from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
#from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

from sqlalchemy import create_engine

import pickle


def load_data(database_filepath):
    database_URL = 'sqlite:///{}'.format(database_filepath)
    engine = create_engine(database_URL)
    df = pd.read_sql('Categorized_Tweets', engine)
    engine.dispose()
    X = df['message']
    Y = df.drop(['id','message','original', 'genre'], axis = 1)
    category_names = Y.columns
    
    return X, Y, category_names


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    LSVCpipeline = Pipeline([
                ('vect', CountVectorizer(tokenizer = tokenize)),
                ('tfidf', TfidfTransformer()),
                ('clf', OneVsRestClassifier(LinearSVC()))
    ])
    
    parameters = {
        'clf__estimator__loss': ('hinge', 'squared_hinge'),
        #'clf__estimator__dual': (True, False)
        #'clf__estimator__C': (.01, .1, 1, 10, 100, 1000)
        #'tfidf__norm': ('l1', 'l2', None),
        #'tfidf__use_idf': (True, False),
        #'tfidf__smooth_idf': (True, False)
        #'tfidf__sublinear_tf': (True, False)
        
    }
    
    cv = GridSearchCV(LSVCpipeline, param_grid=parameters, error_score=0)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    Y_pred = pd.DataFrame(Y_pred, index = Y_test.index, columns = category_names)
    print(classification_report(Y_test, Y_pred, target_names= category_names))
    
    #classification = classification_report(Y_test, Y_pred, target_names= category_names, output_dict = True)
    #with open('models/classification_report.pkl', 'wb') as f:
    #    pickle.dump(classification, f, pickle.HIGHEST_PROTOCOL)

def save_model(model, model_filepath):

    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        print("\nBest Parameters:", model.best_params_)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()