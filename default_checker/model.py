import pandas as pd
import numpy as np

# Data processing and preprocessing
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import make_column_selector, make_column_transformer, ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline

# Pickle i/o
import pickle
import json

### EVALUATING THE MODEL

def load_model():
    """ Simple and dumb, loads the trained model"""
    print('Loading fittest model')
    return pickle.load(open('model.pkl', 'rb'))

def load_preproc():
    '''Simple and dumb, loads the preprocessor'''
    print('Loading (fitted) preprocessor')
    return pickle.load(open('preproc.pkl', 'rb'))

def preprocess_for_evaluation(data : pd.DataFrame,
               preprocessor : ColumnTransformer | Pipeline = None):
    ''' Preprocesses incoming data using an (already fitted)
    preprocessor. Will try to load one if we don't provide
    a preprocessor from scratch'''

    print('Preprocessing data for evaluation')
    if preprocessor is None:
        print('No preprocessor found ... trying to load one')
        preprocessor = load_preproc()

    data = data.drop_duplicates()
    uuids = np.array(data['uuid'])

    if 'uuid' in data.columns:
        data = data.drop(columns=['uuid'])
    if 'name_in_email' in data.columns:
        data = data.drop(columns=['name_in_email'])
    if 'default' in data.columns: # d'uh
        data = data.drop(columns=['default'])

    return preprocessor.transform(data), uuids

def make_prediction(data: pd.DataFrame = None):
    '''Makes a prediction based on an already trained
    model saved as a pickle file.

    Note : it predicts a _probability_, not
    directly the class in question.'''

    print('Working on a prediction')
    model = load_model()

    if data is None:
        print('Reading default test data from test_data.json')
        data = pd.read_json('test_data.json')
    X_to_predict, uuids = preprocess_for_evaluation(data)
    prediction = model.predict_proba(X_to_predict)[:,0]
    output = pd.DataFrame(data={'uuid':uuids, 'pp':prediction})

    return output

### TRAINING THE MODEL

def preprocess_for_train(data : pd.DataFrame):
    ''' Preprocesses incoming data for training purposes
    NB : test data is returned _unpreprocessed_
    AND with a 'uuid' column ! '''

    print('Preprocessing data for training ...')
    data = data.drop_duplicates()
    data = data.drop(columns=['name_in_email'])
    is_test = data['default'].isna()
    data_train, data_test = data[~is_test], data[is_test]
    data_test = data_test.drop(columns=['default'])

    X_train, y_train = data_train.drop(columns=['default', 'uuid']), data_train['default']
    X_rus, y_rus = RandomUnderSampler().fit_resample(X_train, y_train)

    preproc = build_pipeline()
    X_pp = preproc.fit_transform(X_rus)

    return X_pp, data_test, y_rus, preproc  # No y train

def build_pipeline():
    ''' Builds the preprocessing pipeline'''
    is_cat = make_column_selector(dtype_include=object)
    is_num = make_column_selector(dtype_include=np.number)

    num_encoding = make_pipeline(
        SimpleImputer(strategy='median'),
        RobustScaler())

    cat_encoding = make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        OneHotEncoder(handle_unknown='ignore', min_frequency=0.03)
        )

    column_transformer = make_column_transformer(
        (cat_encoding, is_cat),
        (num_encoding, is_num),
        remainder='passthrough'
    )

    return column_transformer

def make_pickles(preprocessor : Pipeline | ColumnTransformer
                 , model:LogisticRegression):
    ''' Makes pickle files out of a sklearn model
    so that we can load the trained model later on
    '''

    print('Saving pickles ...')
    pickle.dump(preprocessor, open('preproc.pkl', 'wb'))
    pickle.dump(model, open('model.pkl', 'wb'))

def save_test_data(data_test:pd.DataFrame):
    '''Saving the test data to make a prediction later'''
    print('Saving test data in JSON')
    data_test.to_json('test_data.json')
    return None

def train_model():
    ''' Trains the model from scratch then saves it as pickle files'''

    data = pd.read_csv('./dataset.csv', sep=';')
    X_rus, data_test, y_rus, preprocessor = preprocess_for_train(data)
    model = LogisticRegression(max_iter = 2000)
    model.fit(X_rus, y_rus)
    make_pickles(preprocessor, model)
    save_test_data(data_test)

    print_dtypes(data)

    return None

def print_dtypes(data : pd.DataFrame):
    for col in data.columns:
        dtype = data[col].dtype
        print(f"    {col} : list[{dtype}]")
