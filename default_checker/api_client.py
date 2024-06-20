import pandas as pd
import os
import sys
import requests

def load_test_data(sep=','):
    '''Loading te
    st data to be sent for prediction later on'''

    test_file = os.environ.get('TEST_FILE')
    print(f'Loading test {test_file} (can be changed in .env)')
    return pd.read_json(test_file)

def make_request():
    ''' Makes a POST request on the uvicorn server
    and writes the corresponding prediction results to a .csv file
    also returns the pd.DataFrame corresponding to the results with the uuids'''

    try:
        file = sys.argv[0]
    except IndexError:
        file = None

    data = load_test_data(file)
    uvicorn_uri = os.environ.get('UVICORN_URI')
    data = data.fillna(-99999)
    data_dict = {col : data[col].to_list() for col in data.columns}

    req = requests.post(uvicorn_uri + 'predict',
                           json=data_dict).json() # Weirdest syntax ever

    results = pd.DataFrame(data=req)
    results.to_csv('prediction_results.csv')

    return results

def ping():
    uvicorn_uri = os.environ.get('UVICORN_URI')
    result = requests.get(uvicorn_uri)
    print(result.content)
    return result
