import function_prj

import logging
import joblib
import sys
import pandas as pd
from fastapi import FastAPI, Response, status
from pydantic import BaseModel
from datetime import datetime

# Initialize FastAPI app
app = FastAPI()

# To check if the server is up
@app.get('/status')
def health_check():
    return "Success!"

def load_model_files():
    """
    Loads the model and the label encoding dictionary
    """
    obj_watches = joblib.load('obj_watches.pkl')                                                            #

    return obj_watches

def check_request_data(request_data):
    """
    Checks if the request data contains values that the model has previously been exposed to
    """
    # Implement this (homework)
    check_status = 1
    error_message = ''
    
    return check_status, error_message

def preprocessing(request_data):
    """
    Preprocessing the request data and converting it into a dataframe
    """
    model_data = request_data.copy()
    
    return model_data['User_ID']

# Request Body
class RequestBody(BaseModel):
    User_ID : str

# To predict the credit worthiness of the user
@app.post('/api/Recomend_Watches: ')
def Recomend_Watches(request_body: RequestBody):
    
    # Convert request_body to data
    request_data = request_body.dict()
    
    # Create response
    response_data = {}
    
    # Check for data sanity
    check_status, error_message = check_request_data(obj_watches)                                              #
    
    if check_status == 0:
        response.status_code = status.HTTP_400_BAD_REQUEST
        logging.info(error_message)
        response_data['error'] = error_message
    else:
        # Dataframe creation
        tick = datetime.now()
        logging.info('Preprocessing request data...')
        input_str = preprocessing(request_data)
        tock = datetime.now()
        diff = str(int((tock - tick).total_seconds() * 1000))
        logging.info('Preprocessed request data in ' + diff + ' ms!')

        # Prediction
        tick = datetime.now()
        logging.info('Predicting credit worthiness...')
        
        recomendations = function_prj.get_recom_watches(input_str)
        
        for i in range(1, 6):
            
            response_data['Recomendation {} '.format(i)] = recomendations.iloc[[i], [1]]['product/title']
        
        tock = datetime.now()
        diff = str(int((tock - tick).total_seconds() * 1000))
        logging.info('Predicted credit worthiness in ' + diff + ' ms!')
        
    return response_data

# Set up logging
root = logging.getLogger()
root.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)        

obj_watches = load_model_files()