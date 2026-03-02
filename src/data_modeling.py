import pandas as pd
import os
import logging
from xgboost import XGBRegressor
import joblib
from sklearn.model_selection import train_test_split
import yaml

log_dirs = "logs"
os.makedirs(log_dirs,exist_ok=True)

logger = logging.getLogger("data_modeling")
logger.setLevel("DEBUG")

stream_handler = logging.StreamHandler()
stream_handler.setLevel('DEBUG')


log_path = os.path.join(log_dirs,"data_modeling.log")
file_handler = logging.FileHandler(log_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
stream_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(stream_handler)
logger.addHandler(file_handler)

def load_params(params:str)->dict:
    try:
        with open(params,"r") as f:
            pr = yaml.safe_load(f)
        f.close()
        logger.debug("Succesfully loaded params")
        return pr
    except FileNotFoundError as e:
        logger.error("FilenotFound: %s", e)
        raise
    except yaml.YAMLError as e:
        logger.error("Yaml error: %s",e )
        raise
    except Exception as e:
        logger.error("Unexpected error occured during loading params file: %s", e)
        raise
def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.debug("Successfully Loaded the dataset")
        return df
    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        raise
    except pd.errors.EmptyDataError as e:
        logger.error("Empty dataset: %s",e)
        raise
    except Exception as e:
        logger.error("Unexpected error occured: %s",e)
        raise
def test(df:pd.DataFrame)->None:
    try:
        x = df.iloc[:,0:12] 
        Y = df.iloc[:,[12]]
        if x.shape[0] != Y.shape[0]:
            raise ValueError("Invalid Shape")

        x_train,x_test,y_train,y_test = train_test_split(x,Y,test_size=0.1,random_state=12)
        os.makedirs(os.path.join("./data","test"),exist_ok=True)
        x_test.to_csv("data/test/xtest.csv",index=False)
        y_test.to_csv("data/test/ytest.csv",index=False)
        logger.debug("Successfully Saved test data")
    except Exception as e:
        logger.error("Unexpected error occured during saving test data: %s", e)
        raise

def model_training(df:pd.DataFrame, params:dict)-> XGBRegressor:
    try:
        x = df.iloc[:,0:12] 
        Y = df.iloc[:,[12]]
        if x.shape[0] != Y.shape[0]:
            raise ValueError("The Number of samples are not equal in train and test!")
        logger.debug("model training started with %s samples",x.shape[0])       
        x_train,x_test,y_train,y_test = train_test_split(x,Y,test_size=0.1,random_state=12)
        xgb = XGBRegressor(
            n_estimators=params["data_modeling"]["n_estimators"],
            learning_rate=params["data_modeling"]["learning_rate"],
            random_state=params["data_modeling"]["random_state"],
            eval_metric='mae',
            tree_method='hist'
        )
        xgb.fit(x_train,y_train)
        
        logger.debug("Model training completed")
        return xgb
    
    except Exception as e:
        logger.error("Unexpected error occured during model training")
        raise
def save_model(model: XGBRegressor,file_path:str)->None:
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,"wb") as f:
            joblib.dump(model,f)
        f.close()
        logger.debug("Successfully saved model: %s", file_path)
    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error occured during saving the model: %s". e)
        raise


def main():
    try:
        # param = {'n_estimators':500,'learning_rate':0.1,'random_state':34}
        param = load_params(params="params.yaml")
        path = "data/preprocessed/preprocessed.csv"
        df = load_data(path)
        t = test(df)
        model = model_training(df,params=param)
        save_model(model,"models/model.pkl")
        logger.debug("Succesfully excecuted data modeling")
    except Exception as e:
        logger.error("Unexpected error occured during excecuting data modeling")
        raise
if __name__ == '__main__':
    main()



