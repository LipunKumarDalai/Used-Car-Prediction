import pandas as pd
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error,root_mean_squared_error
import logging
import joblib
import json
import os
import yaml
from dvclive import Live
import numpy as np

log_dirs = "logs"
os.makedirs(log_dirs,exist_ok=True)

logger = logging.getLogger("model_evaluation")
logger.setLevel('DEBUG')

stream_handler = logging.StreamHandler()
stream_handler.setLevel("DEBUG")

file = os.path.join(log_dirs,"model_evaluation.log")
file_handler = logging.FileHandler(file)
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
def eval(model,x_test: pd.DataFrame,y_test: pd.DataFrame)->tuple[dict[str,float],np.ndarray]:
    try:
        if x_test.shape[0] != y_test.shape[0]:
            raise ValueError("Invalid test shapes")
        prediction = model.predict(x_test)
        mae = mean_absolute_error(y_test,prediction)
        mse = mean_squared_error(y_test,prediction)
        r2 = r2_score(y_test,prediction)
        rmse = root_mean_squared_error(y_test,prediction)
        metrices = {
            'mae': mae,
            'mse': mse,
            'r2': r2,
            'rmse': rmse
        }
        logger.debug("Succesfully calculated metrices")
        return metrices, prediction
    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error occured while calculated metrices")
        raise
def main():
    try:
        params = load_params("params.yaml")
        path = "models/model.pkl"
        x = pd.read_csv("data/test/xtest.csv")
        y = pd.read_csv("data/test/ytest.csv")
        model = joblib.load(path)
        os.makedirs(os.path.dirname("reports/metrics.json"),exist_ok=True)
        diction, pred = eval(model=model,x_test=x,y_test=y)
        with open("reports/metrics.json","w") as f:
            json.dump(diction,f,indent=4)
        f.close()
        #Experimentation trcking using device!
        with Live(save_dvc_exp=True) as live:
            live.log_metric('mae',mean_absolute_error(y,pred))
            live.log_metric('mse',mean_squared_error(y,pred))
            live.log_metric('r2',r2_score(y,pred))
            live.log_metric('rmse',root_mean_squared_error(y,pred))
            live.log_params(params)

        logger.debug("Successfully Evaluated the model")
    except Exception as e:
        logger.error("Unexpected error occured during evaluating the model: %s", e)
        raise
if __name__ =='__main__':
    main()
