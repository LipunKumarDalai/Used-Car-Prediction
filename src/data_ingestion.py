import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging

#Ensure logs exists

log_dir = "logs"
os.makedirs(log_dir,exist_ok=True)

logger = logging.getLogger("data_ingestion") # define object logger
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler() # handles console output
console_handler.setLevel('DEBUG')

log_path = os.path.join(log_dir,"data_ingestion.log")
file_handler = logging.FileHandler(log_path) #handles file path
file_handler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') #format to show in console/terminal or logs file
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
#add to logger object
logger.addHandler(console_handler) 
logger.addHandler(file_handler)

def load_data(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url)
        logger.debug("Data Loaded from %s", url)
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse csv file: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error occured while loading the data: %s",e)
        raise
def validation_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        required_col = ['car_name', 'brand', 'model', 'vehicle_age', 'km_driven', 'seller_type',
       'fuel_type', 'transmission_type', 'mileage', 'engine', 'max_power',
       'seats', 'selling_price']
        col =[i for i in required_col if i not in df.columns]
        if col:
            logger.error("Missing Columns: %s", col)
            raise ValueError("Missing Columns {col}")
        else:
            df.drop(columns="Unnamed: 0",inplace=True)
            logger.debug("Data Validation completed")
            return df
    except KeyError as e:
         logger.error("Missing column: %s",e)
         raise
    except Exception as e:
         logger.error("Unexpected error occured while validating the data: %s",e)
         raise
def save_data(df:pd.DataFrame, path: str) -> None:
    try:
        dt = os.path.join(path,"ingestion")
        os.makedirs(dt,exist_ok=True)
        df.to_csv(os.path.join(dt,"ingested.csv"),index=False)
        logger.debug("Succesfully saved the data: %s", dt)
    except Exception as e:
        logger.error("Unecpected error occured while saving the data: %s", e)   
        raise
def main():
    try:
        data_path = "https://raw.githubusercontent.com/manishkr1754/CarDekho_Used_Car_Price_Prediction/main/notebooks/data/cardekho_dataset.csv"
        df = load_data(data_path)
        val = validation_data(df)
        save_data(val,"./data")
    except Exception as e:
         logger.error("Unexpected error occured during ingestion: %s", e)
         raise
if __name__=='__main__':
    main()







