import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler,OneHotEncoder,RobustScaler,OrdinalEncoder
from sklearn.impute import SimpleImputer
import os
import json
import joblib
import logging

log_dirs = "logs"
os.makedirs(log_dirs,exist_ok=True)

logger = logging.getLogger("data_preprocessing")
logger.setLevel("DEBUG")

stream_handler = logging.StreamHandler()
stream_handler.setLevel('DEBUG')


log_path = os.path.join(log_dirs,"data_preprocessing.log")
file_handler = logging.FileHandler(log_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
stream_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(stream_handler)
logger.addHandler(file_handler)

def preprocessing(df:pd.DataFrame) -> pd.DataFrame:
    try:
        logger.debug("Preprocesing has started!")
        Categorical_features = df.iloc[:,[0,1,2,5,6,7]]
        oe = OrdinalEncoder()
        ar = oe.fit_transform(Categorical_features)
        os.makedirs(os.path.dirname("models/oe.pkl"),exist_ok=True)
        with open("models/oe.pkl","wb") as f:
            joblib.dump(oe,f)
        f.close()
        df2 = pd.DataFrame(ar,columns=oe.get_feature_names_out(Categorical_features.columns))
        df1 = pd.concat([df2,df.drop(columns=['car_name', 'brand', 'model', 'seller_type', 'fuel_type',
       'transmission_type'])],axis=1)
        #----------------------------------------------------------------------------------
        os.makedirs(os.path.dirname("./data/json/cars"),exist_ok=True)
        os.makedirs(os.path.dirname("./data/json/brands"),exist_ok=True)
        cars = df.groupby(["brand"])["car_name"].unique().reset_index(name="Cars")
        cars_brands = dict()
        brands = dict()
        for i in cars.values:
             cars_brands[i[0]] = i[1].tolist()
             brands[i[0]] = len(i[1].tolist())
        with open("data/json/cars","w") as fd:
            json.dump(cars_brands,fd)
        fd.close()
        with open("data/json/brands","w") as f:
            json.dump(brands,f)
        fd.close()

        #----------------------------------------------------------------------------------
        logger.debug("Succesfully Preprocessed the dataset: %s", df1)
        return df1
    except Exception as e:
        logger.error("Unexpected error occured during preprocessing the dataset: %s", e)
        raise
def main():
    try:
        df = pd.read_csv("data/ingestion/ingested.csv")
        prepro = preprocessing(df)
        path = os.path.join("./data","preprocessed")
        os.makedirs(path,exist_ok=True)
        prepro.to_csv(os.path.join(path,"preprocessed.csv"),index=False)
        logger.debug("Succesfully executed data preprocessing")
    except FileNotFoundError as e:
        logger.error("File not Found: %s", e)
    except pd.errors.EmptyDataError as e:
        logger.error("Empty Data: %s", e)
    except Exception as e:
        logger.error("Unexpected error occured during data preprocessing: %s",e)
        raise
if __name__ == '__main__':
    main()


