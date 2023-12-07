import pandas as pd
from sklearn.preprocessing import StandardScaler, FunctionTransformer
import category_encoders as ce
from sklearn.pipeline import Pipeline
import warnings
from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

class DataCleaner():
    def __init__(self, data: pd.DataFrame) -> None:
        self.df = data
        self.train_df = data 

    def clean_data(self):
        
        # We need to strip the '$' from the Amount to cast as a float
        self.df["Amount"]=self.df["Amount"].str.replace("$","").astype(float)

        # Extract the hour and minute to perform a more refined time series analysis
        self.df["Hour"] = self.df["Time"].str [0:2]
        self.df["Minute"] = self.df["Time"].str [3:5]
        self.df = self.df.drop(['Time'],axis=1)

        # change the is fraud column to binary 
        self.df["Is Fraud?"] = self.df["Is Fraud?"].apply(lambda x: 1 if x == 'Yes' else 0)
        print("Data cleaning is done")

    def prepare_data_for_training(self, columns_to_select):
        try:
            self.clean_data()
        except Exception:
            pass
        
        def cleaner(df):
            # Convert data type
            df['Hour'] = df['Hour'].astype('float')
            
            # Scale the "Amount" column
            scaler = StandardScaler()
            df['Amount'] = scaler.fit_transform(df[['Amount']])
        
            # Binary encoding for categorical variables
            cat_col = ['Use Chip', 'Day of Week']
            for col in cat_col:
                if col in df.columns:
                    be = ce.BinaryEncoder(drop_invariant=False)
                    enc_df = pd.DataFrame(be.fit_transform(df[col]), dtype='int8')
                    df = pd.concat([df, enc_df], axis=1)
                    df.drop([col], axis=1, inplace=True)
            
            for col in df.columns:
                df[col] = df[col].astype(float)

            return df
        
        self.df = self.df[columns_to_select]
        # Create the pipeline
        preprocessing_pipeline = Pipeline([
            ('cleaning', FunctionTransformer(cleaner, validate=False)), 
        ], verbose=True)

        self.train_df = preprocessing_pipeline.fit_transform(self.df)
            