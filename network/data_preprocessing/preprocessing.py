import numpy as np
from pandas import DataFrame
from sklearn.impute import KNNImputer

from utils.logger import App_Logger
from utils.read_params import get_log_dic, read_params


class Preprocessor:
    def __init__(self, log_file):
        self.log_file = log_file
        
        self.config = read_params()

        self.null_values_file = self.config["null_values_csv_file"]

        self.knn_params = self.config["knn_imputer"]

        self.log_writer = App_Logger()

    def replace_invalid_values_with_null(self, data):
        log_dic = get_log_dic(
            self.__class__.__name__,
            self.replace_invalid_values_with_null.__name__,
            __file__,
            self.log_file,
        )

        self.log_writer.start_log("start", **log_dic)

        try:
            self.log_writer.log(
                "Started replacing invalid values in the dataframe", **log_dic
            )

            for column in data.columns:
                count = data[column][data[column] == "?"].count()

                if count != 0:
                    data[column] = data[column].replace("?", np.nan)

            self.log_writer.log(
                "Replaced invalid values with null in the dataframe", **log_dic
            )

            self.log_writer.start_log("exit", **log_dic)

            return data

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)

    def separate_label_feature(self, data, label_col_name):
        """
        Method Name :   separate_label_feature
        Description :   This method separates the features and a Label Coulmns.
        
        Output      :   Returns two separate dataframes, one containing features and the other containing labels .
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        log_dic = get_log_dic(
            self.__class__.__name__,
            self.separate_label_feature.__name__,
            __file__,
            self.log_file,
        )

        self.log_writer.start_log("start", **log_dic)

        try:
            self.log_writer.log("Separating label column from the dataframe", **log_dic)

            self.X = data.drop(labels=label_col_name, axis=1)

            self.Y = data[label_col_name]

            self.log_writer.log(
                f"Separated {label_col_name} from {data}", **log_dic
            )

            self.log_writer.start_log("exit", **log_dic)

            return self.X, self.Y

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)

    def is_null_present(self, data):
        """
        Method Name :   is_null_present
        Description :   This method checks whether there are null values present in the pandas Dataframe or not.
        
        Output      :   If null values are present in the dataframe, a csv file is created and then uploaded back to input files bucket
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        log_dic = get_log_dic(
            self.__class__.__name__,
            self.is_null_present.__name__,
            __file__,
            self.log_file,
        )

        self.log_writer.start_log("start", **log_dic)

        try:
            self.log_writer.log(
                "Checking whether null values are present in the dataframe", **log_dic
            )

            null_present = False

            cols_with_missing_values = []

            cols = data.columns

            self.null_counts = data.isna().sum()

            self.log_writer.log(
                f"Got null value count as {self.null_counts}", **log_dic
            )

            self.log_writer.log(f"Null values count is : {self.null_counts}", **log_dic)

            for i in range(len(self.null_counts)):
                if self.null_counts[i] > 0:
                    null_present = True

                    cols_with_missing_values.append(cols[i])

            self.log_writer.log("Created cols with missing values", **log_dic)

            if null_present is True:
                self.log_writer.log(
                    "Null values were found the columns...preparing dataframe with null values",
                    **log_dic,
                )

                self.null_df = DataFrame()

                self.null_df["columns"] = data.columns

                self.null_df["missing values count"] = np.asarray(data.isna().sum())

                self.log_writer.log("Created dataframe with null values", self.log_file)

                self.null_df.to_csv(self.null_values_file, index=None, header=True)

                self.log_writer.log(
                    "Converted null values dataframe to null values csv file", **log_dic
                )

            else:
                self.log_writer.log(
                    "No null values are present in cols. Skipped the creation of dataframe",
                    **log_dic,
                )

            self.log_writer.start_log("exit", **log_dic)

            return null_present

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)

    def impute_missing_values(self, data):
        """
        Method Name :   impute_missing_values
        Description :   This method replaces all the missing values in the dataframe using KNN imputer
        
        Output      :   A dataframe which has all the missing values imputed.
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        log_dic = get_log_dic(
            self.__class__.__name__,
            self.impute_missing_values.__name__,
            __file__,
            self.log_file,
        )

        self.log_writer.start_log("start", **log_dic)

        self.data = data

        try:
            self.log_writer.log("Imputing missing values in the dataframe", **log_dic)

            imputer = KNNImputer(missing_values=np.nan, **self.knn_params)

            self.log_writer.log(f"Initialized {imputer.__class__.__name__}", **log_dic)

            self.new_array = imputer.fit_transform(self.data)

            self.new_data = DataFrame(data=self.new_array, columns=self.data.columns)

            self.log_writer.log("Created new dataframe with imputed values", **log_dic)

            self.log_writer.log("Imputing missing values Successful", **log_dic)

            self.log_writer.start_log("exit", **log_dic)

            return self.new_data

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)
