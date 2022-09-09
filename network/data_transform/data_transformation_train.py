from os import listdir
from utils.logger import App_Logger
from utils.read_params import get_log_dic, read_params
from pandas import read_csv


class Data_Transform_Train:
    def __init__(self):
        self.config = read_params()

        self.good_data_dir = self.config["data"]["train"]["good_data_dir"]

        self.train_data_transform_log = self.config["log"]["train_data_transform"]

        self.log_writer = App_Logger()

    def add_quotes_to_string_values_in_column(self):
        """
        Method Name :   add_quotes_to_string_values_in_column
        Description :   This method converts all the columns with string datatype such that each value for that column is enclosed
        
        Output      :   A dataframe is returned after applying log1p transformation
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        log_dic = get_log_dic(
            self.__class__.__name__,
            self.add_quotes_to_string_values_in_column.__name__,
            __file__,
            self.train_data_transform_log,
        )

        self.log_writer.start_log("start", **log_dic)

        try:
            self.log_writer.log("Addding quotes to string values in columns", **log_dic)

            onlyfiles = [f for f in listdir(self.good_data_dir)]

            self.log_writer.log("Got a list of files from folder", **log_dic)

            for file in onlyfiles:
                f = self.good_data_dir + "/" + file

                data = read_csv(f)

                self.log_writer.log(f"Read {f} csv file", **log_dic)

                for column in data.columns:
                    count = data[column][data[column] == "?"].count()

                    if count != 0:
                        data[column] = data[column].replace("?", "'?'")

                        self.log_writer.log(
                            "Replacing '?' to " "?" " in the dataframe", **log_dic
                        )

                data.to_csv(f, index=None, header=True)

                self.log_writer.log("Converted dataframe to csv file", **log_dic)

            self.log_writer.log("Added quotes to string values in columns", **log_dic)

            self.log_writer.start_log("exit", **log_dic)

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)
