from pandas import read_csv

from utils.logger import App_Logger
from utils.read_params import get_log_dic, read_params


class Data_Getter_Train:
    """
    Description :   This class shall be used for obtaining the df from the input files s3 bucket where the training file is present
    
    Version     :   1.2
    Revisions   :   Moved to setup to cloud 
    """

    def __init__(self, log_file):
        self.config = read_params()

        self.log_file = log_file

        self.train_input_dir = self.config["train_input_dir"]

        self.train_csv_file = self.config["export_csv_file"]["train"]

        self.log_writer = App_Logger()

    def get_data(self):
        """
        Method Name :   get_data
        Description :   This method reads the data from the input files s3 bucket where the training file is stored
        
        Output      :   A pandas dataframe is returned
        On Failure  :   Write an exception log and then raise exception
        
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        log_dic = get_log_dic(
            self.__class__.__name__, self.get_data.__name__, __file__, self.log_file
        )

        self.log_writer.start_log("start", **log_dic)

        try:
            self.log_writer.log("Reading train input csv file", **log_dic)

            f = self.train_input_dir + "/" + self.train_csv_file

            df = read_csv(f)

            self.log_writer.log("Read the train input csv file", **log_dic)

            self.log_writer.start_log("exit", **log_dic)

            return df

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)
