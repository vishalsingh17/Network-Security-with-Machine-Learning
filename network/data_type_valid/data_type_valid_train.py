from network.mongodb_operations.mongo_operations import MongoDB_Operation
from utils.logger import App_Logger
from utils.main_utils import Main_Utils
from utils.read_params import get_log_dic, read_params


class DB_Operation_Train:
    """
    Description :    This class shall be used for handling all the db operations
    
    Version     :   1.2
    Revisions   :   Moved to setup to cloud 
    """

    def __init__(self):
        self.config = read_params()

        self.train_export_csv_file = self.config["export_csv_file"]["train"]

        self.train_input_dir = self.config["train_input_dir"]

        self.good_data_train_dir = self.config["data"]["train"]["good_data_dir"]

        self.train_db_insert_log = self.config["log"]["train_db_insert"]

        self.train_export_csv_log = self.config["log"]["train_export_csv"]

        self.utils = Main_Utils()

        self.mongo = MongoDB_Operation()

        self.log_writer = App_Logger()

    def insert_good_data_as_record(self, good_data_db_name, good_data_collection_name):
        """
        Method Name :   insert_good_data_as_record
        Description :   This method inserts the good data in MongoDB as collection

        Output      :   A MongoDB collection is created with good data present in it
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        log_dic = get_log_dic(
            self.__class__.__name__,
            self.insert_good_data_as_record.__name__,
            __file__,
            self.train_db_insert_log,
        )

        self.log_writer.start_log("start", **log_dic)

        try:
            self.log_writer.log("Inserting dataframes as records in mongodb", **log_dic)

            lst = self.utils.read_csv_from_folder(
                self.good_data_train_dir, self.train_db_insert_log
            )

            [
                self.mongo.insert_dataframe_as_record(
                    f,
                    good_data_db_name,
                    good_data_collection_name,
                    self.train_db_insert_log,
                )
                for f in lst
            ]

            self.log_writer.log(
                "Inserted list of dataframe as collection record in mongodb", **log_dic
            )

            self.log_writer.start_log("exit", **log_dic)

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)

    def export_collection_to_csv(self, good_data_db_name, good_data_collection_name):
        """
        Method Name :   insert_good_data_as_record
        Description :   This method inserts the good data in MongoDB as collection

        Output      :   A csv file stored in input files bucket, containing good data which was stored in MongoDB
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        log_dic = get_log_dic(
            self.__class__.__name__,
            self.export_collection_to_csv.__name__,
            __file__,
            self.train_export_csv_log,
        )

        self.log_writer.start_log("start", **log_dic)

        try:
            self.log_writer.log("Exporting good data collection as csv file", **log_dic)

            df = self.mongo.get_collection_as_dataframe(
                good_data_db_name, good_data_collection_name, self.train_export_csv_log
            )

            self.log_writer.log("Got good data collection as dataframe", **log_dic)

            self.utils.create_directory(self.train_input_dir, self.train_export_csv_log)

            export_f = self.train_input_dir + "/" + self.train_export_csv_file

            df.to_csv(export_f, index=None, header=True)

            self.log_writer.log(
                f"Converted good data collection dataframe to {export_f} csv file name",
                **log_dic,
            )

            self.log_writer.start_log("exit", **log_dic)

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)
