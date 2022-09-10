from network.data_transform.data_transformation_pred import Data_Transform_Pred
from network.data_type_valid.data_type_valid_pred import DB_Operation_Pred
from network.raw_data_validation.pred_data_validation import \
    Raw_Pred_Data_Validation
from utils.logger import App_Logger
from utils.read_params import get_log_dic, read_params


class Pred_Validation:
    """
    Description :   This class is used for validating all the prediction batch files
    Written by  :   iNeuron Intelligence
    
    Version     :   1.2
    Revisions   :   Moved to setup to cloud 
    """

    def __init__(self):
        self.config = read_params()

        self.log_writer = App_Logger()

        self.pred_main_log = self.config["log"]["pred_main"]

        self.good_data_db_name = self.config["mongodb"]["network_db_name"]

        self.good_data_collection_name = self.config["mongodb"][
            "network_pred_data_collection"
        ]

        self.raw_data = Raw_Pred_Data_Validation()

        self.data_transform = Data_Transform_Pred()
        
        self.db_operation = DB_Operation_Pred()

    def pred_validation(self):
        """
        Method Name :   pred_validation
        Description :   This method is responsible for converting raw data to cleaned data for prediction
        
        Output      :   Raw data is converted to cleaned data for prediction
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        log_dic = get_log_dic(
            self.__class__.__name__,
            self.pred_validation.__name__,
            __file__,
            self.pred_main_log,
        )

        self.log_writer.start_log("start", **log_dic)

        try:
            self.log_writer.log("pred Raw Validation started", **log_dic)

            (
                LengthOfDateStampInFile,
                LengthOfTimeStampInFile,
                _,
                noofcolumns,
            ) = self.raw_data.values_from_schema()

            regex = self.raw_data.get_regex_pattern()

            self.raw_data.validate_raw_fname(
                regex, LengthOfDateStampInFile, LengthOfTimeStampInFile,
            )

            self.raw_data.validate_col_length(NumberofColumns=noofcolumns)

            self.raw_data.validate_missing_values_in_col()

            self.log_writer.log("Pred Raw Data Validation completed", **log_dic)

            self.log_writer.log("Starting Data Transformation", **log_dic)

            self.data_transform.add_quotes_to_string_values_in_column()

            self.log_writer.log("Data Transformation completed !!", **log_dic)

            self.log_writer.log("Train Data Type Validation started", **log_dic)

            self.db_operation.insert_good_data_as_record(
                self.good_data_db_name, self.good_data_collection_name
            )

            self.db_operation.export_collection_to_csv(
                self.good_data_db_name, self.good_data_collection_name
            )

            self.log_writer.log("Train Data Type Validation completed", **log_dic)

            self.log_writer.start_log("exit", **log_dic)

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)
