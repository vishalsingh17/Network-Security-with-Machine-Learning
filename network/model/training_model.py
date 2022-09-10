from network.data_ingestion.data_loader_train import Data_Getter_Train
from network.data_preprocessing.preprocessing import Preprocessor
from network.model_finder.tuner import Model_Finder
from utils.logger import App_Logger
from utils.read_params import get_log_dic, read_params


class Train_Model:
    def __init__(self):
        self.config = read_params()

        self.model_train_log = self.config["log"]["model_training"]

        self.target_col = self.config["target_col"]

        self.log_writer = App_Logger()

        self.data_getter_train = Data_Getter_Train(self.model_train_log)

        self.preprocessor = Preprocessor(self.model_train_log)

        self.tuner = Model_Finder(self.model_train_log)

    def training_model(self):
        """
        Method Name :   training_model
        Description :   This method is responsible for applying the preprocessing functions and then train models againist 
                        training data 
        
        Output      :   Models are trained and saved in respective folders
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        log_dic = get_log_dic(
            self.__class__.__name__,
            self.training_model.__name__,
            __file__,
            self.model_train_log,
        )

        self.log_writer.start_log("start", **log_dic)

        try:
            self.log_writer.log("Started model training", **log_dic)

            data = self.data_getter_train.get_data()

            data = self.preprocessor.replace_invalid_values_with_null(data)

            is_null_present = self.preprocessor.is_null_present(data)

            if is_null_present:
                data = self.preprocessor.impute_missing_values(data)

            X, Y = self.preprocessor.separate_label_feature(data, self.target_col)

            Y = self.preprocessor.encode_target_cols(Y)

            lst = self.tuner.train_and_save_models(X, Y)

            self.log_writer.log("Finished model training", **log_dic)

            self.log_writer.start_log("exit", **log_dic)

            return lst

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)
