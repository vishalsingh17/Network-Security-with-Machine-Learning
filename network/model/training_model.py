from network.data_ingestion.data_loader_train import Data_Getter_Train
from utils.logger import App_Logger
from utils.read_params import get_log_dic, read_params


class Train_Model:
    def __init__(self):
        self.config = read_params()

        self.model_train_log = self.config["log"]["model_training"]

        self.target_col = self.config["target_col"]

        self.log_writer = App_Logger()

        self.data_getter_train = Data_Getter_Train(self.model_train_log)

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

            self.log_writer.log("Finished model training", **log_dic)

            self.log_writer.start_log("exit", **log_dic)

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)