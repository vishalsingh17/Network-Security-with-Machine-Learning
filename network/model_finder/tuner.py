from sklearn.model_selection import train_test_split

from utils.logger import App_Logger
from utils.main_utils import Main_Utils
from utils.model_utils import Model_Utils
from utils.read_params import get_log_dic, read_params


class Model_Finder:
    """
    Description :   This class shall be used to find the model with best accuracy and AUC score.
    Version     :   1.2
    
    Revisions   :   Moved to setup to cloud 
    """

    def __init__(self, log_file):
        self.log_file = log_file

        self.config = read_params()

        self.split_kwargs = self.config["base"]

        self.model_utils = Model_Utils()

        self.utils = Main_Utils()

        self.log_writer = App_Logger()

    def get_trained_models(self, X_data, Y_data):
        """
        Method Name :   get_trained_models
        Description :   This methods gets the trained models based on training data
        
        Output      :   A list of tuple of model and model score are returned
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        log_dic = get_log_dic(
            self.__class__.__name__,
            self.get_trained_models.__name__,
            __file__,
            self.log_file,
        )

        self.log_writer.start_log("start", **log_dic)

        try:
            models_lst = list(self.config["train_model"].keys())

            x_train, x_test, y_train, y_test = train_test_split(
                X_data, Y_data, **self.split_kwargs
            )

            lst = [
                (
                    self.model_utils.get_tuned_model(
                        model_name,
                        x_train,
                        y_train,
                        x_test,
                        y_test,
                        log_dic["log_file"],
                    )
                )
                for model_name in models_lst
            ]

            return lst

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)

    def train_and_save_models(self, X_data, Y_data):
        """
        Method Name :   train_and_save_models
        Description :   This methods trains and saves all the models based on train data 
        
        Output      :   Models are trained based on training data,saved to respective folders
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        log_dic = get_log_dic(
            self.__class__.__name__,
            self.train_and_save_models.__name__,
            __file__,
            self.log_file,
        )

        self.log_writer.start_log("start", **log_dic)

        try:
            self.utils.create_model_folders(self.log_file)

            lst = self.get_trained_models(X_data, Y_data)

            self.log_writer.log("Got trained models", **log_dic)

            for _, tm in enumerate(lst):
                self.model = tm[1]

                self.model_utils.save_model(self.model, self.log_file)

            self.log_writer.log(
                "Saved and logged all trained models to mlflow", **log_dic
            )

            self.log_writer.start_log("exit", **log_dic)

            return lst

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)
