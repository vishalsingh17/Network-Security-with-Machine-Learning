from shutil import copy

from utils.logger import App_Logger
from utils.model_utils import Model_Utils
from utils.read_params import get_log_dic, read_params


class Load_Prod_Model:
    def __init__(self):
        self.config = read_params()

        self.log_writer = App_Logger()

        self.model_utils = Model_Utils()

        self.load_prod_model_log = self.config["log"]["load_prod_model"]

    def load_production_model(self, trained_model_list):
        """
        Method Name :   load_production_model
        Description :   This method is responsible for sending the best model to production and rest of the models to staging
        
        Output      :   Best model is pushed to production and rest of the models are pushed to staging
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        log_dic = get_log_dic(
            self.__class__.__name__,
            self.load_production_model.__name__,
            __file__,
            self.load_prod_model_log,
        )

        self.log_writer.start_log("start", **log_dic)

        try:
            self.log_writer.log(
                "Starting pushing best model to production and rest to staging",
                **log_dic,
            )

            best_model = self.model_utils.get_best_model_name(
                trained_model_list, self.load_prod_model_log
            )

            self.log_writer.log(f"Got best model as {best_model} model", **log_dic)

            trained_model_name_list = self.model_utils.get_trained_model_name_list(
                trained_model_list, self.load_prod_model_log
            )

            self.log_writer.log(
                "Got a list of trained model from list of tuple of model score,model and model name",
                **log_dic,
            )

            for model in trained_model_name_list:
                trained_model_file = self.model_utils.get_model_file(
                    model, "trained", self.load_prod_model_log
                )

                self.log_writer.log(f"Got {model} trained model file", **log_dic)

                if model == best_model:
                    prod_model_file = self.model_utils.get_model_file(
                        model, "prod", self.load_prod_model_log
                    )

                    self.log_writer.log(f"Got {model} prod model file", **log_dic)

                    self.log_writer.log(
                        f"Started copying {trained_model_file} to {prod_model_file}",
                        **log_dic,
                    )

                    copy(trained_model_file, prod_model_file)

                    self.log_writer.log(
                        f"Copied {trained_model_file} to {prod_model_file}", **log_dic
                    )

                else:
                    stag_model_file = self.model_utils.get_model_file(
                        model, "stag", self.load_prod_model_log
                    )

                    self.log_writer.log(f"Got {model} stag model file", **log_dic)

                    self.log_writer.log(
                        f"Started copying {trained_model_file} to {stag_model_file}",
                        **log_dic,
                    )

                    copy(trained_model_file, stag_model_file)

                    self.log_writer.log(
                        f"Copied {trained_model_file} to {stag_model_file}", **log_dic
                    )

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)
