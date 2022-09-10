from pandas import DataFrame

from network.data_ingestion.data_loader_prediction import Data_Getter_Pred
from network.data_preprocessing.preprocessing import Preprocessor
from utils.logger import App_Logger
from utils.model_utils import Model_Utils
from utils.read_params import get_log_dic, read_params


class Prediction:
    def __init__(self):
        self.config = read_params()

        self.pred_log = self.config["log"]["pred_main"]

        self.predictions_csv_file = self.config["pred_output_file"]

        self.log_writer = App_Logger()

        self.data_getter_pred = Data_Getter_Pred(self.pred_log)

        self.preprocessor = Preprocessor(self.pred_log)

        self.model_utils = Model_Utils()

    def predict_from_model(self):
        """
        Method Name :   predict_from_model
        Description :   This method is responsible for using the trained model and get predictions based on the prediction data
        
        Output      :   Trained models are used for prediction and results are stored in predictions csv file
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        log_dic = get_log_dic(
            self.__class__.__name__,
            self.predict_from_model.__name__,
            __file__,
            self.pred_log,
        )

        self.log_writer.start_log("start", **log_dic)

        try:
            self.log_writer.log(
                "Started getting predictions based on prediction data", **log_dic
            )

            data = self.data_getter_pred.get_data()

            data = self.preprocessor.replace_invalid_values_with_null(data)

            is_null_present = self.preprocessor.is_null_present(data)

            if is_null_present:
                data = self.preprocessor.impute_missing_values(data)

            prod_model_file = self.model_utils.get_prod_model_file(self.pred_log)

            prod_model = self.model_utils.load_model(prod_model_file, self.pred_log)

            result = list(prod_model.predict(data))

            self.log_writer.log(
                "Used model in production to get predictions", **log_dic
            )

            result = DataFrame(result, columns=["Predictions"])

            self.log_writer.log("Created dataframe for the predictions", **log_dic)

            result.to_csv(self.predictions_csv_file, index=None, header=True)

            self.log_writer.log(
                "Prediction are made using the trained model and results are stored in csv file",
                **log_dic
            )

            self.log_writer.start_log("exit", **log_dic)

            return self.predictions_csv_file, result.head().to_json(orient="records")

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)
