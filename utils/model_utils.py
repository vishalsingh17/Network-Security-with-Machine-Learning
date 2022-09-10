from os import listdir
from os.path import join
from pickle import dump, load

import xgboost
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.utils import all_estimators

from utils.logger import App_Logger
from utils.read_params import get_log_dic, read_params


class Model_Utils:
    """
    Description :   This class is used for model utility functions required in model training
    Version     :   1.2
    
    Revisions   :   Moved to setup to cloud 
    """

    def __init__(self):
        self.config = read_params()

        self.tuner_kwargs = self.config["model_utils"]

        self.artifact_folder = self.config["dir"]["artifacts"]

        self.trained_models_dir = (
            self.artifact_folder + "/" + self.config["model_dir"]["trained"]
        )

        self.save_format = self.config["save_format"]

        self.log_writer = App_Logger()

    def get_model_score(self, model, test_x, test_y, log_file):
        """
        Method Name :   get_model_score
        Description :   This method gets model score againist the test data

        Output      :   A model score is returned 
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        log_dic = get_log_dic(
            self.__class__.__name__, self.get_model_score.__name__, __file__, log_file
        )

        self.log_writer.start_log("start", **log_dic)

        try:
            model_name = model.__class__.__name__

            preds = model.predict(test_x)

            self.log_writer.log(
                f"Used {model_name} model to get predictions on test data", **log_dic
            )

            self.model_score = roc_auc_score(test_y, preds)

            self.log_writer.log(
                f"ROC AUC score for {model_name} is {self.model_score}", **log_dic
            )

            self.log_writer.start_log("exit", **log_dic)

            return self.model_score

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)

    def get_model_params(self, model, x_train, y_train, log_file):
        """
        Method Name :   get_model_params
        Description :   This method gets the model parameters based on model_key_name and train data

        Output      :   Best model parameters are returned
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        log_dic = get_log_dic(
            self.__class__.__name__, self.get_model_params.__name__, __file__, log_file
        )

        self.log_writer.start_log("start", **log_dic)

        try:
            model_name = model.__class__.__name__

            self.model_param_grid = self.config["train_model"][model_name]

            self.model_grid = GridSearchCV(
                model, self.model_param_grid, **self.tuner_kwargs
            )

            self.log_writer.log(
                f"Initialized {self.model_grid.__class__.__name__}  with {self.model_param_grid} as params",
                **log_dic,
            )

            self.model_grid.fit(x_train, y_train)

            self.log_writer.log(
                f"Found the best params for {model_name} model based on {self.model_param_grid} as params",
                **log_dic,
            )

            self.log_writer.start_log("exit", **log_dic)

            return self.model_grid.best_params_

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)

    def get_base_model(self, model_name, log_file):
        """
        Method Name :   get_base_model
        Description :   This method gets the base model from sklearn lib

        Output      :   base model is returned from sklearn lib
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        log_dic = get_log_dic(
            self.__class__.__name__, self.get_base_model.__name__, __file__, log_file
        )

        self.log_writer.start_log("start", **log_dic)

        try:
            if model_name.lower().startswith("xgb") is True:
                model = xgboost.__dict__[model_name]()

            else:
                model_idx = [model[0] for model in all_estimators()].index(model_name)

                model = all_estimators().__getitem__(model_idx)[1]()

            self.log_writer.log(
                f"Got {model.__class__.__name__} as base model", **log_dic
            )

            self.log_writer.start_log("exit", **log_dic)

            return model

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)

    def get_tuned_model(self, model_name, train_x, train_y, test_x, test_y, log_file):
        """
        Method Name :   get_tuned_model
        Description :   This method tuned the base model based on the training data

        Output      :   Tuned model is returned based on the training data
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        log_dic = get_log_dic(
            self.__class__.__name__, self.get_tuned_model.__name__, __file__, log_file
        )

        self.log_writer.start_log("start", **log_dic)

        try:
            self.model = self.get_base_model(model_name, log_file)

            self.model_best_params = self.get_model_params(
                self.model, train_x, train_y, log_file
            )

            self.log_writer.log(
                f"Got best params for {self.model.__class__.__name__} model", **log_dic
            )

            self.model.set_params(**self.model_best_params)

            self.log_writer.log(
                f"Set the best params for {self.model.__class__.__name__} model",
                **log_dic,
            )

            self.log_writer.log(
                f"Fitting the best parameters for {self.model.__class__.__name__} model",
                **log_dic,
            )

            self.model.fit(train_x, train_y)

            self.log_writer.log(
                f"{self.model.__class__.__name__} model is trained with best parameters",
                **log_dic,
            )

            self.preds = self.model.predict(test_x)

            self.log_writer.log(
                f"Used {self.model.__class__.__name__} model for getting predictions",
                **log_dic,
            )

            self.model_score = self.get_model_score(
                self.model, test_x, test_y, log_file
            )

            self.log_writer.start_log("exit", **log_dic)

            return self.model_score, self.model, self.model.__class__.__name__

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)

    def save_model(self, model, log_file):
        """
        Method Name :   save_model
        Description :   This method saves the trained model to train model folder

        Output      :   Trained model is saved to train model folder
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        log_dic = get_log_dic(
            self.__class__.__name__, self.save_model.__name__, __file__, log_file
        )

        self.log_writer.start_log("start", **log_dic)

        try:
            self.log_writer.log(
                "Starting saving trained model to trained model folder", **log_dic
            )

            model_filename = model.__class__.__name__ + self.save_format

            model_file = join(self.trained_models_dir, model_filename)

            with open(model_file, "wb") as f:
                dump(model, f)

            self.log_writer.log(
                "Saved trained model to trained model folder", **log_dic
            )

            self.log_writer.start_log("exit", **log_dic)

            return "success"

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)

    def load_model(self, model_file, log_file):
        """
        Method Name :   load_model
        Description :   This method loads the model from the particular folder

        Output      :   Trained model is loaded from the particular folder
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        log_dic = get_log_dic(
            self.__class__.__name__, self.load_model.__name__, __file__, log_file
        )

        self.log_writer.start_log("start", **log_dic)

        try:
            self.log_writer.log(f"Loading {model_file} model", **log_dic)

            with open(model_file, "rb") as f:
                model = load(f)

            self.log_writer.log(f"Loaded {model_file} model", **log_dic)

            self.log_writer.start_log("exit", **log_dic)

            return model

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)

    def get_best_model_name(self, lst, log_file):
        """
        Method Name :   get_best_model_name
        Description :   This method gets the best model based on the condition from list of tuple of model name and model score

        Output      :   Best model name is returned from list of tuple of model name and model score
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        log_dic = get_log_dic(
            self.__class__.__name__,
            self.get_best_model_name.__name__,
            __file__,
            log_file,
        )

        self.log_writer.start_log("start", **log_dic)

        try:
            self.log_writer.log(
                "Getting the best model name from list of tuple of model name and model score",
                **log_dic,
            )

            min_score_model_name = max(lst)[2]

            self.log_writer.log(
                "Got the best model name from list of tuple of model name and model score",
                **log_dic,
            )

            self.log_writer.start_log("exit", **log_dic)

            return min_score_model_name

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)

    def get_trained_model_name_list(self, lst, log_file):
        """
        Method Name :   get_trained_model_name_list
        Description :   This method gets the trained model names as list from list of tuple of model name and model score

        Output      :   Trained models names are returned as a list from list of tuple of model name and model score
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        log_dic = get_log_dic(
            self.__class__.__name__,
            self.get_trained_model_name_list.__name__,
            __file__,
            log_file,
        )

        self.log_writer.start_log("start", **log_dic)

        try:
            self.log_writer.log(
                "Getting list of trained model names from list of tuple of model name and model score",
                **log_dic,
            )

            trained_model_lst = [model_key[2] for model_key in lst]

            self.log_writer.log(
                "Got a list of model names from list of tuple of model name and model score",
                **log_dic,
            )

            self.log_writer.start_log("exit", **log_dic)

            return trained_model_lst

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)

    def get_model_file(self, model, stage, log_file):
        """
        Method Name :   get_model_file
        Description :   This method gets model file based on the stage with correct path details

        Output      :   Model file is returned based on the stage with correct path details 
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        log_dic = get_log_dic(
            self.__class__.__name__, self.get_model_file.__name__, __file__, log_file
        )

        self.log_writer.start_log("start", **log_dic)

        try:
            self.log_writer.log("Getting the model file based on the stage", **log_dic)

            model_file = (
                self.config["dir"]["artifacts"]
                + "/"
                + self.config["model_dir"][stage]
                + "/"
                + model
                + self.config["save_format"]
            )

            self.log_writer.log("Got model file based on the stage", **log_dic)

            self.log_writer.start_log("exit", **log_dic)

            return model_file

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)

    def get_prod_model_file(self, log_file):
        """
        Method Name :   get_prod_model_name
        Description :   This method gets the prod model name for getting predictions

        Output      :   Prod model name is returned for getting predictions
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        log_dic = get_log_dic(
            self.__class__.__name__,
            self.get_prod_model_file.__name__,
            __file__,
            log_file,
        )

        self.log_writer.start_log("start", **log_dic)

        try:
            self.log_writer.log("Getting prod model name for prediction", **log_dic)

            prod_model_dir = (
                self.config["dir"]["artifacts"] + "/" + self.config["model_dir"]["prod"]
            )

            model_name = listdir(prod_model_dir)[0].split(".")[0]

            model_name = prod_model_dir + "/" + model_name + self.save_format

            self.log_writer.log("Got the prod model name for prediction", **log_dic)

            self.log_writer.start_log("exit", **log_dic)

            return model_name

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)
