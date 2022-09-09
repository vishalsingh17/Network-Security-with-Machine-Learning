from os import listdir
from re import match, split
from shutil import copy, move

from pandas import read_csv

from utils.logger import App_Logger
from utils.main_utils import Main_Utils
from utils.read_params import get_log_dic, read_params


class Raw_Train_Data_Validation:
    """
    Description :   This method is used for validating the raw training data
    Written by  :   iNeuron Intelligence
    
    Version     :   1.2
    Revisions   :   Moved to setup to cloud 
    """

    def __init__(self):
        self.config = read_params()

        self.log_writer = App_Logger()

        self.utils = Main_Utils()

        self.raw_train_data_dir = self.config["data"]["raw_data"]["train_batch"]

        self.good_train_data_dir = self.config["data"]["train"]["good_data_dir"]

        self.bad_train_data_dir = self.config["data"]["train"]["bad_data_dir"]

        self.train_schema_file = self.config["schema_file"]["train_schema_file"]

        self.regex_file = self.config["regex_file"]

        self.train_schema_log = self.config["log"]["train_values_from_schema"]

        self.train_gen_log = self.config["log"]["train_general"]

        self.train_name_valid_log = self.config["log"]["train_name_validation"]

        self.train_col_valid_log = self.config["log"]["train_col_validation"]

        self.train_missing_value_log = self.config["log"]["train_missing_values_in_col"]

    def values_from_schema(self):
        """
        Method Name :   values_from_schema
        Description :   This method gets schema values from the schema_training.json file

        Output      :   Schema values are extracted from the schema_training.json file
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        log_dic = get_log_dic(
            self.__class__.__name__,
            self.values_from_schema.__name__,
            __file__,
            self.train_schema_log,
        )

        try:
            self.log_writer.start_log("start", **log_dic)

            dic = self.utils.read_json(self.train_schema_file, self.train_schema_log)

            LengthOfDateStampInFile = dic["LengthOfDateStampInFile"]

            LengthOfTimeStampInFile = dic["LengthOfTimeStampInFile"]

            column_names = dic["ColName"]

            NumberofColumns = dic["NumberofColumns"]

            message = (
                "LengthOfDateStampInFile:: %s" % LengthOfDateStampInFile
                + "\t"
                + "LengthOfTimeStampInFile:: %s" % LengthOfTimeStampInFile
                + "\t "
                + "NumberofColumns:: %s" % NumberofColumns
                + "\n"
            )

            self.log_writer.log(message, **log_dic)

            self.log_writer.start_log("exit", **log_dic)

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)

        return (
            LengthOfDateStampInFile,
            LengthOfTimeStampInFile,
            column_names,
            NumberofColumns,
        )

    def get_regex_pattern(self):
        """
        Method Name :   get_regex_pattern
        Description :   This method gets regex pattern from input files s3 bucket

        Output      :   A regex pattern is extracted
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        log_dic = get_log_dic(
            self.__class__.__name__,
            self.get_regex_pattern.__name__,
            __file__,
            self.train_gen_log,
        )

        try:
            self.log_writer.start_log("start", **log_dic)

            regex = self.utils.read_text(self.regex_file, self.train_gen_log)

            self.log_writer.log(f"Got {regex} pattern", **log_dic)

            self.log_writer.start_log("exit", **log_dic)

            return regex

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)

    def validate_raw_fname(
        self, regex, LengthOfDateStampInFile, LengthOfTimeStampInFile
    ):
        """
        Method Name :   get_regex_pattern
        Description :   This method gets regex pattern from input files s3 bucket

        Output      :   A regex pattern is extracted
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        log_dic = get_log_dic(
            self.__class__.__name__,
            self.validate_raw_fname.__name__,
            __file__,
            self.train_name_valid_log,
        )

        self.log_writer.start_log("start", **log_dic)

        try:
            self.utils.create_dirs_for_good_bad_data("train", self.train_name_valid_log)

            onlyfiles = [f for f in listdir(self.raw_train_data_dir)]

            self.log_writer.log(
                f"Got a list of files from {self.raw_train_data_dir} folder", **log_dic
            )

            for filename in onlyfiles:
                raw_data_train_fname = self.raw_train_data_dir + "/" + filename

                good_data_train_fname = self.good_train_data_dir + "/" + filename

                bad_data_train_fname = self.bad_train_data_dir + "/" + filename

                self.log_writer.log("Created raw,good and bad data file name", **log_dic)

                if match(regex, filename):
                    splitAtDot = split(".csv", filename)

                    splitAtDot = split("_", splitAtDot[0])

                    if len(splitAtDot[1]) == LengthOfDateStampInFile:
                        if len(splitAtDot[2]) == LengthOfTimeStampInFile:
                            copy(raw_data_train_fname, good_data_train_fname)

                        else:
                            copy(raw_data_train_fname, bad_data_train_fname)

                    else:
                        copy(raw_data_train_fname, bad_data_train_fname)

                else:
                    copy(raw_data_train_fname, bad_data_train_fname)

            self.log_writer.start_log("exit", **log_dic)

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)

    def validate_col_length(self, NumberofColumns):
        """
        Method Name :   validate_col_length
        Description :   This method validates the column length based on number of columns as mentioned in schema values

        Output      :   The files' columns length are validated and good data is stored in good data folder and rest is stored in bad data folder
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        log_dic = get_log_dic(
            self.__class__.__name__,
            self.validate_col_length.__name__,
            __file__,
            self.train_col_valid_log,
        )

        self.log_writer.start_log("start", **log_dic)

        try:
            for file in listdir(self.good_train_data_dir):
                fname = self.good_train_data_dir + "/" + file

                csv = read_csv(fname)

                if csv.shape[1] == NumberofColumns:
                    pass

                else:
                    move(fname, self.bad_train_data_dir)

                    self.log_writer.log(
                        f"Invalid Column Length for the {file} file File moved to Bad Raw Folder",
                        **log_dic,
                    )

            self.log_writer.start_log("exit", **log_dic)

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)

    def validate_missing_values_in_col(self):
        """
        Method Name :   validate_missing_values_in_col
        Description :   This method validates the missing values in columns

        Output      :   Missing columns are validated, and good data is stored in good data folder and rest is to stored in bad data folder
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        log_dic = get_log_dic(
            self.__class__.__name__,
            self.validate_missing_values_in_col.__name__,
            __file__,
            self.train_missing_value_log,
        )

        self.log_writer.start_log("start", **log_dic)

        try:
            for file in listdir(self.good_train_data_dir):
                fname = self.good_train_data_dir + "/" + file

                csv = read_csv(fname)

                count = 0

                for columns in csv:
                    if (len(csv[columns]) - csv[columns].count()) == len(csv[columns]):
                        count += 1

                        move(fname, self.bad_train_data_dir)

                        self.log_writer.log(
                            "Invalid Column Length for the {file} file,File moved to Bad Raw Folder",
                            **log_dic,
                        )

                        break

                if count == 0:
                    good_data_fname = self.good_train_data_dir + "/" + file

                    csv.to_csv(good_data_fname, index=None, header=True)

            self.log_writer.start_log("exit", **log_dic)

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)
