import configparser
import mysql.connector
import numpy as np
import pickle
from pprint import pprint
import git


def read_db_config(filename="config.ini", section="mysql"):
    """Read database configuration file and return a dictionary object

    Args:
        filename (str, optional): name of the configuration file
        section (str, optional): section of database configuration

    Returns:
        dict: Dictionary of database parameters

    Raises:
        Exception: config section not found
    """

    parser = configparser.ConfigParser()
    parser.read(filename)

    db = {}
    if parser.has_section(section):
        items = parser.items(section)
        for item in items:
            db[item[0]] = item[1]
    else:
        raise Exception("{0} not found in the {1} file".format(section, filename))

    return db


def get_git_commit_sha():
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    return sha


class TestResultsDatabase(object):
    """Store and fetch test results from a database.

    The database can be created using the misc/create_database.sql file in this repo.
    The credentials to the database should be stored in a config.ini file in the
    root directory of this project. The file structure of the config.ini file
    must be:

        [mysql]
        host = <host>
        database = model_results
        user = <db user>
        password = <db password>

    The data can be stored and accessed using ``store()`` and ``get()``. Please see
    the docstring for instructions.

    """

    def __init__(self, config="config.ini", verbose=0):
        self.verbose = verbose
        self.config = config
        self.connection = None
        self.cursor = None

    def connect(self, config=None):
        """Connect to database
        """
        config = self.config if config is None else config
        db_config = read_db_config(filename=config)
        conn = None
        try:
            if self.verbose > 0:
                print("[Info] Connecting to MySQL database...")
            conn = mysql.connector.MySQLConnection(**db_config)

            if self.verbose > 0:
                if conn.is_connected():
                    print("[Info] Connection established.")
                else:
                    print("[Error] Connection failed.")

        except mysql.connector.Error as error:
            print(error)

        self.connection = conn
        self.cursor = self.connection.cursor()
        return True

    def close(self):
        """Close connection to database
        """
        if self.connection is not None and self.connection.is_connected():
            self.connection.commit()
            self.connection.close()
            if self.verbose > 0:
                print("[Info] Connection closed.")

    def store(
        self,
        data: np.ndarray,
        state_dict: dict,
        region: str,
        batch_size: int,
        dropout: float,
        epochs: int,
        lr: float,
        normalize: bool,
        optim: str,
        ordinal: bool,
        ratio_transform: bool,
        use_seed: bool,
        timestamp_start: int,
        window_size: int,
        is_timestamps: bool = False,
        is_validation: bool = False,
        normalize_type: int = None,
        ordinal_resolution: int = None,
        seed: int = None,
        use_harmonics: bool = None,
        spp_resolution: int = None,
        lr_scheduler: bool = None,
        target: bool = False,
        verbose: bool = None,
        cuda: bool = None,
        shuffle: bool = None,
        master: bool = True,
        prosjektoppgave: bool = False,
        is_lightgbm: bool = False,
        is_hybrid: bool = False,
        timestamp_end: int = None,
        data_length: int = None,
        git_commit: str = None,
        comment: str = None,
        **other,
    ):
        """Store data in the database for later retraival.

        Args:
            data (np.ndarray): Data to be stored. This can either be the target
                values or the estimated values. If target values are stored remember
                to pass target=True as an argument to this function.
            state_dict (dict): not in use.
            region (str): Region of the relevant time series. This can be NO1, NO2,
                ..., EONon, EONoff, etc.
            batch_size (int): Batch size used for training.
            dropout (float): Dropout probability for the last two layers.
            epochs (int): Number of epochs during training.
            lr (float): Initial learning rate.
            normalize (bool): Normalize the data. This is usually True for neural net,
                and false for random forest and equivalent ensamble learners.
            optim (str): Optimizer used during training.
            ordinal (bool): Use ordinal classification as target.
            ratio_transform (bool): Transform ratio to a more even distribution before
                training. This seems to have a positive effect on the accuracy of the
                model.
            use_seed (bool): Use a seed for random values.
            timestamp_start (int): First timestamp of the value in the series that is
                to be stored. The region+timestamp information is enough to identify
                the production value. The series is assumed to have an hourly
                resolution, so the end timestamp should be deterministic given the
                lenght of the dataset that is stored.
            window_size (int): Number of hours before (and after) the target hour that
                is included in the model. Usually set to 2 for the advanced model.
            is_timestamps (bool, optional): Set to True if dats is timestamps
            is_validation (bool, optional): Set to True if data is validation data
            normalize_type (int, optional): Normalization type. See report for info.

                =====  ===============  ==========================================
                value  type             description
                =====  ===============  ==========================================
                1      Global mean/std  For each parameter, take the global mean
                                        and std and calculate(values−mean)/std.
                2      Local mean/std   For each grid location of each parameter,
                                        take the mean and stdof that location and
                                        calculate (values−mean)/std
                3      Global [0, 1]    For each parameter, map the range to the
                                        interval [0,1] by calculating
                                        (values−min)/(max−min)
                4      Local [0, 1]     For each grid location of each parameter,
                                        map the local range tothe interval [0,1]
                                        by calculating (values−min)/(max−min)
                =====  ===============  ==========================================

            ordinal_resolution (int, optional): Number of ordinal classes that is used.
            seed (int, optional): Random seed that is used.
            use_harmonics (bool, optional): True if seasonal harmonics are used
            spp_resolution (int, optional): Resolution for the Spatial Pyramid Pooling
                layer. Default 5.
            lr_scheduler (bool, optional): Is a learningrate scheduler used during
                training.
            target (bool, optional): Flag to determine if the data stored is
                target values or estimated values from the model
            verbose (bool, optional): Training was done using the verbose flag. Not
                a very useful parameter but needed to be able to use argparse output
                directly with the function.
            cuda (bool, optional): Training was done using cuda. Not a very useful
                parameter
                but needed to be able to use argparse output directly with the
                function.
            shuffle (bool, optional): Training was done using dataset shuffle.
            master (bool, optional): This test data from a model that I developed for
                my master thesis.
            prosjektoppgave (bool, optional): This test data from a model that I
                developed for my fall project.
            is_lightgbm (bool, optional): This test data is from a lightgbm model
            is_hybrid (bool, optional): This test data is from a hybrid between the
                lightgbm model and the
                CNN model.
            timestamp_end (int, optional): Timestamp at the end of the dataset. This
                should be deterministic given the length of the dataset and hourly
                resolution.
            data_length (int, optional): Length of the dataset.
            git_commit (str, optional): Current git HEAD.
            comment (str, optional): Comment appended to data.

            **other: other parameters
        """

        arguments = {
            "region": region,
            "master": master,
            "prosjektoppgave": prosjektoppgave,
            "lightgbm": is_lightgbm,
            "hybrid": is_hybrid,
            "batch_size": batch_size,
            "dropout": dropout,
            "epochs": epochs,
            "lr": lr,
            "normalize": normalize,
            "normalize_type": normalize_type,
            "optim": optim,
            "ordinal": ordinal,
            "ordinal_resolution": ordinal_resolution,
            "ratio_transform": ratio_transform,
            "seed": seed,
            "use_seed": use_seed,
            "window_size": window_size,
            "harmonics": use_harmonics,
            "spp_resolution": spp_resolution,
            "lr_scheduler": lr_scheduler,
            "target": target,
            "cuda": cuda,
            "shuffle": shuffle,
            "timestamp_start": timestamp_start,
            "timestamp_end": timestamp_end,
            "data_length": data_length,
            "git_commit": git_commit,
            "comment": comment,
            "is_timestamps": is_timestamps,
            "is_validation": is_validation,
        }

        if type(data) is not np.ndarray:
            dtype = type(data)
            print(f"error: data type is {dtype}. expected <class 'numpy.ndarray'>")
            return False

        if git_commit == None:
            git_commit = get_git_commit_sha()

        arguments["data_length"] = len(data)
        arguments["git_commit"] = git_commit

        # do not store new data if target already exists
        if target == True:
            ret = self.get(**arguments)
            if len(ret) > 0:
                return False

        # do not store new data if timestamps
        if is_timestamps == True:
            ret = self.get(**arguments)
            if len(ret) > 0:
                return False

        # insert data first
        if is_timestamps == True:
            a = {"data": pickle.dumps(data)}
        else:
            a = {"data": pickle.dumps(data.astype(np.float32))}
        sql = "INSERT INTO data (data) VALUES (%(data)s)"
        self.cursor.execute(sql, a)
        last_id = self.cursor.lastrowid

        arguments["data_id"] = last_id

        # generate sql
        sql = "INSERT INTO results ("
        for key in arguments.keys():
            sql += f"{key}, "
        sql = sql[:-2] + ") VALUES ("
        for key in arguments.keys():
            sql += f"%({key})s, "
        sql = sql[:-2] + ")"

        self.cursor.execute(sql, arguments)
        self.connection.commit()
        return True

    def get(
        self,
        with_state_dict: bool = False,
        region: str = None,
        master: bool = None,
        prosjektoppgave: bool = None,
        lightgbm: bool = None,
        hybrid: bool = None,
        batch_size: int = None,
        dropout: float = None,
        epochs: int = None,
        lr: float = None,
        normalize: bool = None,
        normalize_type: int = None,
        optim: str = None,
        ordinal: bool = None,
        ordinal_resolution: int = None,
        ratio_transform: bool = None,
        seed: int = None,
        use_seed: bool = None,
        window_size: int = None,
        harmonics: bool = None,
        spp_resolution: int = None,
        lr_scheduler: bool = None,
        target: bool = None,
        verbose: bool = None,
        cuda: bool = None,
        shuffle: bool = None,
        timestamp_start: int = None,
        timestamp_end: int = None,
        data_length: int = None,
        git_commit: str = None,
        comment: str = None,
        is_timestamps: bool = False,
        is_validation: bool = False,
        no_comment: bool = None,
        **other,
    ):
        """Get data from the database based on the filters specified.

        The data will be retreived applying the filters using "AND". If "OR" is
        desired, the function must be called multiple times.

        Args:
            with_state_dict (bool, optional): Not used
            region (str, optional): Region of the relevant time series. This can be
                NO1, NO2, ..., EONon, EONoff, etc.
            master (bool, optional): This test data from a model that I developed for
                my master thesis.
            prosjektoppgave (bool, optional): This test data from a model that I
                developed for my fall project.
            lightgbm (bool, optional): This test data is from a lightgbm model
            hybrid (bool, optional): This test data is from a hybrid between the
                lightgbm model and the master model.
            batch_size (int, optional): Batch size used for training.
            dropout (float, optional): Dropout probability for the last two layers.
            epochs (int, optional): Number of epochs during training.
            lr (float, optional): Initial learning rate.
            normalize (bool, optional): Normalize the data. This is usually True for
                neural net, and false for random forest and equivalent ensamble
                learners.
            normalize_type (int, optional): Normalization type. See report for info.

                =====  ===============  ==========================================
                value  type             description
                =====  ===============  ==========================================
                1      Global mean/std  For each parameter, take the global mean
                                        and std and calculate(values−mean)/std.
                2      Local mean/std   For each grid location of each parameter,
                                        take the mean and stdof that location and
                                        calculate (values−mean)/std
                3      Global [0, 1]    For each parameter, map the range to the
                                        interval [0,1] by calculating
                                        (values−min)/(max−min)
                4      Local [0, 1]     For each grid location of each parameter,
                                        map the local range tothe interval [0,1]
                                        by calculating (values−min)/(max−min)
                =====  ===============  ==========================================

            optim (str, optional): Optimizer used during training.
            ordinal (bool, optional): Use ordinal classification as target.
            ordinal_resolution (int, optional): Number of ordinal classes that is
                used.
            ratio_transform (bool, optional): Transform ratio to a more even
                distribution before training. This seems to have a positive effect on
                the accuracy of the model.
            seed (int, optional): Random seed that is used.
            use_seed (bool, optional): Use a seed for random values.
            window_size (int, optional): Number of hours before (and after) the target
                hour that is included in the model. Usually set to 2 for the advanced
                model.
            harmonics (bool, optional): Harmonics are added to force seasonality.
            spp_resolution (int, optional): Resolution for the Spatial Pyramid Pooling
                layer. Default 5.
            lr_scheduler (bool, optional): Is a learningrate scheduler used during
                training.
            target (bool, optional): Flag to determine if the data stored is
                target values or estimated values from the model
            verbose (bool, optional): Training was done using the verbose flag. Not a
                very useful parameter but needed to be able to use argparse output
                directly with the function.
            cuda (bool, optional): Training was done using cuda. Not a very useful
                parameter but needed to be able to use argparse output directly with
                the function.
            shuffle (bool, optional): Training was done using dataset shuffle.
            timestamp_start (int, optional): First timestamp of the value in the
                series that is to be stored. The region+timestamp information is
                enough to identify the production value. The series is assumed to have
                an hourly resolution, so the end timestamp should be deterministic
                given the lenght of the dataset that is stored.
            timestamp_end (int, optional): Timestamp at the end of the dataset. This
                should be deterministic given the length of the dataset and hourly
                resolution.
            data_length (int, optional): Length of the dataset.
            git_commit (str, optional): git HEAD.
            comment (str, optional): Comment attatched to data
            is_timestamps (bool, optional): Timestamps flag. True/False
            is_validation (bool, optional): Validation flag. True/False
            no_comment (bool, optional): The data sjoud be fetch independent of comments.
            **other: Other parameters

        Return:
            numpy.ndarray: Data that fit the filter. The shape of the data will be
            (number of series, data points). If the lenght of the data series are
            different, the function will return error.

        """

        arguments = {
            "region": region,
            "master": master,
            "prosjektoppgave": prosjektoppgave,
            "lightgbm": lightgbm,
            "hybrid": hybrid,
            "batch_size": batch_size,
            "dropout": dropout,
            "epochs": epochs,
            "lr": lr,
            "normalize": normalize,
            "normalize_type": normalize_type,
            "optim": optim,
            "ordinal": ordinal,
            "ordinal_resolution": ordinal_resolution,
            "ratio_transform": ratio_transform,
            "seed": seed,
            "use_seed": use_seed,
            "window_size": window_size,
            "harmonics": harmonics,
            "spp_resolution": spp_resolution,
            "lr_scheduler": lr_scheduler,
            "target": target,
            "cuda": cuda,
            "shuffle": shuffle,
            "timestamp_start": timestamp_start,
            "timestamp_end": timestamp_end,
            "data_length": data_length,
            "git_commit": git_commit,
            "comment": comment,
            "is_timestamps": is_timestamps,
            "is_validation": is_validation,
        }

        tolerance = 1e-6
        # create sql. ignore values that are None
        # use the old database for {NS1, NS2, NS3, NS4} SE4, ENBW
        # sql = "SELECT data FROM results WHERE "

        sql = "SELECT data, created FROM results JOIN data ON results.data_id = data.data_id WHERE "
        for key, value in arguments.items():
            if value is None:
                continue
            if key in ["dropout", "lr"]:  # floats
                sql += f"abs({key}-%({key})s) <= {tolerance} AND "
            else:
                sql += f"{key} = %({key})s AND "
        if no_comment:
            sql += "comment IS NULL"
        else:
            sql = sql[:-5]
        sql += " ORDER BY created ASC"

        self.cursor.execute(sql, arguments)
        data = self.cursor.fetchall()
        data_unloaded = [pickle.loads(tmp[0]) for tmp in data]
        data = np.array(data_unloaded).squeeze()

        if with_state_dict == True:
            sql = "SELECT state_dict FROM results WHERE "
            for key, value in arguments.items():
                if value is None:
                    continue
                if key in ["dropout", "lr"]:  # floats
                    sql += f"abs({key}-%({key})s) <= {tolerance} AND "
                else:
                    sql += f"{key} = %({key})s AND "
            sql = sql[:-5]

            self.cursor.execute(sql, arguments)
            state_dict = self.cursor.fetchall()
            state_dict_unloaded = [pickle.loads(tmp[0]) for tmp in state_dict]
            return data, state_dict_unloaded

        return data


def main():
    t = TestResultsDatabase()
    t.close()


if __name__ == "__main__":
    main()
