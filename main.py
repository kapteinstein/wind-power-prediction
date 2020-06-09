import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

sns.set()

from dataset import Dataset, DatasetOrig, split, IgnoreEdgeSampler
from network import Net

from db import TestResultsDatabase

import argparse
import datetime as dt
import pickle
from pprint import pprint


def get_utc(timestamp):
    """Calculate utc datetime from unix timestamp

    Args:
        timestamp (int): timestamp

    Returns:
        datetime.datetime: Datetime object. UTC.
    """
    return dt.datetime.utcfromtimestamp(timestamp)


def calculate_maape(target, prediction):
    """ MAAPE - Mean arctangent absolute percentage error """
    AAPE = np.arctan2(np.abs(target - prediction), np.abs(target))
    return np.mean(AAPE)


def calculate_mape(target, prediction):
    """ mape - error metric of predictions. most interesting for the production target series."""
    return np.mean(np.abs(prediction - target)) / np.mean(target)


def load_time_series():
    """ time series from gabri. seasonality"""
    series_df = pd.read_csv("harmonics/seasonal_harmonics.csv", header=None)
    return series_df


class Model(object):
    def __init__(self, args, db=None):
        self.args = args
        self.db = db
        self.dataset = DatasetOrig(args.region)
        self.harmonics = self.dataset.fetch_harmonics(self.load_harmonics())

        self.indecies = split(self.dataset.idx, 0.7, 0.85)
        self.idx_train, self.idx_validate, self.idx_test = self.indecies

        self.train_set = self.dataset.get_sample(self.idx_train, self.args)
        self.validate_set = self.dataset.get_sample(self.idx_validate, self.args)
        self.test_set = self.dataset.get_sample(self.idx_test, self.args)

        (
            self.x_train,
            self.y_train,
            self.timestamp_train,
            self.harmonics_train,
        ) = self.train_set
        (
            self.x_val,
            self.y_val,
            self.timestamp_val,
            self.harmonics_val,
        ) = self.validate_set
        (
            self.x_test,
            self.y_test,
            self.timestamp_test,
            self.harmonics_test,
        ) = self.test_set

        self.frame_dim = (self.x_train.size(2), self.x_train.size(3))
        self.mask = self.dataset.get_mask()

        # calculate bins for ordinal - alternative
        # this is based on the training set alone. not validation or test
        # I am not sure if validation set can be included here.
        # I can test both and see if it makes any difference
        # self.bins = None
        # if args.ordinal:
        #    self.nbins = args.ordinal_resolution + 1
        #    self.bins = np.quantile(self.y_train, np.linspace(0, 1, self.nbins))
        #    #self.bins = np.quantile(self.dataset.ratio, np.linspace(0, 1, self.nbins))

        # plt.hist(self.dataset.ratio)
        # plt.show()
        # plt.hist(self.dataset.ratio, self.bins)
        # plt.show()
        # exit()

        self.dataset_train = Dataset(
            self.x_train, self.y_train, self.harmonics_train, self.args
        )
        self.dataset_validate = Dataset(
            self.x_val, self.y_val, self.harmonics_val, self.args
        )
        self.dataset_test = Dataset(
            self.x_test, self.y_test, self.harmonics_test, self.args
        )

    def show_args_info(self, args=None):
        """Show summary of arguments. Note that all is not shown.

        Args:
            args (None, optional): Alternative args from argparse.
        """
        args = self.args if args is None else args
        print("------------- INFO- ------------")
        print(f"    dataset/region: {args.region}")
        print(f"         normalize: {args.normalize}")
        if args.normalize == True:
            print(f"normalization type: {args.normalize_type}")
            print(f"   transform ratio: {args.ratio_transform}")
        print(f"       window size: {args.window_size}")
        print(f"   shuffle dataset: {args.shuffle}")
        print()
        print(f"  number of epochs: {args.epochs}")
        print(f"    minibatch size: {args.batch_size}")
        print(f"         optimizer: {args.optim}")
        print(f"     learning rate: {args.lr if args.lr else '(default)'}")
        print(f"           dropout: {args.dropout}")
        print(f"ordinal regression: {args.ordinal}")
        if args.ordinal:
            print(f"ordinal resolution: {args.ordinal_resolution}")
        print()
        print(f"     verbose level: {args.verbose}")
        print(f"      cuda is used: {args.cuda}")
        print(f" multi gpu is used: {args.multi_gpu}")
        print(f"   storage enabled: {args.store_results}")
        print(f"   mask after conv: {args.use_mask} (experimental)")

    def show_dataset_info(self):
        """Show summary of the data that will be used for training, testing, and validation.
        """
        print("\n------------ DATASET -----------")
        print(f"invalid data: {len(self.dataset.nans)} (removed)")
        print()
        print("---------  training  -----------", end="\t")
        print("--------- validation -----------", end="\t")
        print("---------  testing  ------------")

        print(f"number of samples: {len(self.dataset_train)}", end="\t\t")
        print(f"number of samples: {len(self.dataset_validate)}", end="\t\t\t")
        print(f"number of samples: {len(self.dataset_test)}")

        print(f"timestamp start: {self.timestamp_train[0]}", end="\t\t")
        print(f"timestamp start: {self.timestamp_val[0]}", end="\t\t")
        print(f"timestamp start: {self.timestamp_test[0]}")

        print(f"train start: {get_utc(self.timestamp_train[0])}", end="\t")
        print(f"valid start: {get_utc(self.timestamp_val[0])}", end="\t")
        print(f"test start : {get_utc(self.timestamp_test[0])}")

        print(f"train end  : {get_utc(self.timestamp_train[-1])}", end="\t")
        print(f"valid end  : {get_utc(self.timestamp_val[-1])}", end="\t")
        print(f"test end   : {get_utc(self.timestamp_test[-1])}")

        print("--------------------------------", end="\t")
        print("--------------------------------", end="\t")
        print("--------------------------------")
        print()

    def init_ann(self, args=None):
        """Initialize ANN

        Args:
            args (None, optional): Alternative args from argparse.
        """
        args = self.args if args is None else args
        self.model = Net(args, self.frame_dim, self.mask)
        self.device = torch.device("cpu")
        if args.cuda:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            if torch.cuda.device_count() > 1 and args.multi_gpu:
                self.device = torch.device("cuda:0")
                if args.verbose >= 1:
                    print("Using", torch.cuda.device_count(), "GPUs")
                self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

        if args.ordinal:
            # Binary Cross Entropy With Logits Loss function can also be used.
            # From docs:
            # BCEWithLogitsLoss combines sigmoid and BCELoss and is considered more
            # numerical stable than a sigmoid layer followed by BCELoss.
            self.loss_fn = nn.BCEWithLogitsLoss()

            # the order classification paper also mentioned this loss function.
            # the parameters reduction=None, mean and sum can be used
            # however, the paper did not find it better than simple MSE.
            # loss_fn = nn.BCELoss()
        else:
            self.loss_fn = nn.MSELoss(reduction="mean")

        if args.optim == "SGD":
            lr = args.lr if args.lr else 0.01
            self.args.lr = lr
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5
            )
        elif args.optim == "ADAM":
            lr = args.lr if args.lr else 0.0001
            self.args.lr = lr
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=lr, weight_decay=1e-5
            )
        if args.verbose >= 2:
            print("INFO: model initialized")

    def init_data_loader(self, args=None):
        """Initialize dataloader

        Args:
            args (None, optional): Alternative args from argparse.
        """

        args = self.args if args is None else args
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, verbose=True)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

        self.data_train_sampler = IgnoreEdgeSampler(
            self.dataset_train, shuffle=args.shuffle, window_size=args.window_size
        )
        self.data_validate_sampler = IgnoreEdgeSampler(
            self.dataset_validate, shuffle=False, window_size=args.window_size
        )
        self.data_test_sampler = IgnoreEdgeSampler(
            self.dataset_test, shuffle=False, window_size=args.window_size
        )

        self.data_train_loader = torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=args.batch_size,
            sampler=self.data_train_sampler,
        )
        self.data_validate_loader = torch.utils.data.DataLoader(
            self.dataset_validate,
            batch_size=args.batch_size,
            sampler=self.data_validate_sampler,
        )
        self.data_test_loader = torch.utils.data.DataLoader(
            self.dataset_test,
            batch_size=args.batch_size,
            sampler=self.data_test_sampler,
        )

        if args.verbose >= 2:
            print("INFO: dataloader initialized")
            print("INFO: dataset test length:", len(self.dataset_test))
            print("INFO: test sampler length:", len(self.data_test_sampler))

    def run_train_epoch(self):
        """Run one complete training epoch

        This function goes through the training dataset once and complete forward and
        backwards pass for one complete epoch.

        Returns:
            float: total loss from epoch given the defined loss function for the model.
        """
        epoch_loss = 0
        for x, y, t in self.data_train_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            t = t.to(self.device)

            self.optimizer.zero_grad()
            out, second_to_last_layer_logits = self.model(x, t)

            loss = self.loss_fn(out, y)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss

    def run_validate_epoch(self):
        """Run one complete validation epoch

        This function goes through the validation dataset once and complete a forward
        pass for one complete epoch. The model does not update the gradients.

        Returns:
            float: total loss from epoch given the defined loss function for the model.
        """
        val_loss = 0
        with torch.no_grad():
            for x, y, t in self.data_validate_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                t = t.to(self.device)
                out, second_to_last_layer_logits = self.model(x, t)
                loss = self.loss_fn(out, y)
                val_loss += loss.item()
        return val_loss

    def train(self, args=None):
        """Train the ANN over all epochs

        Args:
            args (None, optional): Alternative args from argparse

        Returns:
            tuple: two numpy.ndarray containing train loss and validation loss history
        """
        args = self.args if args is None else args
        max_epoch = args.epochs

        train_loss = np.zeros(max_epoch)
        validation_loss = np.zeros(max_epoch)

        self.model.train()
        for i in range(max_epoch):
            print("epoch: {}/{}".format(i + 1, max_epoch), end="\r")
            t_loss = self.run_train_epoch()
            v_loss = self.run_validate_epoch()
            # scheduler.step(val_loss)

            train_loss[i] = t_loss
            validation_loss[i] = v_loss

            if args.verbose >= 1:
                print(f"epoch {i + 1}: - train: {t_loss}, validate: {v_loss}")

        if args.verbose >= 2:
            print("INFO: training complete")
        return train_loss, validation_loss

    def final_validation(self, args=None):
        """Final validation of validation set. Store results.

        Args:
            args (None, optional): Alternative args from argparse.
        """
        args = self.args if args is None else args
        predictions = np.zeros(len(self.data_validate_sampler))

        self.model.eval()
        index = 0
        x_range = np.arange(args.ordinal_resolution)
        with torch.no_grad():
            for x, y, t in self.data_validate_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                t = t.to(self.device)

                out, second_to_last_layer_logits = self.model(x, t)

                out = out.cpu().numpy()
                if args.ordinal:
                    for y_out in out:
                        predictions[index] = (
                            np.argmax(y_out < 0) / args.ordinal_resolution
                        )
                        index += 1
                else:
                    predictions[index : index + y.size(0)] = out.flatten()
                    index += y.size(0)

        actual_prod = self.dataset.production[self.idx_validate]
        capacity = self.dataset.capacity[self.idx_validate]

        # remove endpoints of target vectors if window size is > 0
        if args.window_size != 0:
            actual_prod = actual_prod[args.window_size : -args.window_size]
            capacity = capacity[args.window_size : -args.window_size]

        # transform back to normal values
        if args.ratio_transform:
            predictions = 1 - np.sqrt(1 - predictions)

        estimated = np.multiply(predictions, capacity)
        AAPE = np.arctan2(np.abs(actual_prod - estimated), np.abs(actual_prod))
        MAAPE = np.mean(AAPE)
        print("validation MAAPE: ", MAAPE)

        MAPE = np.mean(np.abs(estimated - actual_prod)) / np.mean(actual_prod)
        print("validation MAPE.: ", MAPE)

        if args.store_results:
            status = self.store_results(
                estimated,
                target=False,
                is_validation=True,
                state_dict=self.model.state_dict(),
            )
            if args.verbose >= 2 and status is True:
                print("INFO: storage success")
            status = self.store_results(
                actual_prod,
                target=True,
                is_validation=True,
                state_dict=self.model.state_dict(),
            )
            if args.window_size != 0:
                data = self.timestamp_test[args.window_size : -args.window_size]
            else:
                data = self.timestamp_test
            status = self.store_results(
                data,
                is_timestamps=True,
                is_validation=True,
                state_dict=self.model.state_dict(),
            )

    def test(self, args=None):
        """Test the model using the test set. Store results.

        Args:
            args (None, optional): Alternative args from argparse.
        """
        args = self.args if args is None else args
        predictions = np.zeros(len(self.data_test_sampler))

        self.model.eval()
        index = 0
        x_range = np.arange(args.ordinal_resolution)
        with torch.no_grad():
            for x, y, t in self.data_test_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                t = t.to(self.device)

                out, second_to_last_layer_logits = self.model(x, t)

                out = out.cpu().numpy()
                if args.ordinal:
                    for y_out in out:
                        predictions[index] = (
                            np.argmax(y_out < 0) / args.ordinal_resolution
                        )
                        index += 1
                else:
                    predictions[index : index + y.size(0)] = out.flatten()
                    index += y.size(0)

        actual_prod = self.dataset.production[self.idx_test]
        capacity = self.dataset.capacity[self.idx_test]

        # remove endpoints of target vectors if window size is > 0
        if args.window_size != 0:
            actual_prod = actual_prod[args.window_size : -args.window_size]
            capacity = capacity[args.window_size : -args.window_size]

        # transform back to normal values
        if args.ratio_transform:
            predictions = 1 - np.sqrt(1 - predictions)
            # for i in range(len(predictions)):
            #    index = int(predictions[i] * args.ordinal_resolution)
            #    reverse = (self.bins[index] + self.bins[index + 1]) / 2
            #    predictions[i] = reverse
        #    predictions = 1 - np.sqrt(1 - predictions)

        estimated = np.multiply(predictions, capacity)
        AAPE = np.arctan2(np.abs(actual_prod - estimated), np.abs(actual_prod))
        MAAPE = np.mean(AAPE)
        print("test MAAPE: ", MAAPE)

        MAPE = np.mean(np.abs(estimated - actual_prod)) / np.mean(actual_prod)
        print("test MAPE : ", MAPE)

        if args.store_results:
            status = self.store_results(
                estimated, target=False, state_dict=self.model.state_dict()
            )
            if args.verbose >= 2 and status is True:
                print("INFO: storage success")
            status = self.store_results(
                actual_prod, target=True, state_dict=self.model.state_dict()
            )
            if args.window_size != 0:
                data = self.timestamp_test[args.window_size : -args.window_size]
            else:
                data = self.timestamp_test
            status = self.store_results(
                data, is_timestamps=True, state_dict=self.model.state_dict()
            )

    def store_results(
        self,
        data,
        target=None,
        db=None,
        is_validation=False,
        is_timestamps=False,
        is_lightgbm=False,
        is_hybrid=False,
        state_dict=None,
    ):
        """Store results in database

        Args:
            data (numpy.ndarray): Data to be stored
            target (bool, optional): Data is target values
            db (None, optional): Database connection
            is_validation (bool, optional): Is validation
            is_timestamps (bool, optional): Is timestamps
            is_lightgbm (bool, optional): Is lightgbm model
            is_hybrid (bool, optional): Is hybrid model
            state_dict (None, optional): Not in use.

        Returns:
            TYPE: Description
        """
        db = self.db if db is None else db
        if db is None:
            if self.args.verbose >= 1:
                print("WARNING: database not present. Results will not be stored")
            return False
        db.connect()
        status = db.store(
            data,
            state_dict=state_dict,
            master=True,
            target=target,
            timestamp_start=get_utc(self.timestamp_test[0]),
            is_timestamps=is_timestamps,
            is_validation=is_validation,
            is_lightgbm=is_lightgbm,
            is_hybrid=is_hybrid,
            **vars(self.args),
        )
        db.close()

        return status

    def load_harmonics(self):
        """Load Seasonal Harmonics

        Returns:
            DataFrame: Seasonal Harmonics for all timestamps
        """
        harmonics = load_time_series()
        harmonics.columns = [
            "timestamp",
            "s1",
            "s2",
            "s3",
            "s4",
            "s5",
            "s6",
            "s7",
            "s8",
            "s9",
            "s10",
            "s11",
            "s12",
            "s13",
            "s14",
            "s15",
            "s16",
            "s17",
            "s18",
            "s19",
            "s20",
            "s21",
            "s22",
            "s23",
            "s24",
            "s25",
        ]

        harmonics["timestamp"] = pd.to_datetime(harmonics["timestamp"])
        return harmonics

    def lgbm(self):
        """Run Hybrid and LightGBM model after CNN training.
        """
        samples = self.dataset.get_sample(self.dataset.idx, self.args)
        weather, target, timestamp, harmonics = samples
        capacity = self.dataset.capacity[self.dataset.idx]
        production = self.dataset.production[self.dataset.idx]
        # print("x:", weather.size())
        # print("y:", target.size())
        # print("t:", harmonics.shape)

        dataset = Dataset(weather, target, harmonics, self.args)
        sampler = IgnoreEdgeSampler(
            dataset, shuffle=False, window_size=args.window_size
        )

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=1000, sampler=sampler,
        )

        logits = np.zeros(shape=(len(sampler), 50))
        timestamps = np.zeros(shape=(len(sampler), 1))
        outs = np.zeros(shape=(len(sampler), 100))

        if args.hybrid and args.use_ann:
            self.model.eval()
            with torch.no_grad():
                for i, (x, y, t) in enumerate(data_loader):
                    i = i * 1000
                    x = x.to(self.device)
                    y = y.to(self.device)
                    t = t.to(self.device)
                    out, second_to_last_layer_logits = self.model(x, t)
                    logits[i : i + 1000, :] = second_to_last_layer_logits.cpu().numpy()
                    # timestamps[i:i+1000, 0] = t.cpu().numpy()
                    outs[i : i + 1000, :] = out.cpu().numpy()

        # pprint(timestamps[0])
        # print(timestamps[-1])
        # print(np.max(np.diff(timestamps)))
        # print(np.min(np.diff(timestamps)))

        # print(np.max(timestamp[2:-2]-timestamps))  # verify equality

        harmonics = self.load_harmonics()

        # get original data from dataset
        wind = self.dataset.wind_speed[self.dataset.idx].reshape(
            len(self.dataset.idx), -1
        )
        temperature = self.dataset.temperature[self.dataset.idx].reshape(
            len(self.dataset.idx), -1
        )
        pressure = self.dataset.pressure[self.dataset.idx].reshape(
            len(self.dataset.idx), -1
        )
        # angle = self.dataset.wind_angle[self.dataset.idx].reshape(len(self.dataset.idx), -1)

        wind_prev = np.roll(wind, 1, axis=0)
        wind_prev2 = np.roll(wind, 2, axis=0)
        wind_prev3 = np.roll(wind, -1, axis=0)
        wind_prev4 = np.roll(wind, -2, axis=0)

        # trim of edges if window size is > 0
        if args.window_size != 0:
            wind = wind[args.window_size : len(wind) - args.window_size]
            # angle = angle[args.window_size:len(angle)-args.window_size]
            wind_prev = wind_prev[args.window_size : len(wind_prev) - args.window_size]
            wind_prev2 = wind_prev2[
                args.window_size : len(wind_prev2) - args.window_size
            ]
            wind_prev3 = wind_prev3[
                args.window_size : len(wind_prev3) - args.window_size
            ]
            wind_prev4 = wind_prev4[
                args.window_size : len(wind_prev4) - args.window_size
            ]
            temperature = temperature[
                args.window_size : len(temperature) - args.window_size
            ]
            pressure = pressure[args.window_size : len(pressure) - args.window_size]

            target = target[
                args.window_size : len(target) - args.window_size
            ]  # ratio. bad naming
            capacity = capacity[args.window_size : len(capacity) - args.window_size]
            production = production[
                args.window_size : len(production) - args.window_size
            ]

            timestamp = timestamp[args.window_size : len(timestamp) - args.window_size]

        timestamps[:, 0] = timestamp

        if args.verbose >= 3:
            print("[Debug] wind.shape.......:", wind.shape)
            print("[Debug] temperature.shape:", temperature.shape)
            print("[Debug] pressure.shape...:", pressure.shape)
            print("[Debug] target.shape.....:", target.shape)

        if args.hybrid and args.use_harmonics and args.use_ann:
            data = np.concatenate(
                (
                    timestamps[:, :],
                    wind[:, ::3],
                    wind_prev[:, ::3],
                    temperature[:, ::3],
                    pressure[:, ::3],
                    logits[:, :],
                ),
                axis=1,
            )
        elif args.hybrid and args.use_ann:
            data = np.concatenate(
                (
                    wind[:, ::3],
                    wind_prev[:, ::3],
                    temperature[:, ::3],
                    pressure[:, ::3],
                    logits[:, :],
                ),
                axis=1,
            )
        elif args.use_harmonics:
            data = np.concatenate(
                (
                    timestamps[:, :],
                    wind[:, ::3],
                    wind_prev[:, ::3],
                    temperature[:, ::3],
                    pressure[:, ::3],
                ),
                axis=1,
            )
        else:
            data = np.concatenate(
                (
                    wind[:, ::3],
                    wind_prev[:, ::3],
                    temperature[:, ::3],
                    pressure[:, ::3],
                ),
                axis=1,
            )
            if args.hybrid == True and args.use_ann == False:
                print(
                    "[Warning] hybrid selected, but ANN is disabled. Fallback to no ANN"
                )

        train_df = pd.DataFrame(data)

        target_df = pd.DataFrame(
            np.concatenate((timestamps[:, :], target.reshape(len(target), 1)), axis=1)
        )

        train_df = train_df.rename(columns={0: "timestamp"})
        target_df = target_df.rename(columns={0: "timestamp", 1: "target"})
        train_df["timestamp"] = pd.to_datetime(train_df["timestamp"], unit="s")
        target_df["timestamp"] = pd.to_datetime(target_df["timestamp"], unit="s")
        if args.use_harmonics:
            data_df = pd.merge(train_df, harmonics, how="left", on=["timestamp"])
        else:
            data_df = train_df

        # verify equality. this shoud be zero
        # print(np.max(np.abs(train_df['timestamp']-target_df['timestamp'])))

        del data_df["timestamp"]
        del target_df["timestamp"]

        idx_train, idx_test = np.split(
            np.arange(len(timestamps), dtype=int), [int(len(timestamps) * 0.7)]
        )

        if args.rf:
            if args.verbose >= 1:
                print("[Info] random forest is used")
            model = random_forest(idx_train, idx_test, data_df, target_df)
        elif args.knn:
            if args.verbose >= 1:
                print("[Info] KNN is used")
            model = knn_regressor(idx_train, idx_test, data_df, target_df)
        elif args.adaboost:
            if args.verbose >= 1:
                print("[Info] AdaBoost is used")
            model = AdaBoost_regressor(idx_train, idx_test, data_df, target_df)
        elif args.gradientboosting:
            if args.verbose >= 1:
                print("[Info] GradientBoosting is used")
            model = GradientBoosting_regressor(idx_train, idx_test, data_df, target_df)
        else:
            if args.verbose >= 1:
                print("[Info] LightGBM is used")
            model = lgbm(idx_train, idx_test, data_df, target_df)
        lgbm_params = {
            "boosting": "gbdt",
            # "max_bin": 63,
            "objective": "regression",
            # "metric": {"l2", "l1"},
            "metric": {"mape"},
            # "num_leaves": 31,
            # "learning_rate": 0.05,
            # "feature_fraction": 0.9,
            # "bagging_fraction": 0.8,
            # "bagging_freq": 5,
            "verbose": args.verbose if args.verbose > 0 else 0,
        }
        rf_params = {
            "max_depth": 10,
            "random_state": 0,
            # "n_estimators": 10,
            "n_jobs": -1,
            "verbose": args.verbose if args.verbose > 0 else 0,
        }
        knn_params = {"n_neighbors": 10, "weights": "distance", "n_jobs": -1}

        adaboost_params = {}
        gradientboosting_params = {}

        if args.rf:
            model.train_regressor(rf_params)
        elif args.knn:
            model.train_regressor(knn_params)
        elif args.adaboost:
            model.train_regressor(adaboost_params)
        elif args.gradientboosting:
            model.train_regressor(gradientboosting_params)
        else:
            model.train_regressor(lgbm_params)
        predictions = model.get_predictions()
        # transform back to normal values
        if args.ratio_transform:
            predictions[predictions < 0] = 0
            predictions[predictions > 1] = 1
            predictions = 1 - np.sqrt(1 - predictions)

        estimated = np.multiply(predictions, capacity[idx_test])

        if args.store_results:
            valid_timestamp, test_timestamp = np.array_split(timestamp[idx_test], 2)
            if args.verbose >= 2:
                print(
                    "[Debug] valid_timestamps: {} to {}".format(
                        get_utc(valid_timestamp[0]), get_utc(valid_timestamp[-1])
                    )
                )
                print(
                    "[Devug] test_timestamps: {} to {}".format(
                        get_utc(test_timestamp[0]), get_utc(test_timestamp[-1])
                    )
                )
            valid_estimated, test_estimated = np.array_split(estimated, 2)
            valid_production, test_production = np.array_split(production[idx_test], 2)
            if args.hybrid:
                status_1v = self.store_results(
                    valid_estimated,
                    is_validation=True,
                    target=False,
                    is_lightgbm=True,
                    is_hybrid=True,
                )
                status_2v = self.store_results(
                    valid_production,
                    is_validation=True,
                    target=True,
                    is_lightgbm=True,
                    is_hybrid=True,
                )
                status_3v = self.store_results(
                    valid_timestamp,
                    is_validation=True,
                    is_timestamps=True,
                    is_lightgbm=True,
                    is_hybrid=True,
                )

                status_1t = self.store_results(
                    test_estimated, target=False, is_lightgbm=True, is_hybrid=True
                )
                status_2t = self.store_results(
                    test_production, target=True, is_lightgbm=True, is_hybrid=True
                )
                status_3t = self.store_results(
                    test_timestamp, is_timestamps=True, is_lightgbm=True, is_hybrid=True
                )
            else:
                status_1v = self.store_results(
                    valid_estimated, is_validation=True, target=False, is_lightgbm=True,
                )
                status_2v = self.store_results(
                    valid_production, is_validation=True, target=True, is_lightgbm=True
                )
                status_3v = self.store_results(
                    valid_timestamp,
                    is_validation=True,
                    is_timestamps=True,
                    is_lightgbm=True,
                )

                status_1t = self.store_results(
                    test_estimated, target=False, is_lightgbm=True,
                )
                status_2t = self.store_results(
                    test_production, target=True, is_lightgbm=True
                )
                status_3t = self.store_results(
                    test_timestamp, is_timestamps=True, is_lightgbm=True
                )
            if args.verbose >= 2:
                print("[Info] save valid_estimated.:", status_1v)
                print("[Info] save valid_targets...:", status_2v)
                print("[Info] save valid_timestamps:", status_3v)

                print("[Info] save test_estimated.:", status_1t)
                print("[Info] save test_targets...:", status_2t)
                print("[Info] save test_timestamps:", status_3t)

        maape = calculate_maape(production[idx_test], estimated)
        mape = calculate_mape(production[idx_test], estimated)
        print("hybrid maape:", maape)
        print("hybrid mape.: ", mape)


class random_forest(object):
    """ Random Forest regressor """

    def __init__(self, idx_train, idx_test, data_df, target_df):
        self.idx_train = idx_train
        self.idx_test = idx_test
        self.data = data_df
        self.target = target_df

        self.train_data = self.data.iloc[self.idx_train, :]
        self.train_target = self.target.iloc[self.idx_train, :]
        self.test_data = self.data.iloc[self.idx_test, :]
        self.test_target = self.target.iloc[self.idx_test, :]

    def train_regressor(self, params):
        self.model = RandomForestRegressor(**params)
        self.model.fit(self.train_data, self.train_target.values.ravel())

    def get_predictions(self):
        predicted_ratio = self.model.predict(self.test_data)
        return predicted_ratio


class GradientBoosting_regressor(object):
    """ Random Forest regressor """

    def __init__(self, idx_train, idx_test, data_df, target_df):
        self.idx_train = idx_train
        self.idx_test = idx_test
        self.data = data_df
        self.target = target_df

        self.train_data = self.data.iloc[self.idx_train, :]
        self.train_target = self.target.iloc[self.idx_train, :]
        self.test_data = self.data.iloc[self.idx_test, :]
        self.test_target = self.target.iloc[self.idx_test, :]

    def train_regressor(self, params):
        self.model = GradientBoostingRegressor(**params)
        self.model.fit(self.train_data, self.train_target.values.ravel())

    def get_predictions(self):
        predicted_ratio = self.model.predict(self.test_data)
        return predicted_ratio


class AdaBoost_regressor(object):
    """ AdaBoost regressor """

    def __init__(self, idx_train, idx_test, data_df, target_df):
        self.idx_train = idx_train
        self.idx_test = idx_test
        self.data = data_df
        self.target = target_df

        self.train_data = self.data.iloc[self.idx_train, :]
        self.train_target = self.target.iloc[self.idx_train, :]
        self.test_data = self.data.iloc[self.idx_test, :]
        self.test_target = self.target.iloc[self.idx_test, :]

    def train_regressor(self, params):
        self.model = AdaBoostRegressor(**params)
        self.model.fit(self.train_data, self.train_target.values.ravel())

    def get_predictions(self):
        predicted_ratio = self.model.predict(self.test_data)
        return predicted_ratio


class knn_regressor(object):
    """ KNN regressor """

    def __init__(self, idx_train, idx_test, data_df, target_df):
        self.idx_train = idx_train
        self.idx_test = idx_test
        self.data = data_df
        self.target = target_df

        self.train_data = self.data.iloc[self.idx_train, :]
        self.train_target = self.target.iloc[self.idx_train, :]
        self.test_data = self.data.iloc[self.idx_test, :]
        self.test_target = self.target.iloc[self.idx_test, :]

    def train_regressor(self, params):
        self.model = KNeighborsRegressor(**params)
        self.model.fit(self.train_data, self.train_target.values.ravel())

    def get_predictions(self):
        predicted_ratio = self.model.predict(self.test_data)
        return predicted_ratio


class lgbm(object):
    """ LightGBM regressor """

    def __init__(self, idx_train, idx_test, data_df, target_df):
        self.idx_train = idx_train
        self.idx_test = idx_test
        self.data = data_df
        self.target = target_df

        self.train_data = self.data.iloc[self.idx_train, :]
        self.train_target = self.target.iloc[self.idx_train, :]
        self.test_data = self.data.iloc[self.idx_test, :]
        self.test_target = self.target.iloc[self.idx_test, :]

    def train_regressor(self, params):
        # kf = KFold(n_splits=3, shuffle=True)
        # self.train, self.ratio_train = shuffle(self.train, self.ratio_train)
        idx = np.arange(len(self.train_data))
        # np.random.shuffle(idx)
        self.model = []
        train_index, test_index = np.split(idx, [int(0.85 * len(idx))])
        # data_df = pd.merge(self.train_data, self.train_target, right_index=True, left_index=True)

        # for i in range(3):
        # np.random.shuffle(idx)
        # data_df = data_df.sample(frac=1).reset_index(drop=True)

        # self.train_data = self.train_data.sample(frac=1).reset_index(drop=True)
        # self.train_target = self.train_target.sample(frac=1).reset_index(drop=True)

        # for train_index, test_index in kf.split(idx):

        lgb_train_all = lgb.Dataset(self.train_data, self.train_target)

        lgb_train = lgb.Dataset(
            self.train_data.iloc[train_index, :], self.train_target.iloc[train_index, :]
        )
        lgb_eval = lgb.Dataset(
            self.train_data.iloc[test_index, :], self.train_target.iloc[test_index, :]
        )

        gbm = lgb.train(
            params,
            # lgb_train,
            lgb_train_all,
            num_boost_round=100,
            # valid_sets=lgb_eval,
            # early_stopping_rounds=20,
            verbose_eval=False,
        )

        self.model.append(gbm)

    def get_predictions(self):
        X = np.zeros((len(self.model), len(self.test_data)))
        for i in range(len(self.model)):
            predicted_ratio = self.model[i].predict(self.test_data)
            X[i] = predicted_ratio
            # X[i] = np.multiply(predicted_ratio, self.test_capacity)
            # print(predicted_ratio)
            # print(self.test_target)
        predicted = np.mean(X, axis=0)
        # print(predicted_production)
        return predicted


def main(args):
    """Train and test the model. See `python main.py --help` for help.

    Args:
        args (TYPE): Alternative args from argparse.
    """
    db = None
    if args.store_results:
        db = TestResultsDatabase(verbose=args.verbose)
    model = Model(args, db)

    model.show_args_info()
    model.show_dataset_info()
    model.init_data_loader()
    for i in range(args.runs):
        if args.use_ann:
            model.init_ann()
            model.train()
            model.final_validation()
            model.test()

        if (
            args.hybrid
            or args.lgbm
            or args.rf
            or args.knn
            or args.adaboost
            or args.gradientboosting
        ):
            model.lgbm()

    if args.store_results:
        db.close()


def get_parser():
    """Generate argument parser

    Returns:
        argparse: argparse parser
    """
    parser = argparse.ArgumentParser(
        description="This is the main thing. Wind power prediction."
    )
    model_group = parser.add_argument_group(title="model")
    data_group = parser.add_argument_group(title="dataset")
    flag_group = parser.add_argument_group(title="flags")

    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="increase output verbosity"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        metavar="N",
        help="""Number of repeated training/testing iterations to do.
                This is usefull when the model accuracy is evaluated as the
                only part changins is the initial weights.""",
    )
    parser.add_argument(
        "--comment", type=str, help="Comment appended to result on storage"
    )

    model_group.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="batch size (default: 32)",
    )
    model_group.add_argument(
        "--epochs", type=int, default=50, help="upper epoch limit (default: 50)"
    )

    model_group.add_argument(
        "--lr",
        type=float,
        help="learning rate (default: 0.01 for SDG, 0.0001 for ADAM)",
    )
    model_group.add_argument(
        "--optim",
        type=str,
        choices=["SGD", "ADAM"],
        default="SGD",
        help="optimizer to use (default: SGD)",
    )
    model_group.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        metavar="N",
        help="""probability of dropout in the last fully connected
                layers. (default: 0.2)""",
    )
    model_group.add_argument(
        "--seed", type=int, default=0, help="random seed (default: 0)"
    )
    model_group.add_argument(
        "--ordinal-resolution",
        type=int,
        default=100,
        metavar="N",
        help="""number of ordinal classification classes.
                The classes are equally distributed. (default: 100)""",
    )

    model_group.add_argument(
        "--hybrid", action="store_true", help="""enable hybrid model"""
    )

    model_group.add_argument(
        "--lgbm",
        action="store_true",
        help="""enable the LightGBM model. Not needed if --hybrid is used.""",
    )

    model_group.add_argument(
        "--adaboost", action="store_true", help="""enable adaboost"""
    )
    model_group.add_argument(
        "--gradientboosting", action="store_true", help="""enable gradientboosting."""
    )
    model_group.add_argument(
        "--rf",
        action="store_true",
        help="""enable the random forest model. Mutual exclusive to --lgbm.""",
    )

    model_group.add_argument(
        "--knn",
        action="store_true",
        help="""enable the knn model. Mutual exclusive to --lgbm.""",
    )

    data_group.add_argument(
        "--region", type=str, default="DK1on", help="region to use (default: DK1on)"
    )
    data_group.add_argument(
        "--normalize-type",
        type=int,
        choices=[1, 2, 3, 4],
        default=1,
        metavar="N",
        help="""normalization type. see report for details. ignored if
                --no-normalize is used. (default: 1)""",
    )
    data_group.add_argument(
        "--window-size",
        type=int,
        default=2,
        metavar="N",
        help="""number of hours before and after the actual timestamp to include
                in estimation. (default: 2)""",
    )
    data_group.add_argument(
        "--use-harmonics",
        action="store_true",
        help="""use harmonics to force seasonality""",
    )

    flag_group.add_argument(
        "--use-mask",
        action="store_true",
        help="""Apply mask after the convolutional layers in the network. The network
                is same-padded. (experimental)""",
    )
    flag_group.add_argument(
        "--multi-gpu",
        action="store_true",
        help="""use multiple GPUs if available.
                The data will be split evenly across the GPUs.""",
    )
    flag_group.add_argument(
        "--no-cuda", action="store_false", dest="cuda", help="disable CUDA"
    )
    flag_group.add_argument(
        "--no-shuffle",
        action="store_false",
        dest="shuffle",
        help="""disable shuffeling of dataset. Remember that the shuffle
                is deterministic given the seed.""",
    )
    flag_group.add_argument(
        "--no-transform",
        action="store_false",
        dest="ratio_transform",
        help="disable ratio transformation. see report for details.",
    )
    flag_group.add_argument(
        "--no-normalize",
        action="store_false",
        dest="normalize",
        help="disable normalization",
    )
    flag_group.add_argument(
        "--no-seed",
        action="store_false",
        dest="use_seed",
        help="disable seed and use a random seed instead.",
    )
    flag_group.add_argument(
        "--no-ordinal",
        action="store_false",
        dest="ordinal",
        help="disable ordinal regression and use norman one-node regression as target.",
    )
    flag_group.add_argument(
        "--no-storage",
        action="store_false",
        dest="store_results",
        help="disable storage of model performance.",
    )
    flag_group.add_argument(
        "--no-ann", action="store_false", dest="use_ann", help="disable neural network"
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    if args.use_seed:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    main(args)
