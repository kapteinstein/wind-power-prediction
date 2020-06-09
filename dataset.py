import h5py as h5
import numpy as np
import torch
import torch.utils.data
import datetime
import pandas as pd


class DatasetOrig(object):
    def __init__(self, region, folder="dataset", index=None):
        """Wrapper around the raw values of the dataset

            This dataset is connected to a region.

            Attributes:
                capacity (numpy.ndarray): Capacity of the region in MWh.
                dates (numpy.ndarray): List of dates in the dataset.
                idx (numpy.ndarray): List of indices in the dataset.
                mask (numpy.ndarray): Mask of the region if the region is not a rectangle.
                    see ``dataconvert.convert()``. :meth: `dataconvert.convert`.
                nans (numpy.ndarray): List of indicies that have invalid data. This data is
                    removed.
                norm_data (numpy.ndarray): Data used for normalization of the dataset.
                pressure (numpy.ndarray): Atmospheric pressure in hPa in the region.
                production (numpy.ndarray): The production value each hour in MWh
                ratio (numpy.ndarray): Target value. Calculated as produciton/capacity
                raw_data (numpy.ndarray): Pointer to the actual datafile.
                region (str): Region of the dataset (market).
                temperature (numpy.ndarray): Temperature in the region in Kelvin.
                wind_speed (numpy.ndarray): Wind speed in the region in m/s.

            """
        self.region = region
        try:
            self.raw_data = h5.File(f"./{folder}/{region}.h5", "r")
        except:
            exit("error: could not open dataset './{}/{}.h5'".format(folder, region))

        # weather
        self.wind_speed = self.raw_data["weather/wind_speed"][:]
        self.temperature = self.raw_data["weather/temperature"][:]
        self.pressure = self.raw_data["weather/pressure"][:]
        # self.wind_angle = self.raw_data["weather/wind_angle"][:]

        # remove outliers
        # values larger than 25 counts for 0.06%(?) of the dataset
        # +/- 3x std
        self.wind_speed[self.wind_speed > 25] = 0
        self.temperature[self.temperature > 300] = 300
        self.temperature[self.temperature < 265] = 265

        # target
        self.production = self.raw_data["target/production"][:]
        self.capacity = self.raw_data["target/capacity"][:]
        self.ratio = self.raw_data["target/ratio"][:]

        # identify if any target values contains nan
        self.nans = np.array(
            list(
                set(np.where(np.isnan(self.production))[0])
                | set(np.where(np.isnan(self.capacity))[0])
            )
        )

        # meta
        self.idx = self.raw_data["meta/index/idx"][:]
        self.idx = np.delete(self.idx, self.nans)  # remove idx with nans
        self.mask = self.raw_data["meta/index/mask"][:]
        self.dates = self.raw_data["meta/dates/target"][:]
        self.norm_data = self.fetch_normalization_info()

    def fetch_harmonics(self, harmonics):
        harmonics = harmonics.set_index("timestamp")
        series = np.zeros(shape=(len(self.dates), 25), dtype=np.float32)
        for i in range(series.shape[0]):
            ts = pd.Timestamp(self.dates[i], unit="s")
            series[i] = harmonics.loc[ts].to_numpy().astype(np.float32)
        self.harmonics = series
        return self.harmonics

    def get_mask(self):
        return self.mask

    def fetch_normalization_info(self):
        """Fetch normaization data stored in the dataset

        This data is used to normalize the data to get values closer to the range
        [0, 1] or centered around 0 with a standard deviation of 1. This format
        represent the same situation, but is easier for a neural network to work
        with.

        Returns:
            dict: Dictionary with the collected data.

        """

        norm_data = {}
        orig = self.raw_data["meta/norm_data"]
        norm_data["wind_speed_mean_per_cell"] = orig["wind_speed_mean_per_cell"][:]
        norm_data["wind_speed_mean"] = orig["wind_speed_mean"][...]
        norm_data["wind_speed_std_per_cell"] = orig["wind_speed_std_per_cell"][:]
        norm_data["wind_speed_std"] = orig["wind_speed_std"][...]
        norm_data["wind_speed_min_per_cell"] = orig["wind_speed_min_per_cell"][:]
        norm_data["wind_speed_min"] = orig["wind_speed_min"][...]
        norm_data["wind_speed_max_per_cell"] = orig["wind_speed_max_per_cell"][:]
        norm_data["wind_speed_max"] = orig["wind_speed_max"][...]

        norm_data["temperature_mean_per_cell"] = orig["temperature_mean_per_cell"][:]
        norm_data["temperature_mean"] = orig["temperature_mean"][...]
        norm_data["temperature_std_per_cell"] = orig["temperature_std_per_cell"][:]
        norm_data["temperature_std"] = orig["temperature_std"][...]
        norm_data["temperature_min_per_cell"] = orig["temperature_min_per_cell"][:]
        norm_data["temperature_min"] = orig["temperature_min"][...]
        norm_data["temperature_max_per_cell"] = orig["temperature_max_per_cell"][:]
        norm_data["temperature_max"] = orig["temperature_max"][...]

        norm_data["pressure_mean_per_cell"] = orig["pressure_mean_per_cell"][:]
        norm_data["pressure_mean"] = orig["pressure_mean"][...]
        norm_data["pressure_std_per_cell"] = orig["pressure_std_per_cell"][:]
        norm_data["pressure_std"] = orig["pressure_std"][...]
        norm_data["pressure_min_per_cell"] = orig["pressure_min_per_cell"][:]
        norm_data["pressure_min"] = orig["pressure_min"][...]
        norm_data["pressure_max_per_cell"] = orig["pressure_max_per_cell"][:]
        norm_data["pressure_max"] = orig["pressure_max"][...]
        return norm_data

    def get_sample(self, index, args):
        """Get sample from dataset

        The dataset is indexed from ``0->dataset.length``. This method accepts a list
        of indices and return the (input, target) pair associated whit the indices. The
        data is also transformed and normalized if desired. The neural network usually
        generalize better if the data is normalized. A naive network can be avoided
        (at least with less difficuly) if the ratio (target) value is transformed
        to an even distribution.

        The different normalization strategies that can be used are:

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


        Args:
            index (numpy.ndarray): List of sample indices.
            args (argparse.Namespace): List of arguments that determine a few
                properties of how the data is handled. These properties include:


                - normalize (bool):
                    Flag to determine if data should be normalized or not.
                    (Default: True).

                - normalize_type (int):
                    Normalize type as discussed in the thesis.
                    (Default: global mean/std).

                - ratio_transform (bool):
                    Flag to determine if ratio shoud be ransformed or not. A
                    transformed ratio is more even, but at the expence of slightly
                    worse accuracy.

        Returns:
            (tuple): tuple containing:
                (torch.Tensor): The frame used as input to an ANN.
                (torch.Tensor): The target values that corresponds with the frame.
                (numpy.ndarray): The timestamps for assosiated with the data.
        """
        wind_speed = self.wind_speed[index]
        # wind_angle = self.wind_angle[index]
        temperature = self.temperature[index]
        pressure = self.pressure[index]
        target = self.ratio[index]
        norm = self.norm_data

        timestamp = self.dates[index]
        harmonics = self.harmonics[index]

        # fetch relevant normalization data
        if args.normalize_type == 1:
            wind_mean = self.norm_data["wind_speed_mean"]
            wind_std = self.norm_data["wind_speed_std"]
            temperature_mean = self.norm_data["temperature_mean"]
            temperature_std = self.norm_data["temperature_std"]
            pressure_mean = self.norm_data["pressure_mean"]
            pressure_std = self.norm_data["pressure_std"]

        elif args.normalize_type == 2:
            wind_mean = self.norm_data["wind_speed_mean_per_cell"]
            wind_std = self.norm_data["wind_speed_std_per_cell"]
            temperature_mean = self.norm_data["temperature_mean_per_cell"]
            temperature_std = self.norm_data["temperature_std_per_cell"]
            pressure_mean = self.norm_data["pressure_mean_per_cell"]
            pressure_std = self.norm_data["pressure_std_per_cell"]

        elif args.normalize_type == 3:
            wind_min = self.norm_data["wind_speed_min"]
            wind_max = self.norm_data["wind_speed_max"]
            temp_min = self.norm_data["temperature_min"]
            temp_max = self.norm_data["temperature_max"]
            pressure_min = self.norm_data["pressure_min"]
            pressure_max = self.norm_data["pressure_max"]

        elif args.normalize_type == 4:
            wind_min = self.norm_data["wind_speed_min_per_cell"]
            wind_max = self.norm_data["wind_speed_max_per_cell"]
            temp_min = self.norm_data["temperature_min_per_cell"]
            temp_max = self.norm_data["temperature_max_per_cell"]
            pressure_min = self.norm_data["pressure_min_per_cell"]
            pressure_max = self.norm_data["pressure_max_per_cell"]

        elif args.normalize:
            raise ("unknown normalization type")

        # normalize
        if args.normalize and (args.normalize_type == 1 or args.normalize_type == 2):
            wind_speed = np.divide(
                (wind_speed - wind_mean),
                wind_std,
                out=np.zeros_like(wind_speed),
                where=wind_std != 0,
            )
            temperature = np.divide(
                (temperature - temperature_mean),
                temperature_std,
                out=np.zeros_like(temperature),
                where=temperature_std != 0,
            )
            pressure = np.divide(
                (pressure - pressure_mean),
                pressure_std,
                out=np.zeros_like(pressure),
                where=pressure_std != 0,
            )

        elif args.normalize and (args.normalize_type == 3 or args.normalize_type == 4):
            w1 = wind_speed - wind_min
            w2 = wind_max - wind_min
            t1 = temperature - temp_min
            t2 = temp_max - temp_min
            p1 = pressure - pressure_min
            p2 = pressure_max - pressure_min
            wind_speed = np.divide(w1, w2, out=np.zeros_like(wind_speed), where=w2 != 0)
            temperature = np.divide(
                t1, t2, out=np.zeros_like(temperature), where=t2 != 0
            )
            pressure = np.divide(p1, p2, out=np.zeros_like(pressure), where=p2 != 0)

        frame = np.stack((wind_speed, pressure, temperature), axis=1)
        frame = frame * self.mask.astype(np.float32)
        frame = torch.from_numpy(frame)
        if args.ratio_transform:
            target = torch.from_numpy(-np.multiply(target, target) + 2 * target)
        else:
            target = torch.from_numpy(target)

        target.unsqueeze_(1)

        harmonics = torch.from_numpy(harmonics)

        return frame, target, timestamp, harmonics


class Dataset(torch.utils.data.Dataset):

    """High abstraction dataset.

    This dataset is used in the pytorch dataloader. It is simple and well behaved.
    This class impements ``__len__()`` and ``__getitem__()``.

    Attributes:
        length (int): Length of dataset
        ordinal (bool): Flag to determine if ordinal classification shoud be used.
            This will require the target values to be changed to a different
            representation.
        ordinal_resolution (int): Number of ordinal classes. Ignored if
            ``ordinal = False``.
        window_size (int): Number of hours to include in the receptive field around
            the target hour.
        x (numpy.ndarray): Inputs to the neural network.
        y (numpy.ndarray): Target values corresponding to the inputs.
    """

    def __init__(self, x, y, harmonics, args):
        self.length = len(x)
        self.x = x
        self.y = y
        self.harmonics = harmonics
        # self.bins = bins
        self.window_size = args.window_size
        self.ordinal = args.ordinal
        self.ordinal_resolution = args.ordinal_resolution
        self.transform = args.ratio_transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        x = self.x[index - self.window_size : index + self.window_size + 1]
        y = self.y[index]
        t = self.harmonics[index]

        # if self.transform:
        #    y = np.argmax(self.bins>y.item())/self.ordinal_resolution
        #    y = torch.from_numpy(np.array([y],dtype=np.float32))

        # create ordinal set. resolution 100
        if self.ordinal:
            tmp = np.zeros(self.ordinal_resolution, dtype=np.float32)
            # tmp[: np.argmax(self.bins>y.item())] = 1
            # tmp[: int(np.round(y.item() * self.ordinal_resolution))] = 1

            # The model seems to be overshooting in general. A better encoding
            # could possible be to floor the class rather than round it. This needs
            # way more testing, but it would be interesting to see at lest some results
            tmp[: int(np.floor(y.item() * self.ordinal_resolution))] = 1
            y = torch.from_numpy(tmp)

        x = np.swapaxes(x, 0, 1)  # change from (N, D, C, H, W) to (N, C, D, H, W)
        return x, y, t


class IgnoreEdgeSampler(torch.utils.data.sampler.Sampler):

    """A sampler used for torch.utils.data.DataLoader

    This sampler will makes sure that the edges are not included in the dataset target
    values if the window is greater than 1. This will make every sample a valid sample
    at the cost of a few hours loss in both ends of the dataset.

    This class implements ``__iter__()`` and ``__len__()``.

    Attributes:
        num_samples (int): Number of valid samples
        shuffle (bool): Shuffle the dataset. The window hours are not shuffled.
            (Default: True).
        window_size (int): How many hours to look before and after the target hour.
            (Default: 2).
    """

    def __init__(self, data, shuffle=False, window_size=0):
        self.window_size = window_size  # this is the number of frames to include in the hours before and after the target hour
        self.num_samples = len(data) - 2 * window_size
        self.shuffle = shuffle  # shuffle which samples are picked out in the minibatch

    def __iter__(self):
        l = np.arange(self.window_size, self.num_samples + self.window_size)
        if self.shuffle:
            np.random.shuffle(l)
        # print(l[0:5])
        return iter(l)

    def __len__(self):
        return self.num_samples


def split(array, validate, test):
    """Split the array into train-test-validate

    Example::

        validate = 0.6
        test = 0.8

        0            20%           40%           60%           80%           100%
        |-------------|-------------|-------------|-------------|-------------|
        <----------------------train-------------> <--validate-> <---test---->

    Args:
        array (numpy.ndarray): Dataset to be split.
        validate (float): At which point shoud the split between train and validate
            occur.
        test (float): At which point shoud the split between validate and test
            occur.

    Returns:
        (tuple): tuple containing:
            (numpy.ndarray): Training set
            (numpy.ndarray): Validation set
            (numpy.ndarray): Testing set

    """

    l = len(array)
    val_split = int(l * validate)
    test_split = int(l * test)

    return np.split(array, [val_split, test_split])
