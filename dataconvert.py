import numpy as np
import h5py as h5
from pprint import pprint
from datetime import datetime, timedelta
import pytz
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import os

import sys


def create_mask(lat, lon):
    """Create mask for region

    Some of the regions are not rectangular. This function create a mask that is zero
    everywhere outside of the region and one inside the region. This mask makes sure
    the region can be placed in a rectangular shape.

    Args:
        lat (numpy.ndarray): List of latitudes
        lon (numpy.ndarray): List of longitudes

    Returns:
        numpy.ndarray: Mask
    """

    h, w = len(set(lat.flatten())), len(set(lon.flatten()))
    print("lat length:", h)
    print("lon length:", w)

    mask = np.zeros(shape=(h, w))
    lat_idx = (lat.flatten() - np.min(lat.flatten())) / 0.125
    lon_idx = (lon.flatten() - np.min(lon.flatten())) / 0.125
    mask[-lat_idx.astype(int), lon_idx.astype(int)] = 1

    return mask


def convert(data, region, folder="dataset"):
    """Convert dataset from .mat to .h5

    This convert functions does a lot. It reads the datafile in .mat format and does
    the following: For each region, create a separate datafile. This datafile are
    are structured in a way that can be read by the neural net later. The regions are
    placed in a numpy matrix that is correct (relative) the actual spatial locations
    of the grid points. This will help the CNN capture local spatial correlations.
    The regions that are not rectangular are placed in a rectangular grid (the
    smallest possible) where data outside the region is set to -1 (for easier filter
    later. No other parameters produce negative numbers except wind angle, but I do
    not use this information).

    Some statistics about the data is collected and calculated. This includes
    various means and stadard deviations. The wind speed is converted to polar
    coordinates: (u, v) -> (r, theta).

    The structure of the dataset is::

        region.h5
            |
            |- weather, shape=(number of samples, datagrid at time t)
            |   |
            |   |- wind_speed
            |   |- wind_angle
            |   |- wind_u
            |   |- wind_v
            |   |- temperature
            |   `- pressure
            |
            |- target, shape(number of samples,)
            |   |
            |   |- capacity
            |   |- production
            |   `- ratio
            |
            `- meta
                |
                |- coordinates
                |   |
                |   |- lon
                |   `- lat
                |
                |- dates, shape=(number of samples,)
                |   |
                |   |- target
                |   `- weather
                |
                |- index,
                |   |
                |   |- idx          shape=(number of common timestamps,)
                |   |- idx_target   shape=(number of target timestamps,)
                |   |- idx_weather  shape=(number of weather timestamps,)
                |   `- mask         shape=shape of region
                |
                `- norm_data, over all samples. per cell has same shape as the region.
                    |
                    |- pressure_max
                    |- pressure_max_per_cell
                    |- pressure_mean
                    |- pressure_mean_per_cell
                    |- pressure_min
                    |- pressure_min_per_cell
                    |- pressure_std
                    |- pressure_std_per_cell
                    |
                    |- temperature_max
                    |- temperature_max_per_cell
                    |- temperature_mean
                    |- temperature_mean_per_cell
                    |- temperature_min
                    |- temperature_min_per_cell
                    |- temperature_std
                    |- temperature_std_per_cell
                    |
                    |- wind_speed_max
                    |- wind_speed_max_per_cell
                    |- wind_speed_mean
                    |- wind_speed_mean_per_cell
                    |- wind_speed_min
                    |- wind_speed_min_per_cell
                    |- wind_speed_std
                    `- wind_speed_std_per_cell

    Args:
        data (matlab-file): Original dataset to convert. This is a .mat file
        region (str): Name of region to convert.
        folder (str, optional): Output folder for the converted datasets.

    Returns:
        bool: Return true at the end.
    """

    f = h5.File("./{}/{}.h5".format(folder, region), "w", libver="latest")
    print("\nregion: {}".format(region))

    num_lat_points = len(data[region]["EC/properties/lat_to_keep"][:][0])

    try:
        lat = data[region]["EC/properties/lat_all"][:]
        lon = data[region]["EC/properties/lon_all"][:]
    except:
        lat = data[region]["EC/properties/lat_to_keep"][:]
        lon = data[region]["EC/properties/lon_to_keep"][:]

    mask = create_mask(lat, lon)
    region_shape = mask.shape

    f.create_dataset("meta/coordinates/lat", data=lat)
    f.create_dataset("meta/coordinates/lon", data=lon)
    norm_data = {}

    # assuming all wather parameters have the same timestamps
    datenum = data[region]["EC/Temperature_2m/dates"][:]
    datenum = datenum[0]

    naive_dt = [
        pytz.utc.localize(datetime.fromordinal(int(matlab_datenum)))
        + timedelta(days=matlab_datenum % 1)
        - timedelta(days=366)
        for matlab_datenum in datenum
    ]
    timestamp = [round(a.timestamp()) for a in naive_dt]
    datetimes = [datetime.utcfromtimestamp(a) for a in timestamp]

    print("start date: {}".format(datetimes[0]))
    print("end date  : {}".format(datetimes[-1]))
    print("all sorted: {}".format(np.all(np.diff(timestamp) > 0)))
    print("max diff  : {}".format(np.max(np.diff(timestamp))))
    print("min diff  : {}".format(np.min(np.diff(timestamp))))
    print("shape     : {}".format(region_shape))
    print("samples   : {}".format(len(timestamp)))

    f.create_dataset("meta/dates/weather", data=timestamp)
    f.create_dataset("meta/index/idx_weather", data=np.arange(len(timestamp)))

    print("compressing temperature: ....", end="\r")
    temperature = data[region]["EC/Temperature_2m/values"][:]

    temperature = temperature.T
    temperature_out = np.zeros((temperature.shape[0], *mask.shape))
    np.rot90(temperature_out, 3, axes=(1, 2))[:, mask.astype(bool).T] = temperature
    temperature_out = np.flip(temperature_out, axis=1)

    norm_data["temperature_mean_per_cell"] = np.mean(temperature_out, axis=0)
    norm_data["temperature_mean"] = np.mean(
        temperature_out[np.where(temperature_out > 0)]
    )
    norm_data["temperature_std_per_cell"] = np.std(temperature_out, axis=0)
    norm_data["temperature_std"] = np.std(
        temperature_out[np.where(temperature_out > 0)]
    )
    norm_data["temperature_min_per_cell"] = np.min(temperature_out, axis=0)
    norm_data["temperature_min"] = np.min(
        temperature_out[np.where(temperature_out > 0)]
    )
    norm_data["temperature_max_per_cell"] = np.max(temperature_out, axis=0)
    norm_data["temperature_max"] = np.max(
        temperature_out[np.where(temperature_out > 0)]
    )

    f.create_dataset(
        "weather/temperature",
        data=temperature_out.astype(np.float32),
        chunks=temperature_out.shape,
        compression="lzf",
    )
    print("compressing temperature: done")

    print("compressing pressure: ....", end="\r")
    pressure = data[region]["EC/Pressure_reduced_to_MSL/values"][:]
    pressure = pressure.T
    pressure_out = np.zeros((pressure.shape[0], *mask.shape))
    np.rot90(pressure_out, 3, axes=(1, 2))[:, mask.astype(bool).T] = pressure
    pressure_out = np.flip(pressure_out, axis=1)

    norm_data["pressure_mean_per_cell"] = np.mean(pressure_out, axis=0)
    norm_data["pressure_mean"] = np.mean(pressure_out[np.where(pressure_out > 0)])
    norm_data["pressure_std_per_cell"] = np.std(pressure_out, axis=0)
    norm_data["pressure_std"] = np.std(pressure_out[np.where(pressure_out > 0)])
    norm_data["pressure_min_per_cell"] = np.min(pressure_out, axis=0)
    norm_data["pressure_min"] = np.min(pressure_out[np.where(pressure_out > 0)])
    norm_data["pressure_max_per_cell"] = np.max(pressure_out, axis=0)
    norm_data["pressure_max"] = np.max(pressure_out[np.where(pressure_out > 0)])

    f.create_dataset(
        "weather/pressure",
        data=pressure_out.astype(np.float32),
        chunks=pressure_out.shape,
        compression="lzf",
    )
    print("compressing pressure: done")

    print("compressing wind_u: ....", end="\r")
    wind_u = data[region]["EC/U_component_of_wind_100m/values"][:]
    wind_u = wind_u.T
    wind_u_out = np.zeros((wind_u.shape[0], *mask.shape))
    np.rot90(wind_u_out, 3, axes=(1, 2))[:, mask.astype(bool).T] = wind_u
    wind_u_out = np.flip(wind_u_out, axis=1)
    f.create_dataset(
        "weather/wind_u",
        data=wind_u_out.astype(np.float32),
        chunks=wind_u_out.shape,
        compression="lzf",
    )
    print("compressing wind_u: done")

    print("compressing wind_v: ....", end="\r")
    wind_v = data[region]["EC/V_component_of_wind_100m/values"][:]
    wind_v = wind_v.T
    wind_v_out = np.zeros((wind_v.shape[0], *mask.shape))
    np.rot90(wind_v_out, 3, axes=(1, 2))[:, mask.astype(bool).T] = wind_v
    wind_v_out = np.flip(wind_v_out, axis=1)
    f.create_dataset(
        "weather/wind_v",
        data=wind_v_out.astype(np.float32),
        chunks=wind_v_out.shape,
        compression="lzf",
    )
    print("compressing wind_v: done")

    print("compressing speed and angle: ....", end="\r")
    wind_speed = np.sqrt(wind_u_out ** 2 + wind_v_out ** 2)
    wind_angle = np.arctan2(wind_u_out, wind_v_out)

    wind_speed = wind_speed * mask
    wind_angle = wind_angle * mask

    norm_data["wind_speed_mean_per_cell"] = np.mean(wind_speed, axis=0)
    norm_data["wind_speed_mean"] = np.mean(wind_speed[np.where(wind_speed > 0)])
    norm_data["wind_speed_std_per_cell"] = np.std(wind_speed, axis=0)
    norm_data["wind_speed_std"] = np.std(wind_speed[np.where(wind_speed > 0)])
    norm_data["wind_speed_min_per_cell"] = np.min(wind_speed, axis=0)
    norm_data["wind_speed_min"] = np.min(wind_speed[np.where(wind_speed > 0)])
    norm_data["wind_speed_max_per_cell"] = np.max(wind_speed, axis=0)
    norm_data["wind_speed_max"] = np.max(wind_speed[np.where(wind_speed > 0)])

    f.create_dataset(
        "weather/wind_speed",
        data=wind_speed.astype(np.float32),
        chunks=wind_speed.shape,
        compression="lzf",
    )
    f.create_dataset(
        "weather/wind_angle",
        data=wind_angle.astype(np.float32),
        chunks=wind_angle.shape,
        compression="lzf",
    )
    print("compressing speed and angle: done")

    # handle Producition

    # assuming all production parameters have the same timestamps
    datenum_2 = data[region]["production/dates"][:][0]
    datenum_3 = data[region]["capacity/dates"][:][0]

    if np.max(np.abs(datenum_2 - datenum_3)) > 0.1:
        print("date error: production and capacity have different dates")
        exit()

    if np.max(np.abs(datenum - datenum_2)) > 0.1:
        print("date error: production/capacity and weather have different dates")
        exit()

    print("compressing metadata: ....", end="\r")
    f.create_dataset("meta/index/mask", data=mask)
    for key, value in norm_data.items():
        f.create_dataset(
            f"meta/norm_data/{key}", data=np.array(value).astype(np.float32)
        )

    f.create_dataset("meta/dates/target", data=timestamp)
    f.create_dataset("meta/index/idx_target", data=np.arange(len(timestamp)))
    f.create_dataset("meta/index/idx", data=np.arange(len(timestamp)))

    capacity = data[region]["capacity/values"][:][0]
    production = data[region]["production/values"][:][0]
    ratio = production / capacity

    f.create_dataset("target/capacity", data=capacity.astype(np.float32))
    f.create_dataset("target/production", data=production.astype(np.float32))
    f.create_dataset("target/ratio", data=ratio.astype(np.float32))
    print("compressing metadata: done")

    f.close()
    return True


def main():
    try:
        f = h5.File(sys.argv[1], "r")
        print("opened file: {}".format(sys.argv[1]))
    except:
        print("usage: python dataconvert.py <file to convert>")
        print("\nconvert .mat file from Gabriele to the h5 format used for my model")

    data = f["modelData"]

    regions = list(data.keys())
    print("regions: {}".format(regions))

    folder = "dataset"
    if not os.path.exists(folder):
        os.mkdir(folder)
    # convert region to a new file:
    for region in regions:
        convert(data, region, folder)

    f.close()
    print("file closed")


if __name__ == "__main__":
    main()
