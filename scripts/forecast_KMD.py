import os
import sys
import pathlib
import yaml

import netCDF4 as nc
from cftime import date2num
import xarray as xr
import numpy as np
from tensorflow.keras.utils import Progbar
from scipy.interpolate import RegularGridInterpolator


sys.path.insert(1,"../")
from data.data import (
    HOURS,
    all_fcst_fields,
    nonnegative_fields,
    accumulated_fields,
    fcst_norm,
    logprec,
    denormalise,
    load_hires_constants,
)
from config import set_gpu_mode, get_data_paths, read_downscaling_factor
from setupmodel import setup_model
from model.noise import NoiseGenerator
import tensorflow as tf

from datetime import datetime, timedelta

# %%
# Define the latitude and longitude arrays for later
latitude = np.arange(-13.65, 24.7, 0.1)
longitude = np.arange(19.15, 54.3, 0.1)

# Some setup
set_gpu_mode()  # set up whether to use GPU, and mem alloc mode
data_paths = get_data_paths()  # need the constants directory
downscaling_steps = read_downscaling_factor()["steps"]
assert fcst_norm is not None

TRUTH_PATH = data_paths["GENERAL"]["TRUTH_PATH"]
ds_truth_test = xr.open_dataset(TRUTH_PATH+'2018/rr_adj_20180107.nc')
lons_truth = ds_truth_test.Lon.values
lats_truth = ds_truth_test.Lat.values

# %%
# Open and parse forecast.yaml
fcstyaml_path = "../config/forecast.yaml"
with open(fcstyaml_path, "r") as f:
    try:
        fcst_params = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)

# %%
model_folder = fcst_params["MODEL"]["folder"]
checkpoint = fcst_params["MODEL"]["checkpoint"]
input_folder = fcst_params["INPUT"]["folder"]
#dates = fcst_params["INPUT"]["dates"]
start_hour = fcst_params["INPUT"]["start_hour"]
end_hour = fcst_params["INPUT"]["end_hour"]
output_folder = fcst_params["OUTPUT"]["folder"]
ensemble_members = fcst_params["OUTPUT"]["ensemble_members"]

assert start_hour % HOURS == 0, f"start_hour must be divisible by {HOURS}"
assert end_hour % HOURS == 0, f"end_hour must be divisible by {HOURS}"

# Open and parse GAN config file
config_path = os.path.join(model_folder, "setup_params.yaml")
with open(config_path, "r") as f:
    try:
        setup_params = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)

mode = setup_params["GENERAL"]["mode"]
arch = setup_params["MODEL"]["architecture"]
padding = setup_params["MODEL"]["padding"]
filters_gen = setup_params["GENERATOR"]["filters_gen"]
noise_channels = setup_params["GENERATOR"]["noise_channels"]
latent_variables = setup_params["GENERATOR"]["latent_variables"]
filters_disc = setup_params["DISCRIMINATOR"]["filters_disc"]  # TODO: avoid setting up discriminator in forecast mode?
constant_fields = 2

assert mode == "GAN", "standalone forecast script only for GAN, not VAE-GAN or deterministic model"

# Set up pre-trained GAN
weights_fn = os.path.join(model_folder, "models", f"gen_weights-{checkpoint:07}.h5")
input_channels = 2*len(all_fcst_fields)

model = setup_model(mode=mode,
                    arch=arch,
                    downscaling_steps=downscaling_steps,
                    input_channels=input_channels,
                    constant_fields=constant_fields,
                    filters_gen=filters_gen,
                    filters_disc=filters_disc,
                    noise_channels=noise_channels,
                    latent_variables=latent_variables,
                    padding=padding)
gen = model.gen
gen.load_weights(weights_fn)

network_const_input = load_hires_constants(batch_size=1)  # 1 x lats x lons x 2

# %%
def create_output_file(nc_out_path):
    netcdf_dict = {}
    rootgrp = nc.Dataset(nc_out_path, "w", format="NETCDF4")
    netcdf_dict["rootgrp"] = rootgrp
    rootgrp.description = "GAN 24-hour rainfall ensemble members in the ICPAC region."

    # Create output file dimensions
    rootgrp.createDimension("latitude", len(latitude))
    rootgrp.createDimension("longitude", len(longitude))
    rootgrp.createDimension("member", ensemble_members)
    rootgrp.createDimension("time", None)
    rootgrp.createDimension("valid_time", None)

    # Create variables
    latitude_data = rootgrp.createVariable("latitude",
                                           "f4",
                                           ("latitude",))
    latitude_data.units = "degrees_north"
    latitude_data[:] = latitude     # Write the latitude data

    longitude_data = rootgrp.createVariable("longitude",
                                            "f4",
                                            ("longitude",))
    longitude_data.units = "degrees_east"
    longitude_data[:] = longitude   # Write the longitude data

    ensemble_data = rootgrp.createVariable("member",
                                           "i4",
                                           ("member",))
    ensemble_data.units = "ensemble member"
    ensemble_data[:] = range(1, ensemble_members+1)

    netcdf_dict["time_data"] = rootgrp.createVariable("time",
                                                      "f4",
                                                      ("time",))
    netcdf_dict["time_data"].units = "hours since 1900-01-01 00:00:00.0"

    netcdf_dict["valid_time_data"] = rootgrp.createVariable("fcst_valid_time",
                                                            "f4",
                                                            ("time", "valid_time"))
    netcdf_dict["valid_time_data"].units = "hours since 1900-01-01 00:00:00.0"

    netcdf_dict["precipitation"] = rootgrp.createVariable("precipitation",
                                                          "f4",
                                                          ("time", "member", "valid_time",
                                                           "latitude", "longitude"),
                                                          compression="zlib",
                                                          chunksizes=(1, 1, 1, len(latitude), len(longitude)))
    netcdf_dict["precipitation"].units = "mm h**-1"
    netcdf_dict["precipitation"].long_name = "Precipitation"

    return netcdf_dict

ds_truth_test = xr.open_dataset(TRUTH_PATH+'2018/rr_adj_20180107.nc')
def interp_to_rfe(lats,lons,data):
    interp = RegularGridInterpolator((lats, lons), data)
    lats_truth_grid, lons_truth_grid = np.meshgrid(lats_truth, lons_truth, indexing='ij')
    return interp((lats_truth_grid, lons_truth_grid))


# %%
def make_fcst(input_folder=input_folder, output_folder=output_folder,
              start_hour=start_hour,end_hour=end_hour,HOURS=HOURS,
              all_fcst_fields=all_fcst_fields,nonnegative_fields=nonnegative_fields,
              gen=gen,ensemble_members=ensemble_members):
    dates = np.arange('2023-06-01','2024-05-31',np.timedelta64(1,'D'),dtype='datetime64[ns]')
    valid_times = np.arange(start_hour,end_hour+1,HOURS)
    for day in dates:
        d = datetime(day.astype('datetime64[D]').astype(object).year,
                     day.astype('datetime64[D]').astype(object).month,
                     day.astype('datetime64[D]').astype(object).day)
                    
        print(f"{d.year}-{d.month:02}-{d.day:02}")
        
        # %%
        # Specify input folder for year
        if d.year == 2024:
            if d.month == 1:
                input_folder_year = input_folder+f"2024_Jan/"
            elif d.month == 2:
                input_folder_year = input_folder+f"2024_Feb/"
            elif d.month in [3,4]:
                input_folder_year = input_folder+f"2024_MarApr/"
            elif d.month == 5:
                input_folder_year = input_folder+f"2024_May/"
            else:
                input_folder_year = input_folder+f"2024/"
        else:
            input_folder_year = input_folder+f"{d.year}/"
        
        # Create output netCDF file
        output_folder_year = output_folder+f"{d.year}/"
        pathlib.Path(output_folder_year).mkdir(parents=True, exist_ok=True)
        nc_out_path = os.path.join(output_folder_year, f"GAN_{d.year}{d.month:02}{d.day:02}.nc")
        
        netcdf_dict = create_output_file(nc_out_path)
        netcdf_dict["time_data"][0] = [date2num(d,units="hours since 1900-01-01 00:00:00.0")]
        
        # loop over time chunks. output forecasts may not start from hour 0, so
        # generate output and input valid time indices using enumerate(...)
        for out_time_idx, in_time_idx in enumerate(range(start_hour//24, end_hour//24)):
            # copy across valid_time from input file
            netcdf_dict["valid_time_data"][0, out_time_idx] = date2num(d+timedelta(hours=int(valid_times[out_time_idx])),units="hours since 1900-01-01 00:00:00.0")
           
            field_arrays = []
        
            # the contents of the next loop are v. similar to load_fcst from data.py,
            # but not quite the same, since that has different assumptions on how the
            # forecast data is stored.  TODO: unify the data normalisation between these?
            for field in all_fcst_fields:
                # Original:
                # nc_in[field] has shape 1 x 5 x 29 x 384 x 352
                # corresponding to n_forecasts x n_ensemble_members x n_valid_times x n_lats x n_lons
                # Ensemble mean:
                # nc_in[field] has shape len(nc_in["time"]) x 29 x 384 x 352
                
                # Open input netCDF file
                input_file = f"{field}.nc"
                nc_in_path = os.path.join(input_folder_year, input_file)
                #Specify lead_idx for 30 hour lead time it is 5 (30/6)
                lead_idx = 5
                # open using netCDF
                nc_file = xr.open_dataset(nc_in_path)
                lats = nc_file["latitude"].values
                lons = nc_file["longitude"].values
                all_data_mean = nc_file[f"{field}_mean"]
                all_data_sd = nc_file[f"{field}_sd"]
                # data is stored as [day of year, valid time index, lat, lon]

                # calculate first index (i.e., day of year, with Jan 1 = 0)
                fcst_date = datetime.datetime.strptime(day, "%Y%m%d").date()
                fcst_idx = fcst_date.toordinal() - datetime.date(day.astype('datetime64[D]').astype(object).year, 1, 1).toordinal()

                if field in accumulated_fields:
                    # return mean, sd, 0, 0.  zero fields are so that each field returns a 4 x ny x nx array.
                    # accumulated fields have been pre-processed s.t. data[:, j, :, :] has accumulation between times j and j+1
                    data1 = np.mean(all_data_mean[fcst_idx, lead_idx:lead_idx+4, :, :], axis=0)            # Mean of the accumulations
                    data2 = np.sqrt(np.mean(all_data_sd[fcst_idx, lead_idx:lead_idx+4, :, :]**2, axis=0))  # RMS of the standard deviations
                    #interpolate to rfe grid
                    data1 = interp_to_rfe(lats,lons,data1)
                    data2 = interp_to_rfe(lats,lons,data2)
                    data = np.stack([data1, data2], axis=-1)
                else:
                    # return mean and std computed using the trapezium rule
                    temp_data_mean = all_data_mean[fcst_idx, lead_idx:lead_idx+5, :, :]
                    temp_data_var = all_data_sd[fcst_idx, lead_idx:lead_idx+5, :, :]**2  # Convert to variances
                    data1 = (temp_data_mean[0, :, :]/2 + np.sum(temp_data_mean[1:4,:,:], axis=0) + temp_data_mean[4,:,:]/2)/4
                    data2 = (temp_data_var[0, :, :]/2 + np.sum(temp_data_var[1:4,:,:], axis=0) + temp_data_var[4,:,:]/2)/4
                    #interpolate to rfe grid
                    data1 = interp_to_rfe(lats,lons,data1)
                    data2 = interp_to_rfe(lats,lons,np.sqrt(data2))
                    data = np.stack([data1, data2], axis=-1)

                nc_file.close()


                if field in nonnegative_fields:
                    data = np.maximum(data, 0.0)  # eliminate any data weirdness/regridding issues

                if field in ["tp", "cp"]:
                    # precip is measured in metres, so multiply to get mm
                    data *= 1000
                    data /= HOURS  # convert to mm/hr
                elif field in accumulated_fields:
                    # for all other accumulated fields [just ssr for us]
                    data /= (HOURS*3600)  # convert from a 6-hr difference to a per-second rate

                if field in ["tp", "cp"]:
                    return logprec(data, True)
                # apply transformation to make fields O(1), based on historical
                # forecast data from one of the training years
                if fcst_norm is None:
                    raise RuntimeError("Forecast normalisation dictionary has not been loaded")
                elif field in ["sp", "t2m"]:
                    # these are bounded well away from zero, so subtract mean from ens mean (but NOT from ens sd!)
                    data[:, :, 0] -= fcst_norm[field]["mean"]
                    #data[:, :, 2] -= fcst_norm[field]["mean"]
                    data = data/fcst_norm[field]["std"]
                elif field in nonnegative_fields:
                    data = data/fcst_norm[field]["max"]
                else:
                    # winds
                    data = data/max(-fcst_norm[field]["min"], fcst_norm[field]["max"])
                field_arrays.append(data)

            network_fcst_input = np.concatenate(field_arrays, axis=-1)  # lat x lon x 4*len(all_fcst_fields)
            network_fcst_input = np.expand_dims(network_fcst_input, axis=0)  # 1 x lat x lon x 4*len(...)
            noise_shape = network_fcst_input.shape[1:-1] + (noise_channels,)
            noise_gen = NoiseGenerator(noise_shape, batch_size=1)
            
            #print(network_fcst_input.shape,network_const_input.shape,noise_gen().shape)
            progbar = Progbar(ensemble_members)
            for ii in range(ensemble_members):
                gan_inputs = [network_fcst_input, network_const_input, noise_gen()]
                gan_prediction = gen.predict(gan_inputs, verbose=False)  # 1 x lat x lon x 1
                netcdf_dict["precipitation"][0, ii, out_time_idx, :, :] = denormalise(gan_prediction[0, :, :, 0])
                progbar.add(1)
            
        netcdf_dict["rootgrp"].close()

if __name__=="__main__":
    
    make_fcst()
    
