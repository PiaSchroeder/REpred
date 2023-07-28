### Get weather data from open meteo ###########################################################################
import pandas as pd
import numpy as np
import os
import requests
from datetime import datetime as dt, timedelta


### Constants ##################################################################################################

WEATHER_ENDPOINT_HISTORICAL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_ENDPOINT_FORECAST = "https://api.open-meteo.com/v1/forecast"

# The historical and forecast APIs have slightly different wind features (100m vs. 120m height)
HOURLY_VARIABLES_HISTORICAL = ["temperature_2m",
                    "relativehumidity_2m",
                    "precipitation",
                    "cloudcover",
                    "shortwave_radiation",
                    "direct_radiation",
                    "direct_normal_irradiance",
                    "diffuse_radiation",
                    "windspeed_10m",
                    "windspeed_100m",
                    "winddirection_10m",
                    "winddirection_100m",
                    "windgusts_10m",
                    ]
HOURLY_VARIABLES_FORECAST = ["temperature_2m",
                    "relativehumidity_2m",
                    "precipitation",
                    "cloudcover",
                    "shortwave_radiation",
                    "direct_radiation",
                    "direct_normal_irradiance",
                    "diffuse_radiation",
                    "windspeed_10m",
                    "windspeed_120m",
                    "winddirection_10m",
                    "winddirection_120m",
                    "windgusts_10m",
                    ]



### Functions ##################################################################################################

def downcast_dtypes(df):
    '''
    Downcast dtypes to save time and space.
    '''
    
    downcast_df = df.copy()
    
    try:
        downcast_df["temperature_2m"] = downcast_df["temperature_2m"].astype("float32")
        downcast_df["precipitation"] = downcast_df["precipitation"].astype("float32")
        downcast_df["shortwave_radiation"] = downcast_df["shortwave_radiation"].astype("float32")
        downcast_df["direct_radiation"] = downcast_df["direct_radiation"].astype("float32")
        downcast_df["direct_normal_irradiance"] = downcast_df["direct_normal_irradiance"].astype("float32")
        downcast_df["diffuse_radiation"] = downcast_df["diffuse_radiation"].astype("float32")
        downcast_df["windspeed_10m"] = downcast_df["windspeed_10m"].astype("float32")
        downcast_df["windspeed_100m"] = downcast_df["windspeed_100m"].astype("float32")
        downcast_df["windgusts_10m"] = downcast_df["windgusts_10m"].astype("float32")
    except:
        print("Float downcast failed. Data might contain nans.")
    
    try:
        downcast_df["cloudcover"] = downcast_df["cloudcover"].astype("int8")
        downcast_df["relativehumidity_2m"] = downcast_df["relativehumidity_2m"].astype("int8")
        downcast_df["winddirection_10m"] = downcast_df["winddirection_10m"].astype("int16")
        downcast_df["winddirection_100m"] = downcast_df["winddirection_100m"].astype("int16")
    except:
        print("Int downcast failed. Data might contain nans.")    
    
    return downcast_df 


##########

def get_weather_data(cluster_df, save_path, energy_type, start, end, timezone):
    '''
    Request historical and forecast weather data from open meteo for each cluster for given time period and average weather weighted by cluster 
    capacities.
    '''
    
    print("Retrieving weather data...")
    
    save_path = f"{save_path}/weather/{energy_type}/{cluster_df.shape[0]}_clusters"
    if not os.path.isdir(save_path): 
        os.makedirs(save_path)
        
    end = dt.strptime(end, "%Y-%m-%d")
    start = dt.strptime(start, "%Y-%m-%d")
    
    cluster_df = cluster_df.copy()
    
    # Split api calls into two parts: historical and forecast
    today = dt.now().replace(hour=0, minute=0, second=0, microsecond=0)
    cutoff = today - timedelta(days=7)
    
    # forecast only (includes now - 7 days)
    if start > cutoff:
        start_hist = None
        end_hist = None
        start_fc = start
        end_fc = min(end, today+timedelta(days=7)) # Forecast max 7 days into the future
    # start in historical data, end in forecast
    elif (start <= cutoff) & (end > cutoff):
        start_hist = start
        end_hist = cutoff
        start_fc = cutoff+timedelta(days=1)
        end_fc = min(end, today+timedelta(days=7)) # Forecast max 7 days into the future
    # historical only
    elif end <= cutoff:
        start_hist = start
        end_hist = end
        start_fc = None
        end_fc = None
    
        
    # Loop over clusters
    weather_frames = []
    for idx, row in cluster_df.iterrows():

        print(f"{idx+1}/{cluster_df.shape[0]}", end="\r")

        # Get cluster coordinates
        lat_long = row[["Breitengrad", "Laengengrad"]].values

        ### Get historical weather data
        if start_hist is not None:
            
            params = {
                "start_date": start_hist.strftime("%Y-%m-%d"),
                "end_date": end_hist.strftime("%Y-%m-%d"),
                "hourly": HOURLY_VARIABLES_HISTORICAL,
                "timezone": timezone,
            }

            # Make historical API request
            response = requests.get(url=WEATHER_ENDPOINT_HISTORICAL, params=(params|{"latitude": lat_long[0], "longitude": lat_long[1]}))
            response.raise_for_status()

            try:
                # Create data frame from API response
                weather_df_hist = pd.DataFrame(response.json()["hourly"])

                # Set timestamp as index
                weather_df_hist["time"] = pd.to_datetime(weather_df_hist["time"])
                weather_df_hist = weather_df_hist.set_index("time")

            except:
                print(f"Something went wrong for cluster {idx}. Skipping cluster...")
                cluster_df = cluster_df.loc[cluster_df["cluster_label"]!=idx].copy()
                continue
                
        else:
            weather_df_hist = pd.DataFrame()
        
        ### Get weather forecast data
        if start_fc is not None:
            
            params = {
                "start_date": start_fc.strftime("%Y-%m-%d"),
                "end_date": end_fc.strftime("%Y-%m-%d"),
                "hourly": HOURLY_VARIABLES_FORECAST,
                "timezone": timezone,
            }

            # Make forecast API request
            response = requests.get(url=WEATHER_ENDPOINT_FORECAST, params=(params|{"latitude": lat_long[0], "longitude": lat_long[1]}))
            response.raise_for_status()

            try:
                # Create data frame from API response
                weather_df_fc = pd.DataFrame(response.json()["hourly"])
            except:
                print(f"Something went wrong with the API call for cluster {idx}. Skipping cluster...")
                cluster_df = cluster_df.loc[cluster_df["cluster_label"]!=idx].copy()
                continue
                
            # Set timestamp as index
            weather_df_fc["time"] = pd.to_datetime(weather_df_fc["time"])
            weather_df_fc = weather_df_fc.set_index("time")

            # Rename columns (wind data for 100m and 120m shouldn't differ to much, ok approximation for my purposes)
            weather_df_fc = weather_df_fc.rename(columns={"windspeed_120m":"windspeed_100m", "winddirection_120m":"winddirection_100m"})
            
        else:
            weather_df_fc = pd.DataFrame()

                
        # Concatenate historical and forecast dfs
        weather_df = pd.concat([weather_df_hist, weather_df_fc], axis=0)
        
        # Downcast data types
        weather_df = downcast_dtypes(weather_df)

        # Save data frame
        weather_df.to_parquet(f"{save_path}/weather_data_{idx}.parquet")

        # Add data frame to collection for later averaging
        weather_frames.append(weather_df)
            
        
    # Get weighted average
    weights = np.array(cluster_df.total_capacity / cluster_df.total_capacity.sum())
    
    # Initialize 3-dimensional NumPy array
    data_array = np.empty((weather_frames[0].shape[0], weather_frames[0].shape[1], len(weather_frames)))

    # Fill NumPy array with weather data
    for idx, df in enumerate(weather_frames):
        data_array[:, :, idx] = df.values

    # Compute weighted average
    weighted_average = np.average(data_array, axis=2, weights=weights)

    # Fill new data frame
    weighted_weather = pd.DataFrame(index=weather_frames[0].index, columns=weather_frames[0].columns, data=weighted_average)
    
    # Save average weather
    weighted_weather.to_parquet(f"{save_path}/weather_data_weighted_avg.parquet")
    
    print("\nDone!")
    
    return weighted_weather