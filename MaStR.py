### Extract data from MaStR (Marktstammdatenregister) #############################################################
import pandas as pd
import numpy as np
import os
from datetime import datetime as dt
import requests
import keys
from regional_info import zip_list, zip_data

### Constants #####################################################################################################
MaStR_FPATH    = "data/MaStR_Gesamtdatenexport_20230621/"
MaStR_ENCODING = "utf-16"
MAPS_ENDPOINT  = "https://maps.googleapis.com/maps/api/geocode/json"
MAPS_API_KEY   = keys.MAPS_API_KEY
LAT_LENGTH     = 111.
LONG_LENGTH    = 67.

### Data info ####################################################################################################
MaStR_INFO = {
    "solar": {
        "facility_files": [f"{MaStR_FPATH}AnlagenEegSolar_{idx}.xml" for idx in range(1,32)],
        "unit_files": [f"{MaStR_FPATH}EinheitenSolar_{idx}.xml" for idx in range(1,33)],
        "facility_features": ["EegMaStRNummer", 
                              "InstallierteLeistung"
                             ],
        "unit_features":     ["EegMaStRNummer",
                              "Inbetriebnahmedatum",
                              "Bundesland",
                              "Landkreis",
                              "Gemeinde",
                              "Gemeindeschluessel",
                              "Postleitzahl",
                              "Ort",
                              "Laengengrad",
                              "Breitengrad",
                              "Bruttoleistung",
                              "Nettonennleistung"
                             ],
        
    },
    "wind": {
        "facility_files": [f"{MaStR_FPATH}AnlagenEegWind.xml"],
        "unit_files": [f"{MaStR_FPATH}EinheitenWind.xml"],
        "facility_features": ["EegMaStRNummer", 
                              "InstallierteLeistung"
                             ],
        "unit_features":     ["EegMaStRNummer",
                              "Inbetriebnahmedatum",
                              "Bundesland",
                              "Landkreis",
                              "Gemeinde",
                              "Gemeindeschluessel",
                              "Postleitzahl",
                              "Ort",
                              "Laengengrad",
                              "Breitengrad",
                              "Bruttoleistung",
                              "Nettonennleistung",
                              "ClusterOstsee"
                             ],
    }
}


### Functions ####################################################################################################

def get_MaStR(energy_type, save_path=None, subtype=None):
    '''
    Main function to extract MaStR data for energy type passed in energy_type. Saves and returns cleaned and filtered 
    data frame.
    '''
    
    assert energy_type in MaStR_INFO.keys(), f"Unknown energy type! Allowed types: {list(MaStR_INFO.keys())}"
    
    if not save_path:
        save_path = os.getcwd()
        os.makedirs(f"MaStR_data", exist_ok=True)
    elif not os.path.isdir(f"{save_path}/MaStR_data"):
        os.makedirs(f"{save_path}/MaStR_data")
        
    if subtype:
        addon = f"_{subtype}"
    else:
        addon = ""
    
    # get facilities
    print(f"Loading {energy_type} facilities...")
    facility_df = get_df(files=MaStR_INFO[energy_type]["facility_files"], \
                         features=MaStR_INFO[energy_type]["facility_features"])
    facility_df.to_parquet(f"{save_path}/MaStR_data/{energy_type}{addon}_facility_df_{dt.now().strftime('%Y%m%d')}.parquet")
    
    # get units
    print(f"Loading {energy_type} units...")
    unit_df = get_df(files=MaStR_INFO[energy_type]["unit_files"], features=MaStR_INFO[energy_type]["unit_features"])
    unit_df.to_parquet(f"{save_path}/MaStR_data/{energy_type}{addon}_unit_df_{dt.now().strftime('%Y%m%d')}.parquet")
    
    # Select units in area of interest (50Hertz grid)
    if subtype == "offshore":
        # Offshore parks have no zip code
        unit_df = unit_df.loc[~unit_df["ClusterOstsee"].isna()].copy() 
    else:
        # Select by zip code 
        unit_df = unit_df.loc[unit_df["Postleitzahl"].isin(zip_list)].copy()
    
    # merge units and facilities
    mastr_df = unit_df.merge(facility_df, on="EegMaStRNummer", how="inner").reset_index(drop=True)
    
    # clean data
    print("Cleaning data...")
    mastr_df = clean_data(mastr_df)
    
    # get coordinates
    if mastr_df["Breitengrad"].isna().any():
        
        print(f"Retrieving missing coordinates...")
        
        # Get missing coordinates from google maps API
        lat_lon_lookup = lookup_latlongs(mastr_df)
        lat_lon_lookup.to_parquet(\
            f"{save_path}/MaStR_data/{energy_type}{addon}_geo_lookup_{dt.now().strftime('%Y%m%d')}.parquet")
    
        # Fix coordinates
        mastr_df = fix_coordinates(mastr_df, lat_lon_lookup)
    
    # Save data frame
    print(f"Saving {energy_type}{addon} data frame to disk...")
    mastr_df.to_parquet(f"{save_path}/MaStR_data/{energy_type}{addon}_MaStR_{dt.now().strftime('%Y%m%d')}.parquet")
    
    print("Done!")
    
    return mastr_df


##########

def get_df(files, features):
    '''
    Load files and return aggregate data frame holding features of interest of active facilities/units. 
    '''
    
    df = pd.DataFrame(columns=features)

    # Loop through files
    for idx, file in enumerate([files[0]]):

        # Print progress
        print(f"Loading file {idx+1}/{len(files)}", end="\r")
        
        # Load file, filter and extract relevant information
        new_page = (pd.read_xml(file, encoding=MaStR_ENCODING)\
                      .rename(columns={"EinheitBetriebsstatus": "Betriebsstatus", \
                                       "AnlageBetriebsstatus": "Betriebsstatus"}))
        new_page = new_page.loc[new_page["Betriebsstatus"] == 35, features]

        # Add new data to data frame
        df = pd.concat([df, new_page])

    df = df.reset_index(drop=True)

    # Check for duplicates
    assert df["EegMaStRNummer"].nunique() == df.shape[0], "Duplicate units!"

    print("\nDone!")
    
    return df


##########

def clean_data(df):
    '''
    Perform basic cleaning of data frame, including recoding of zip codes, dtype conversion, and removal of unnecessary 
    columns.
    '''
    
    df = df.copy()
    
    # Remove Insel Neuwerk (presumably not served by 50Hertz)
    df = df.loc[~(df["Ort"]=="Hamburg-Insel Neuwerk")].copy()

    if not df["Postleitzahl"].isna().all():
        # Recode zip codes: Lookup zip codes in zip_data because codes provided in Marktstammdatenregister contain errors
        df = df.merge(zip_data, on=["Postleitzahl"], suffixes=["", "_lookup"], how="left")
        df["Bundesland"] = df["Bundesland_lookup"]
        df = df.drop(columns=["Bundesland_lookup"])

        # Delete rows with missing data
        df = df.loc[~df["Landkreis"].isna()].copy()
        
        # Convert from float to int (removes trailing dots) to string
        df["Gemeindeschluessel"] = df["Gemeindeschluessel"].astype("int").astype("str").str.zfill(8)
        df["Postleitzahl"] = df["Postleitzahl"].astype("int").astype("str").str.zfill(5)

    # Convert to datetime
    df["Inbetriebnahmedatum"] = pd.to_datetime(df["Inbetriebnahmedatum"])

    # Convert to category (saves space)
    df["Bundesland"] = df["Bundesland"].astype("category")
    
    # Drop offshore column
    try: df = df.drop(columns={"ClusterOstsee"}) 
    except: pass
    
    return df


##########

def get_lat_long(zip_code, place, district, country="DE"):
    '''
    Make API calls to Google Maps API to get and return geo locations of places passed in arguments.
    '''
    
    params = {
        "key": MAPS_API_KEY,
        "address": f"{place}, {zip_code}, {district}, {country}"
    }
    response = requests.get(url=MAPS_ENDPOINT, params=params)
    response.raise_for_status()
    try:
        lat_long = response.json()["results"][0]["geometry"]["location"]
    except:
        print(f"No results for {zip_code} {place}")
        lat_long = {"lat": np.nan, "lng": np.nan}
    
    return (lat_long["lat"], lat_long["lng"])


##########

def lookup_latlongs(df):
    '''
    Get and return geo locations for all place-zipcode-district combinations in df (to be used as lookup table).
    '''
    
    # Create Dataframe holding all place-zipcode-district combinations 
    geo_lookup = df[["Postleitzahl", "Ort", "Landkreis"]].drop_duplicates()
    geo_lookup[["Breitengrad", "Laengengrad"]] = np.nan
    geo_lookup = geo_lookup.reset_index(drop=True)

    # Use slow for loop here in case API requests fail at some point (e.g. exceeded quota)
    # This way, the previous requests won't be lost (go into debugger and save)
    for idx, row in geo_lookup.iterrows():
        print(f"{idx+1}/{geo_lookup.shape[0]}", end="\r")
        lat_long = get_lat_long(row["Postleitzahl"], row["Ort"], row["Landkreis"])
        geo_lookup.loc[idx, "Breitengrad"] = lat_long[0]
        geo_lookup.loc[idx, "Laengengrad"] = lat_long[1]  
    
    # Fix incorrect coordinates
    # Frankfurt (Oder)
    geo_lookup.loc[
        (geo_lookup["Ort"]=="Frankfurt") & (geo_lookup["Postleitzahl"]=="15236"),
        ["Breitengrad", "Laengengrad"]] = (52.293446, 14.4889278)
    geo_lookup.loc[
        (geo_lookup["Ort"]=="Frankfurt") & (geo_lookup["Postleitzahl"]=="15234"),
        ["Breitengrad", "Laengengrad"]] = (52.3498817, 14.4720371)
    geo_lookup.loc[
        (geo_lookup["Ort"]=="Frankfurt") & (geo_lookup["Postleitzahl"]=="15232"),
        ["Breitengrad", "Laengengrad"]] = (52.3295384, 14.537357)
    geo_lookup.loc[
        (geo_lookup["Ort"]=="Frankfurt") & (geo_lookup["Postleitzahl"]=="15230"),
        ["Breitengrad", "Laengengrad"]] = (52.3367474, 14.5552348)

    # Friedrichsdorf (Anhalt-Bitterfeld)
    geo_lookup.loc[
        (geo_lookup["Ort"]=="Friedrichsdorf") & (geo_lookup["Postleitzahl"]=="06386"),
        ["Breitengrad", "Laengengrad"]] = (51.77197169999999, 12.073529)
    
    # Remove Insel Neuwerk (presumably not served by 50Hertz)
    geo_lookup = geo_lookup.loc[~(geo_lookup["Ort"]=="Hamburg-Insel Neuwerk")].copy()

    # Make sure all coordinates fall within region of interest
    assert all(  (geo_lookup["Breitengrad"] > 50) 
               & (geo_lookup["Breitengrad"] < 55)
               & (geo_lookup["Laengengrad"] > 9.5) 
               & (geo_lookup["Laengengrad"] < 15.1)),\
               "Coordinates outside region of interest!"

    return geo_lookup


##########

def compute_distance(row, geo_lookup):
    '''
    Compute and return distances between coordinates in MaStR and geo coordinates retrieved from Google Maps API.
    '''
    
    lat_ori, long_ori = row[["Breitengrad", "Laengengrad"]]
    
    if ~np.isnan(lat_ori):
        geo = geo_lookup.loc[
            (geo_lookup["Postleitzahl"] == row["Postleitzahl"]) &
            (geo_lookup["Ort"] == row["Ort"]) &
            (geo_lookup["Landkreis"] == row["Landkreis"])]
        if geo.shape[0] > 0:
            lat_geo = geo["Breitengrad"].values[0]
            long_geo = geo["Laengengrad"].values[0]
            distance = np.sqrt((lat_ori*LAT_LENGTH-lat_geo*LAT_LENGTH)**2 + (long_ori*LONG_LENGTH-long_geo*LONG_LENGTH)**2)
            return distance
        
    return np.nan


##########

def fix_coordinates(df, geo_lookup, thr=20):
    '''
    Fix coordinates that were presumably entered incorrectly (>thr km away from corresponding geo coordinate) and fill 
    missing coordinates with corresponding geo coordinates in lookup table.
    '''
    
    df = df.copy()
    
    ### 1. Remove coordinates that were presumably entered incorrectly
    
    # Get slice of MaStR data that has coordinates and make copy
    has_latlong = df.loc[~df["Breitengrad"].isna()].copy()

    # Compute distance to looked up coordinates
    has_latlong["distance"] = has_latlong.apply(lambda row: compute_distance(row,geo_lookup), axis=1)
    
    # Add distances to MaStR df
    df = df.merge(has_latlong[["EegMaStRNummer","distance"]], on="EegMaStRNummer", how="left", validate="1:1")

    # Remove wrong coordinates
    df.loc[df["distance"]>thr, ["Laengengrad", "Breitengrad"]] = np.nan

    # Drop distance column
    df = df.drop(columns="distance")
    
    ### 2. Fill up missing coordinates with corresponding lookup coordinates
    df = df.merge(geo_lookup, 
                  on=["Postleitzahl", "Ort", "Landkreis"], 
                  suffixes=["", "_fillnans"], 
                  how="left", 
                  validate="m:1")
    df["Laengengrad"] = df["Laengengrad"].fillna(df["Laengengrad_fillnans"])
    df["Breitengrad"] = df["Breitengrad"].fillna(df["Breitengrad_fillnans"])
    df = df.drop(columns=["Laengengrad_fillnans", "Breitengrad_fillnans"])
    
    # Downcast floats
    df["Laengengrad"] = df["Laengengrad"].astype("float32")
    df["Breitengrad"] = df["Breitengrad"].astype("float32")
    
    return df

