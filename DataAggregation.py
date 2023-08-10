### Aggregate data from MaStR, open meteo, entsoe ##########################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import colorcet as cc
from sklearn.cluster import KMeans


### Constants ##############################################################################################
LAT_LENGTH = 111.
LON_LENGTH = 67.

# Taken from Agorameter documentation (https://www.agora-energiewende.de/veroeffentlichungen/agorameter-dokumentation/)
emission_factors = {
    "Fossil Brown coal/Lignite": 1.09,
    "Fossil Gas": 0.37,
    "Fossil Hard coal": 0.82,
    "Other fossils": 1.5,
}

### Functions ##############################################################################################

def clean_consumption(df, ref_df):
    '''
    Preprocess consumption data: match datetime indices between data frames, rename columns, and fix data types.
    '''
    
    df = df.copy()
    
    # Quick fix for different languages returned by smard
    df = df.rename(columns={\
        "Datum":"Date", 
        "Anfang":"Start", 
        "Ende":"End",
        "Gesamt (Netzlast) [MWh] Berechnete Auflösungen":"Gesamt (Netzlast) [MWh] Calculated resolutions",
        "Residuallast [MWh] Berechnete Auflösungen":"Residuallast [MWh] Calculated resolutions",
        "Pumpspeicher [MWh] Berechnete Auflösungen":"Pumpspeicher [MWh] Calculated resolutions",
    })
    
    # Make datetime index (for completeness, but see next step)
    df["datetime"] = df[["Date", "Start"]].agg('T'.join, axis=1)
    df["datetime"] = pd.to_datetime(df["datetime"], format="%d.%m.%YT%H:%M")
    df = df.set_index(df["datetime"])
    df = df.drop(columns=["Date", "Start", "End", "datetime"])
    
    # Smard data has switches to daylight-saving time --> use index from generation_df instead
    assert (df.shape[0]==ref_df.shape[0]) \
        and (df.index.min()==ref_df.index.min())\
        and (df.index.max()==ref_df.index.max()),\
        "Indices don't match!"
    df = df.set_index(ref_df.index)
    
    # Rename columns
    df = df.rename(columns = {
        "Gesamt (Netzlast) [MWh] Calculated resolutions": "Gesamtnetzlast", 
        "Residuallast [MWh] Calculated resolutions": "Residuallast",
        "Pumpspeicher [MWh] Calculated resolutions": "Pumpspeicher",
    })
    
    # Fix data types
    for col in df.columns:
        df[col] = df[col].str.replace(".", "", regex=False)
        df[col] = df[col].str.replace(",", ".", regex=False)
        df[col] = df[col].astype("float32")
    
    return df


##########

def compute_emissions(df):
    ''' 
    Compute CO2 emissions from generation from fossil fuels and their respective emission factors.
    '''
        
    emission_df = pd.DataFrame(index=df.index)
    
    # Total emissions in tons
    emission_df["total_emissions"] = df["Fossil Brown coal/Lignite"]*emission_factors["Fossil Brown coal/Lignite"]\
                                   + df["Fossil Gas"]*emission_factors["Fossil Gas"]\
                                   + df["Fossil Hard coal"]*emission_factors["Fossil Hard coal"]\
                                   + df["Other fossils"]*emission_factors["Other fossils"]
                
    # Hourly transmission factor of the energy mix in g/kWh
    emission_df["emission_factor"] = (emission_df["total_emissions"]/df.sum(axis=1))*1000
    
    return emission_df
    
    
##########

def cluster_facilities(df, k):
    '''
    Use k-means clustering to cluster facility locations.
    '''
    
    df = df.copy()
    lat_lon = df[["Laengengrad", "Breitengrad"]].copy()

    # Scale
    lat_lon["Laengengrad"] = lat_lon["Laengengrad"]*LON_LENGTH
    lat_lon["Breitengrad"] = lat_lon["Breitengrad"]*LAT_LENGTH
    
    # Run k-means clustering
    my_kmeans = KMeans(n_clusters=k, n_init="auto", random_state=123)
    my_kmeans.fit(lat_lon)
    cluster_labels = my_kmeans.predict(lat_lon)

    # Add cluster labels to data frame
    df["cluster"] = cluster_labels
    
    # Get centroids
    cluster_centers = pd.DataFrame(my_kmeans.cluster_centers_, columns=lat_lon.columns)

    # Rescale back to longitude and latitude
    cluster_centers["Laengengrad"] = cluster_centers["Laengengrad"]/LON_LENGTH
    cluster_centers["Breitengrad"] = cluster_centers["Breitengrad"]/LAT_LENGTH

    # Make cluster label column (redundant with index but just to make it explicit)
    cluster_centers = cluster_centers.reset_index(drop=False, names="cluster_label")

    # Add summed capacity per cluster (in MW)
    cluster_centers["total_capacity"] = df.groupby("cluster")["InstallierteLeistung"].sum()/1000
    
    return df, cluster_centers    


##########

def plot_facilities(df, clustered=False, cluster_centers=None, color=None, size_max=None):
    '''
    Plot facilities and clusters on map. Marker size is defined by capacity.
    '''

    if not color:
        color = "#0929AC"
    
    # Generate 100 unique colors from the color map
    color_map = cc.glasbey[:100]

    centre_map = {
        "lat": 52.5,
        "lon": 12.5
    }
    
    # Preprocess df for plotting
    plot_df = preprocess_plot_df(df)

    if clustered:
        fig = px.scatter_mapbox(data_frame=plot_df, 
                                lat="Breitengrad", 
                                lon="Laengengrad",
                                color="cluster",
                                color_continuous_scale=color_map,
                                size_max=3,
                                opacity=0.5, 
                                zoom=5, 
                                center=centre_map, 
                                width=400,
                                height=500,
                                mapbox_style="carto-positron")
        # Update marker size
        fig.update_traces(marker=dict(size=4))

        fig.add_trace(px.scatter_mapbox(data_frame=cluster_centers,
                                lat="Breitengrad",
                                lon="Laengengrad",
                                color_discrete_sequence=["black"],
                                opacity=0.8,
                                size="total_capacity",
                                size_max=10).data[0])
    else:
        fig = px.scatter_mapbox(data_frame=plot_df, 
                                lat="Breitengrad", 
                                lon="Laengengrad",
                                color_discrete_sequence=[color],
                                size="InstallierteLeistung",
                                size_max=size_max,
                                opacity=0.5, 
                                zoom=5, 
                                center=centre_map, 
                                width=400,
                                height=500,
                                mapbox_style="carto-positron")         
    
    fig.show()
    

##########

def preprocess_plot_df(df):
    '''
    Preprocess facility data for plotting on map (group by clusters and coordinates and compute radius).
    '''
    
    df = df.copy()
    
    if "cluster" in df.columns:
        df = (df.loc[df["InstallierteLeistung"]>0]
                        .groupby(["Breitengrad", "Laengengrad", "cluster"])
                        .agg({"InstallierteLeistung":"sum"})
                        .reset_index())
    else:
        df = (df.loc[df["InstallierteLeistung"]>0]
                        .groupby(["Breitengrad", "Laengengrad"])
                        .agg({"InstallierteLeistung":"sum"})
                        .reset_index())
                   
    # Compute radius for facilities (logarithm of capacity)
    df["log_capacity"] = np.log10(df["InstallierteLeistung"])

    # Normalize the logarithmic capacity to the desired range
    range_min = 1
    range_max = 10
    min_cap = df["log_capacity"].min()
    max_cap = df["log_capacity"].max()
    df["radius"] = range_min + (df["log_capacity"] - min_cap) / (max_cap - min_cap) * (range_max-range_min)

    df = df.drop(columns="log_capacity")
    
    return df


##########

def aggregate_data(MaStR_df, weather_df, gen_df=None, plot=False):
    '''
    Compute cumulative capacity and aggregate capacity, weather, and generation data frames (creates final data frames for prediction).
    '''
    
    # Compute cumulative capacity
    MaStR_df = MaStR_df.set_index("Inbetriebnahmedatum").sort_index()
    capacity_df = pd.DataFrame(index=weather_df.index)
    if MaStR_df.index.max() < capacity_df.index.min():
        capacity_df["cumulative_capacity"] = MaStR_df["InstallierteLeistung"].sum()/1000
    else:
        capacity_df["cumulative_capacity"] = MaStR_df["InstallierteLeistung"].resample("H").sum().cumsum()/1000
        capacity_df["cumulative_capacity"] = capacity_df["cumulative_capacity"].fillna(method="ffill")        
    
    # Add electricity generation data
    if gen_df is not None:
        
        # Rename generation column
        gen_df.columns = ["generated_electricity"]

        # Make sure the time periods match
        assert all(weather_df.index == capacity_df.index) and all(weather_df.index == gen_df.index), "Index mismatch!"

        # Merge data frames
        df = (capacity_df
            .merge(weather_df, left_index=True, right_index=True, validate="1:1")
            .merge(gen_df, left_index=True, right_index=True, validate="1:1")
         )
    else:
        # Make sure the time periods match
        assert all(weather_df.index == capacity_df.index), "Index mismatch!"

        # Merge data frames
        df = (capacity_df
            .merge(weather_df, left_index=True, right_index=True, validate="1:1")
         )
        
    
    if plot and (gen_df is not None):
        # Show pairplot
        sns.pairplot(df, y_vars=["generated_electricity"])
        plt.show()
    
    return df

