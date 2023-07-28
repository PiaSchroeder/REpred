### Get electricity generation data and prices from the entso-e transparency platform ##############################
import pandas as pd
from datetime import datetime as dt, timedelta
import matplotlib.pyplot as plt
import keys
from entsoe import EntsoePandasClient


###  Create Client from entso-e python wrapper (https://github.com/EnergieID/entsoe-py) ###########################
client = EntsoePandasClient(api_key=keys.ENTSOE_API_TOKEN)


### Functions #####################################################################################################

def get_generation_data(area_code, start, end):
    '''
    Request, clean, and return generation data from entso-e transparency platform.
    Input:
        area_code:  str - area code for regions of interest (see API documentation: 
                    https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html)
        start:      pandas Timestamp
        end:        pandas Timestamp
    '''    
    
    print("Retrieving electricity generation data from entso-e transparency platform...")
    
    gen_df = client.query_generation(area_code, start=start,end=end)
    
    # Compute net generation for hydro pumped storage
    gen_df[("Hydro Pumped Storage", "Actual Aggregated")] = gen_df[("Hydro Pumped Storage", "Actual Aggregated")] \
                                                            - gen_df[("Hydro Pumped Storage", "Actual Consumption")]
    
    # Drop and rename columns
    gen_df = gen_df.loc[:, ~gen_df.columns.get_level_values(1).str.contains('Actual Consumption')]
    gen_df.columns = [col[0] for col in gen_df.columns]
    gen_df = gen_df.drop(columns="Geothermal") # No geothermal energy in control area
    
    # Make index datetime and remove time zone info
    gen_df.index = pd.to_datetime(gen_df.index.tz_localize(None))
    
    # Downsample to hourly samples
    gen_df = gen_df.resample("H").mean()
    
    # Forward fill missing values
    gen_df = gen_df.fillna(method='ffill', axis=0)
    
    # Aggregate production types
    gen_df = aggregate_generation_data(gen_df)
    
    print("Done!")
    
    return gen_df


##########

def aggregate_generation_data(df):
    '''
    Aggregate minor energy types into "other" category.
    '''
    
    df = df.copy()
    
    # Other renewables = Biomass, Hydro Run-of-river and poundage, Other renewable
    df["Other renewables"] = df["Biomass"] + df["Hydro Run-of-river and poundage"] + df["Other renewable"]
    df = df.drop(columns=["Biomass", "Hydro Run-of-river and poundage", "Other renewable"])
    
    # Other fossils = Waste, Fossil Oil, Other
    df["Other fossils"] = df["Waste"] + df["Fossil Oil"] + df["Other"]
    df = df.drop(columns=["Waste", "Fossil Oil", "Other"])
    
    return df


##########

def get_price(area_code, start, end, ref_df):
    '''
    Request, clean, and return price data from entso-e transparency platform.
    Input:
        area_code:  str - area code for regions of interest (see API documentation: 
                    https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html)
        start:      pandas Timestamp
        end:        pandas Timestamp
    '''    
    
    print("Retrieving electricity price data from entso-e transparency platform...")
    
    prices = client.query_day_ahead_prices(area_code, start=start, end=end)
    
    # Make data frame
    price_df = pd.DataFrame(columns=["prices"], index=prices.index, data=prices)
    
    # Make index datetime and remove time zone info (but see next step)
    price_df.index = pd.to_datetime(price_df.index.tz_localize(None))
    price_df = price_df.iloc[:-1].copy()
    
    # price data considers daylight saving times. fix with generation_df
    # price data is only available from 2018-10-01
    start_date = price_df.index.min()
    ref_df = ref_df.loc[ref_df.index>=start_date].copy()
    assert (price_df.shape[0]==ref_df.shape[0]) \
        and (price_df.index.min()==ref_df.index.min())\
        and (price_df.index.max()==ref_df.index.max()),\
        "Indices don't match!"
    price_df = price_df.set_index(ref_df.index)
    
    # Forward fill missing values
    price_df = price_df.fillna(method='ffill', axis=0)
    
    print("Done!")
    
    return price_df


##########

def plot_generation(gen_df, color_dict, cons_df=None, em_df=None, pr_df=None, days=7):
    '''
    Plot fluctuations in energy generation from all sources for the last days in gen_df and optionally add consumption/emissions/price.
    '''
    
    plot_order = ["Other renewables", "Wind Offshore", "Wind Onshore", "Solar", 
                  "Fossil Brown coal/Lignite", "Fossil Hard coal", "Fossil Gas", "Other fossils", 
                  "Hydro Pumped Storage"]
    
    # Get data to plot
    cutoff = gen_df.index.max() - timedelta(days=days)
    plot_df = gen_df.loc[gen_df.index >= cutoff, plot_order]/1000
    plot_df = plot_df.applymap(lambda x: 0 if x < 0 else x) # null negative values in hydro-pumped storage

    if em_df is None:
        # Plot generation using custom color dictionary
        ax = plot_df.plot(kind='area', 
                          stacked=True, 
                          lw=0,
                          color=[color_dict.get(c, '#D420C6') for c in plot_df.columns], 
                          figsize=(10, 4),
                          xlabel="Date",
                          ylabel="Production in GW"
                          )


        # Plot total consumption
        if cons_df is not None:
            plot_cons_df = cons_df.loc[cons_df.index >= cutoff, "Gesamtnetzlast"]/1000
            plot_cons_df.plot(kind='line',
                              color='#e62e77',
                              linewidth=2,
                              ylabel="Production/consumption in GW",
                              ylim=(0,None),
                              ax=ax)
        # Move the legend outside the plot area  
        legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=3)
        legend.set_frame_on(False)

        # Format y tick labels
        ax.yaxis.set_major_formatter('{x:,.0f}')
        
        # Remove all spines
        for spine in ax.spines.values():
            spine.set_visible(False)

    else:
        
        fig, ax1 = plt.subplots(figsize=(10, 4))
        ax2 = ax1.twinx()
        ax3 = ax1.twinx()
        
        # Plot generation using custom color dictionary but faded out
        plot_df.plot(kind='area', 
                      stacked=True, 
                      lw=0,
                      alpha=0.3,
                      color=[color_dict.get(c, '#D420C6') for c in plot_df.columns], 
                      legend=False,
                      ax=ax3
                      )
        

        # Plot CO2 emissions
        if em_df is not None:
            plot_em_df = em_df.loc[em_df.index >= cutoff, "emission_factor"]
            plot_em_df.plot(kind='line',
                              color='#9016fa',
                              linewidth=2,
                              # secondary_y=True,
                              ylabel="CO2 emission factor in g/kWh",
                              ylim=(0,None),
                              ax=ax1)
        
        # Plot price
        if pr_df is not None:
            plot_pr_df = pr_df.loc[pr_df.index >= cutoff, "prices"]
            plot_pr_df.plot(kind='line',
                  color='#16fae3',
                  linewidth=2,
                  secondary_y=True,
                  ylabel="Price in EUR/MWh",
                  ylim=(0,None),
                  ax=ax2)

        # Hide the third y-axis
        ax3.spines['right'].set_visible(False)
        ax3.yaxis.set_ticks([])  # Remove ticks


    # Show the plot
    plt.show()
    
    