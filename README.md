# REpred
Forecasting of renewable energy generation in the German 50Hertz Transmission GmbH control area

Main workflow is implemented in Jupyter Notebooks (for details see below):
 - RE_preprocessing: Data retrieval and preprocessing
 - RE_model_fitting: Fit models to historical data (XGBoost)
 - RE_prediction: Make forecast

![example](https://github.com/PiaSchroeder/REpred/assets/45008571/41b536af-4c53-4eb2-8051-c31512eb3865)

Note: The electricity price is influenced by electricity supply throughout Germany as well as geopolitical events, meaning that the accuracy of price forecasts based on energy production in the 50 Hertz area alone is limited.

### Data sources
- Historical generation and price data: entso-e (European Network of Transmission System Operators for Electricity) transparency platform (https://transparency.entsoe.eu/) via the python client for the entso-e API (https://github.com/EnergieID/entsoe-py)
- Consumption data: Bundesnetzagentur (https://www.smard.de/home/downloadcenter/download-marktdaten/)
- Facility locations and capacity: Marktstammdatenregister (https://www.marktstammdatenregister.de/MaStR)
- Historical weather data and forecast: Open-Meteo API (https://open-meteo.com/)
- Geo locations: Google Maps Geocoding API (https://developers.google.com/maps/documentation/geocoding/overview)
- CO2 emission factors: Agorameter documentation (https://www.agora-energiewende.de/veroeffentlichungen/agorameter-dokumentation/)

### List of files

#### 1. Preprocessing

##### Notebook: RE_preprocessing.ipynb
- Data retrieval, preprocessing, plotting, and aggregation into final data frames
- Per energy type
- Imports from MaStR.py, entso_e.py, DataAggregation.py, and Weather.py

##### Scripts:
- **MaStR.py**
   - Get and preprocess facility data from the Marktstammdatenregister (MaStR,
     previously downloaded and saved to file)
   - imports from regional_info.py
   - Functions:
       - **get_MaStR(energy_type, save_path=None, subtype=None)**:
         Main function to extract MaStR data for energy_type. Saves and returns cleaned and filtered data frame.
       - **get_df(files, features)**:
         Load files and return aggregate data frame holding features of interest of active facilities.
       - **clean_data(df)**:
         Perform basic cleaning of data frame, including recoding of zip codes, dtype conversion, and removal of unnecessary columns.
       - **get_lat_long(zip_code, place, district, country="DE")**:
         Make API calls to Google Maps Geocoding API to get and return geo locations.
       - **lookup_latlongs(df)**:
         Get and return geo locations for all place-zipcode-district combinations in df (to be used as lookup table).
       - **compute_distance(row, geo_lookup)**:
         Compute and return distances between coordinates in MaStR and geo coordinates retrieved from Google Maps API.
       - **fix_coordinates(df, geo_lookup, thr=20)**:
         Fix coordinates that were presumably entered incorrectly (>thr km away from corresponding geo coordinate) and fill missing
         coordinates with corresponding geo coordinates in lookup table.

- **regional_info.py**
   - loads and processes info on zip codes from file.

- **entso_e.py**
   - Get electricity generation data and prices from the entso-e transparency platform
   - Functions:
       - **get_generation_data(area_code, start, end)**:
         Request, clean, and return generation data from entso-e transparency platform.
       - **aggregate_generation_data(df)**:
         Aggregate minor energy types into "other" category.
       - **get_price(area_code, start, end, ref_df)**:
         Request, clean, and return price data from entso-e transparency platform.
       - **plot_generation(gen_df, color_dict, cons_df=None, em_df=None, pr_df=None, days=7)**:
         Plot fluctuations in energy generation from all sources and optionally add consumption/emissions/price.

- **Weather.py**
   - Get historical weather data and weather forecasts from the Open-Meteo API
   - Functions:
       - **downcast_dtypes(df)**:
         Downcast dtypes to save time and space.
       - **get_weather_data(cluster_df, save_path, energy_type, start, end, timezone)**:
         Request historical and forecast weather data from open meteo for each cluster for a given time period and compute the average
         weather weighted by cluster capacities.

- **DataAggregation.py**
   - Miscellaneous functions to aggregate data from MaStR, open meteo, and entso-e
   - Functions:
       - **clean_consumption(df, ref_df)**:
         Preprocess consumption data: match datetime indices between data frames, rename columns, and fix data types.
       - **compute_emissions(df)**:
         Compute CO2 emissions from generation from fossil fuels and their respective emission factors.
       - **cluster_facilities(df, k)**:
         Use k-means clustering to cluster facility locations.
       - **plot_facilities(df, clustered=False, cluster_centers=None, color=None, size_max=None)**:
         Plot facilities and clusters on map. Marker size is defined by capacity.
       - **preprocess_plot_df(df)**:
         Preprocess facility data for plotting on map (group by clusters and coordinates and compute radius).
       - **aggregate_data(MaStR_df, weather_df, gen_df=None, plot=False)**:
         Compute cumulative capacity and aggregate capacity, weather, and generations data frames (creates final data frames for
         prediction).


#### 2. Model fitting

##### Notebook: RE_model_fitting.ipynb
- Fits XGBoost regression models using cross-validated grid search to forecast generation based on weather and seasonal data.
- imports from REForecasting.py

##### Scripts:
- **REForecasting.py**
    - Various functions implementing model fitting and forecasting
    - Functions:
        - **add_datetime_features(df)**:
          Create time series features based on datetime components.
        - **train_test_split(df, split_date, target=None, plot=True)**:
          Split data into train and test set at specified split date.
        - **run_baseline_xgboost(train, test, n_estimators=2000, learning_rate=0.01, max_depth=6, target=None)**:
          Fit basic model (pre parameter tuning) to determine feature importance.
        - **data_prep(df, split_date, fit_base=True, fi=None, target=None)**:
          Add datetime features, create train/test split, fit base model, and plot feature importance.
        - **run_cv_xgboost(df, features, param_search, target=None)**:
          Run grid search with cross validation to optimise parameters.
        - **predict_test(df, model, compute_error=True, target=None)**:
          Predict held-out test set. Returns test df with added column "prediction" and, optionally, a prediction score (RMSE).
        - **plot_predictions(true_df, pred_df, start=None, end=None, target=None)**:
          Plot true data and predictions for the entire time period and, optionally, a period defined by start and end (YYYY-MM-DD
          format).
        - **run_grid_search(df, df_train, df_test, features, params, target=None)**:
          Run grid-search, predict test set, and plot prediction against true data.
        - **make_forecast(test_df, model, true_df=None, plot_start=None, plot_end=None, target=None)**:
          Use trained model to predict generation data based on features in test_df. If true_df is passed, predicted and actual
          generation data will be plotted.


#### 3. Prediction

##### Notebook: RE_prediction.ipynb
- Forecast unseen data per production type using the fitted models
- Imports from REForecasting.py

##### Scripts:
- **REForecasting.py**
    - See above
