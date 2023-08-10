### Auxiliaries for time series forecasting #############################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt, timedelta
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error

### Functions ###########################################################################################################

def add_datetime_features(df):
    '''
    Create time series features based on datetime components.
    '''
    
    df = df.copy()
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["week"] = df.index.isocalendar().week.astype("int64")
    df["month"] = df.index.month
    df["year"] = df.index.year
    df["day_of_year"] = df.index.dayofyear

    return df


#########

def train_test_split(df, split_date, target=None, plot=True):
    '''
    Split data into train and test set at specified split date.
    '''
    
    if target is None:
        target = "generated_electricity"
        
    train = df.loc[df.index < split_date].copy()
    test = df.loc[df.index >= split_date].copy()

    if plot:
        fig, ax = plt.subplots(figsize=(8,2))
        train[target].plot(ax=ax, title="Train/Test Split", color="#093d91")
        test[target].plot(ax=ax, color="#fcb03d")
        ax.axvline(pd.Period(split_date, freq='H'), color="black", ls="--")
        plt.legend(["Training set", "Test set"], bbox_to_anchor=(1.02, 0.5), loc='center left')
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    return train, test


##########

def run_baseline_xgboost(train, test, n_estimators=2000, learning_rate=0.01, max_depth=6, target=None):
    '''
    Fit basic model (pre parameter tuning) to determine feature importance.
    '''
    
    if target is None:
        target = "generated_electricity"
    
    # Split features and target
    TARGET = target
    FEATURES = list(train.columns)
    FEATURES.remove(TARGET)

    X_train = train[FEATURES]
    y_train = train[TARGET]
    X_test = test[FEATURES]
    y_test = test[TARGET]
    
    # Initialize XGBoost
    reg = xgb.XGBRegressor(n_estimators=n_estimators, 
                           learning_rate=learning_rate,
                           max_depth=max_depth,
                           early_stopping_rounds=50, 
                           random_state=123)
    
    # Fit model
    reg.fit(X_train, 
            y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=0)
    
    # Feature importance
    feature_importance = pd.DataFrame(data=reg.feature_importances_,
                                      index=reg.feature_names_in_,
                                      columns=["importance"])
    
    return reg, feature_importance
        

##########

def data_prep(df, split_date, fit_base=True, fi=None, target=None):
    '''
    Add datetime features, create train/test split, fit base model, and plot feature importance.
    '''
    
    if target is None:
        target = "generated_electricity"
    
    # Add datetime features
    print("Adding datetime features... ", end="")
    df = add_datetime_features(df)
    print("Done!")
    
    # Create train/test split
    print("Creating train/test split... ", end="")
    df_train, df_test = train_test_split(df, split_date, target=target)
    print("Done!")
    
    # Run basic model on all features to determine feature importance
    if fit_base:
        print("Fitting base XGB... ", end="")
        base_xgb, fi = run_baseline_xgboost(df_train, df_test, target=target)
        print("Done!")
    else:
        base_xgb = None
    
    # Plot feature importance
    if fi is not None:
        ax = fi.sort_values("importance").plot(kind="barh", title="Feature Importance", color="#093d91")
        ax.legend().set_visible(False)
        
    return df_train, df_test, base_xgb, fi


##########

def run_cv_xgboost(df, features, param_search, target=None):
    '''
    Run grid search with cross validation to optimise parameters.
    '''
    
    if target is None:
        target = "generated_electricity"

    split_size = 30 
    n_splits = 3    
    
    # Create training and validation set
    split_date = df.index.max() - timedelta(days=split_size)
    cv_train = df.loc[df.index < split_date].copy()
    cv_val = df.loc[df.index >= split_date].copy()

    X_train, y_train = cv_train[features], cv_train[target]
    X_val, y_val = cv_val[features], cv_val[target]

    reg = xgb.XGBRegressor(early_stopping_rounds=50, random_state=123)
    
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=24*split_size)
    gsearch = GridSearchCV(estimator=reg, 
                           cv=tscv,
                           param_grid=param_search,
                           scoring="neg_root_mean_squared_error",
                           verbose=3)

    # Run grid search
    print("Running cross validated grid search...")
    gsearch.fit(X_train, 
                y_train, 
                eval_set=[(X_train, y_train),(X_val, y_val)],
                verbose=0)
    print("Done!")
    
    # Display winning model parameters
    winning_model = gsearch.best_estimator_
    winning_params = winning_model.get_params()
    print("Winning model parameters:")
    for param in param_search.keys():
        print(f"{param}: {winning_params[param]}")
    
    # Refit best model on entire training set (without CV splits)
    print("Fitting winning model to entire training set...", end="")
    winning_model.fit(X_train, 
                      y_train, 
                      eval_set=[(X_train, y_train),(X_val, y_val)],
                      verbose=0)
    print("Done!")

    return winning_model


##########

def predict_test(df, model, compute_error=True, target=None):
    '''
    Predict held-out test set. Returns test df with added column "prediction" and, optionally, a prediction score (RMSE).
    '''
    
    if target is None:
        target = "generated_electricity"
    
    df = df.copy()
    
    X_test = df[model.feature_names_in_]

    df["prediction"] = model.predict(X_test)
    
    if compute_error:
        rsme = np.sqrt(mean_squared_error(df[target], df["prediction"]))
        print(f"RSME = {rsme:0.2f}")
    else:
        rsme = None
    
    return df, rsme


##########

def plot_predictions(true_df, pred_df, start=None, end=None, target=None):
    '''
    Plot true data and predictions for the entire time period and, optionally, a period defined by start and end (YYYY-MM-DD format).
    '''

    if target is None:
        target = "generated_electricity"
        
    # All data
    ax = true_df[target].plot(figsize=(8,2), color="#093d91")
    pred_df["prediction"].plot(ax=ax, color="#fcb03d")
    plt.legend(["True data", "Prediction"], bbox_to_anchor=(1.02, 0.5), loc='center left')
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()
    
    # Time period specified by start and end (must be valid dates)
    if (start is not None) and (end is not None):
        ax = true_df.loc[(true_df.index >= start) & (true_df.index < end), [target]].plot(figsize=(8,2), color="#093d91", lw=2)
        pred_df.loc[(pred_df.index >= start) & (pred_df.index < end), ["prediction"]].plot(ax=ax, color="#fcb03d", lw=2)
        plt.legend(["True data", "Prediction"], bbox_to_anchor=(1.02, 0.5), loc='center left')
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.show()
    

##########

def run_grid_search(df, df_train, df_test, features, params, target=None):
    '''
    Run grid-search, predict test set, and plot prediction against true data.
    '''
    
    if target is None:
        target = "generated_electricity"
    
    # Run cross-validated grid search
    best_model = run_cv_xgboost(df_train, features, params, target=target)
    
    # Predict held-out test data with best model from grid search
    print("Predicting held-out test set... ", end="")
    df_test, rmse = predict_test(df_test, best_model, target=target)
    print("Done!")
    
    # Plot prediction (Entire period + last 30 days)
    plot_start = (df_test.index[-1]-timedelta(days=30)).strftime("%Y-%m-%d")
    plot_end = df_test.index[-1].strftime("%Y-%m-%d")
    plot_predictions(df, df_test, start=plot_start, end=plot_end, target=target)
    
    return best_model, df_test, rmse    


##########

def make_forecast(test_df, model, true_df=None, plot_start=None, plot_end=None, target=None):
    '''
    Use trained model to predict generation data based on features in test_df. If true_df is passed, predicted and actual generation data will 
    be plotted.
    '''
    
    if target is None:
        target = "generated_electricity"
        
    test_df = test_df.copy()
    
    # Add/remove features
    test_df = add_datetime_features(test_df)
    test_df = test_df[list(model.feature_names_in_)].copy()
    
    # Make prediction
    test_df, rmse = predict_test(test_df, model, compute_error=False)
    
    # Plot true vs. predicted data
    if true_df is not None:
        plot_predictions(true_df, test_df, start=plot_start, end=plot_end, target=target)
    
    return test_df


