### Auxiliaries for time series forecasting #############################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt, timedelta
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

### Functions ###########################################################################################################

def add_datetime_features(df):
    """
    Create time series features based on datetime components.
    """
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
    
    if target is None:
        target = "generated_electricity"
        
    train = df.loc[df.index < split_date].copy()
    test = df.loc[df.index >= split_date].copy()

    if plot:
        fig, ax = plt.subplots(figsize=(9,3))
        train[target].plot(ax=ax, title="Train/Test Split")
        test[target].plot(ax=ax)
        ax.axvline(pd.Period(split_date, freq='H'), color="black", ls="--")
        ax.legend(["Training set", "Test set"])
        
    return train, test


##########

def run_baseline_xgboost(train, test, n_estimators=2000, learning_rate=0.01, max_depth=6, target=None):
    
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
            verbose=500)
    
    # Predict test data
    test_pred, rmse, mape = predict_test(test, reg, target=target)
    
    # Feature importance
    feature_importance = pd.DataFrame(data=reg.feature_importances_,
                                      index=reg.feature_names_in_,
                                      columns=["importance"])
    feature_importance.sort_values("importance").plot(kind="barh", title="Feature Importance")
    
    return reg, feature_importance
        

##########

def run_cv_xgboost(df, features, param_search, target=None):
    
    if target is None:
        target = "generated_electricity"

    # Use 90 days as test split size. With 5 cv folds and 1 eval set this adds up to ~2 years 
    # split_size = 120 
    split_size = 30
    split_date = df.index.max() - timedelta(days=split_size)
    cv_train = df.loc[df.index < split_date].copy()
    cv_val = df.loc[df.index >= split_date].copy()

    X_train, y_train = cv_train[features], cv_train[target]
    X_val, y_val = cv_val[features], cv_val[target]

    reg = xgb.XGBRegressor(early_stopping_rounds=50, random_state=123)
    
    tscv = TimeSeriesSplit(n_splits=3, test_size=24*split_size)
    gsearch = GridSearchCV(estimator=reg, 
                           cv=tscv,
                           param_grid=param_search,
                           scoring="neg_root_mean_squared_error",
                           verbose=100)

    print("Running cross validation...")
    gsearch.fit(X_train, y_train, eval_set=[(X_train, y_train),(X_val, y_val)],
                verbose=500)
    print("Done!")

    print("Fitting winning model to entire training set")
    winning_model = gsearch.best_estimator_
    winning_model.fit(X_train, y_train, eval_set=[(X_train, y_train),(X_val, y_val)],
                verbose=100)
    print("Done!")

    return winning_model


##########

def predict_test(df, model, compute_error=False, target=None):
    
    if target is None:
        target = "generated_electricity"
    
    df = df.copy()
    
    X_test = df[model.feature_names_in_]

    df["prediction"] = model.predict(X_test)
    
    if compute_error:
        rsme = np.sqrt(mean_squared_error(df[target], df["prediction"]))
        mape = mean_absolute_percentage_error(df[target], df["prediction"])
        print(f"RSME = {rsme:0.2f}")
        print(f"MAPE = {mape:0.2f}")
    else:
        rsme = None
        mape = None
    
    return df, rsme, mape


##########

def plot_predictions(df, test_df, start, end, target=None):
    
    if target is None:
        target = "generated_electricity"
        
    # All data
    ax = df[target].plot(figsize=(15,5))
    test_df["prediction"].plot(ax=ax)
    ax.set_title("Raw data and predictions")
    plt.legend(["True data", "Prediction"])
    plt.show()
    
    # Time period specified by start and end (must be valid dates)
    ax = df.loc[(df.index > start) & (df.index < end), [target]].plot(figsize=(15,5))
    test_df.loc[(test_df.index > start) & (test_df.index < end), ["prediction"]].plot(ax=ax)
    ax.set_title(f"{start} - {end}")
    plt.legend(["True data", "Prediction"])
    plt.show()
    
def plot_predictions_2(true_df, pred_df, start, end, linestyle='-'):
    
    # Time period specified by start and end (must be valid dates)
    ax = true_df.loc[(true_df.index >= start) & (true_df.index < end)].plot(figsize=(15,5), color="#093d91", lw=2)
    pred_df.loc[(pred_df.index >= start) & (pred_df.index < end)].plot(ax=ax, color="#fcb03d", lw=2, style=linestyle)
    # ax.set_title(f"{start} - {end}")
    plt.legend(["True data", "Prediction"])
    plt.legend(bbox_to_anchor=(1.02, 0.5), loc='center left')
    
    # Remove all spines
    # for spine in ax.spines.values():
    #     spine.set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
        
    plt.show()

