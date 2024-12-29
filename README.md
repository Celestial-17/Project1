# Project

## Overview

The codebase is structured to implement the core methodologies outlined in the project:

1. **Ensemble Model Stacking**, which combines predictions from multiple Numerical Weather Prediction (NWP) models to improve wind power forecasting accuracy.
2. **Adaptive Online Post-Processing**, designed to mitigate distribution shifts resulting from the growing presence of solar capacity in the online test dataset.
3. **Quantile-Based Probabilistic Aggregation**, aimed at generating precise probabilistic forecasts for total hybrid energy production.
4. **Stochastic Trading Optimization**, which focuses on maximizing expected trading profits by accounting for uncertainties in electricity price fluctuations.
5. **Value-Oriented Price Spread Forecasting** to further enhance the trading revenue.


## Usage

### Data Preparation
Download the dataset and place it in the `./data/raw` directory.

### Data Preprocessing

To preprocess the dataset, run the following command:

```
python dataPreProcess.py
```

To generate dataset for case1 and case2, run the following command:

```
python generateDataset.py
python generateLatestDataset.py
```

### Model Training

Train LightGBM models for dense quantile regression:

```
python train.py
```

Train the stacked multi-source NWPs sister forecasting model for wind power forecasting:

```
python stacking_wind.py
```

### Hyperparameter Tuning

run the following command:

```
python params_search.py
```

### Case Study

Validate the effectiveness of the stacked multi-source NWPs sister forecasting model:

```
python test_wind_ensemble_history.py
python test_wind_ensemble_latest.py
```

Validate the effectiveness of the solar online post-processing model:

```
python test_solar_online.py
```

Validate the effectiveness of the probabilistic aggregation technique:

```
python test_aggregation_history.py
python test_aggregation_latest.py
```

Validate the effectiveness of the stochastic trading strategy:

```
python test_trading_history.py
python test_trading_latest.py
```

Validate the effectiveness of the value-oriented price spread forecasting:

```
python vof_pre_train_history.py
python vof_history.py
python vof_pre_train_latest.py
python vof_latest.py
```

### Others

The following files are used to plot the figures involved in the project:

```
python corelation.py
python plot_decision_loss.py
python prices_anal.py
python solar_comp.py
```
