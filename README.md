# Dissertation--Eddy

## Overview

The codebase is structured to reproduce the key methods described in the project:

1. **Stacking Models** trained on various Numerical Weather Predictions (NWPs) for wind power forecasting.
2. **Online Post-Processing** model to address distribution shifts caused by increased solar capacity in the online test set.
3. **Probabilistic Aggregation** technique to provide accurate quantile forecasts of total hybrid generation.
4. **Stochastic Trading Strategy** to maximize expected trading revenue considering uncertainties in electricity prices.
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

## Acknowledgements
We thank Professor Chris Dent and Gomez Anaya, Sergio for their helpful suggestions on the research project.
