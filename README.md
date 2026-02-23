# ⏰ Time-Series Forecasting - ARIMA Statistical Models

A **comprehensive guide to ARIMA (AutoRegressive Integrated Moving Average)** statistical models for time-series forecasting with hands-on implementation and parameter selection techniques.

## 🎯 Overview

This project covers:
- ✅ ARIMA fundamentals
- ✅ Stationarity testing (ADF test)
- ✅ ACF/PACF analysis
- ✅ Parameter selection (p, d, q)
- ✅ Auto ARIMA
- ✅ Seasonal ARIMA (SARIMA)
- ✅ Model diagnostics

## 📊 Time-Series Components

```python
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

class TimeSeriesAnalysis:
    """Analyze time-series components"""
    
    def __init__(self, series, freq='D'):
        self.series = series
        self.freq = freq
    
    def decompose_series(self, model='additive'):
        """Decompose series into components"""
        decomposition = seasonal_decompose(
            self.series,
            model=model,
            period=365 if self.freq == 'D' else 12
        )
        
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        
        axes[0].plot(decomposition.observed, color='blue')
        axes[0].set_ylabel('Observed')
        
        axes[1].plot(decomposition.trend, color='green')
        axes[1].set_ylabel('Trend')
        
        axes[2].plot(decomposition.seasonal, color='orange')
        axes[2].set_ylabel('Seasonal')
        
        axes[3].plot(decomposition.resid, color='red')
        axes[3].set_ylabel('Residual')
        
        plt.tight_layout()
        plt.show()
        
        return decomposition
```

## 🔍 Stationarity Testing

```python
from statsmodels.tsa.stattools import adfuller, kpss

class StationarityTest:
    """Test and achieve stationarity"""
    
    @staticmethod
    def adf_test(series):
        """Augmented Dickey-Fuller Test"""
        result = adfuller(series.dropna(), autolag='AIC')
        
        print('ADF Test Results:')
        print(f'ADF Statistic: {result[0]:.6f}')
        print(f'P-value: {result[1]:.6f}')
        print(f'Critical Values:')
        for key, value in result[4].items():
            print(f'  {key}: {value:.3f}')
        
        if result[1] <= 0.05:
            print('✓ Series is STATIONARY (reject H0)')
            return True
        else:
            print('✗ Series is NON-STATIONARY (fail to reject H0)')
            return False
    
    @staticmethod
    def kpss_test(series):
        """KPSS Test"""
        result = kpss(series.dropna(), regression='c')
        
        print('KPSS Test Results:')
        print(f'KPSS Statistic: {result[0]:.6f}')
        print(f'P-value: {result[1]:.6f}')
    
    @staticmethod
    def make_stationary(series, d=1):
        """Differencing to achieve stationarity"""
        differenced = series.diff(d).dropna()
        return differenced
```

## 📈 ACF & PACF Analysis

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

class ACFPACFAnalysis:
    """Autocorrelation analysis"""
    
    @staticmethod
    def plot_acf_pacf(series, lags=40):
        """Plot ACF and PACF"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # ACF
        plot_acf(series.dropna(), lags=lags, ax=axes[0])
        axes[0].set_title('Autocorrelation Function (ACF)')
        
        # PACF
        plot_pacf(series.dropna(), lags=lags, ax=axes[1])
        axes[1].set_title('Partial Autocorrelation Function (PACF)')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def interpret_parameters(acf_plot, pacf_plot):
        """Interpret p, d, q from plots"""
        guidance = """
        Parameter Selection from ACF/PACF:
        
        AR(p): Look at PACF
        ├─ If PACF cuts off after lag p → AR(p)
        ├─ ACF tail off → AR component
        └─ Examples: 1 spike in PACF → AR(1)
        
        MA(q): Look at ACF
        ├─ If ACF cuts off after lag q → MA(q)
        ├─ PACF tail off → MA component
        └─ Examples: 1 spike in ACF → MA(1)
        
        d: Differencing order
        ├─ 0: Already stationary
        ├─ 1: First difference (most common)
        └─ 2: Second difference (rarely needed)
        """
        print(guidance)
```

## 🧠 ARIMA Model

```python
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

class ARIMAModel:
    """ARIMA implementation"""
    
    def __init__(self, order=(1, 1, 1)):
        """
        order = (p, d, q)
        p: AR terms, d: differencing, q: MA terms
        """
        self.order = order
        self.model = None
        self.fitted = None
    
    def fit(self, series):
        """Fit ARIMA model"""
        self.model = ARIMA(series, order=self.order)
        self.fitted = self.model.fit()
        
        print(self.fitted.summary())
        return self.fitted
    
    def forecast(self, steps=30):
        """Forecast future values"""
        forecast_result = self.fitted.get_forecast(steps=steps)
        forecast_df = forecast_result.summary_frame()
        
        return forecast_df
    
    def plot_diagnostics(self):
        """Plot model diagnostics"""
        self.fitted.plot_diagnostics(figsize=(12, 8))
        plt.tight_layout()
        plt.show()
```

## ⚙️ Auto ARIMA

```python
from pmdarima import auto_arima

class AutoARIMA:
    """Automated ARIMA parameter selection"""
    
    @staticmethod
    def find_best_order(series, seasonal=False, m=12):
        """Find best p, d, q"""
        auto_model = auto_arima(
            series,
            start_p=0, max_p=5,
            start_d=0, max_d=2,
            start_q=0, max_q=5,
            seasonal=seasonal,
            m=m,
            stepwise=True,
            trace=True,
            information_criterion='aic'
        )
        
        print(f"\nBest ARIMA order: {auto_model.order}")
        if seasonal:
            print(f"Best seasonal order: {auto_model.seasonal_order}")
        
        return auto_model
```

## 🌍 SARIMA (Seasonal ARIMA)

```python
class SARIMAModel:
    """Seasonal ARIMA for data with seasonality"""
    
    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
        """
        order = (p, d, q)
        seasonal_order = (P, D, Q, s) where s is seasonal period
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted = None
    
    def fit(self, series):
        """Fit SARIMA model"""
        self.model = SARIMAX(
            series,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        self.fitted = self.model.fit(disp=False)
        
        print(self.fitted.summary())
        return self.fitted
    
    def forecast_with_ci(self, steps=30):
        """Forecast with confidence intervals"""
        forecast = self.fitted.get_forecast(steps=steps)
        forecast_df = forecast.summary_frame()
        
        # Plot with confidence interval
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Actual data
        self.fitted.fittedvalues.plot(ax=ax, label='Fitted', alpha=0.7)
        forecast_df['mean'].plot(ax=ax, label='Forecast', color='red')
        
        # Confidence interval
        ax.fill_between(
            forecast_df.index,
            forecast_df['mean_ci_lower'],
            forecast_df['mean_ci_upper'],
            alpha=0.3,
            color='red'
        )
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid()
        plt.tight_layout()
        plt.show()
        
        return forecast_df
```

## 📊 Model Evaluation

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

class ARIMAEvaluator:
    """Evaluate forecast accuracy"""
    
    @staticmethod
    def calculate_metrics(actual, predicted):
        """Calculate error metrics"""
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mape = mean_absolute_percentage_error(actual, predicted)
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }
    
    @staticmethod
    def compare_models(results_dict):
        """Compare multiple models"""
        comparison_df = pd.DataFrame(results_dict).T
        print("\nModel Comparison:")
        print(comparison_df)
        
        return comparison_df
```

## 💡 Interview Talking Points

**Q: ARIMA vs Prophet vs LSTM?**
```
Answer:
- ARIMA: Statistical, interpretable, stationary requirement
- Prophet: Handles seasonality/trends, robust to missing data
- LSTM: Deep learning, non-linear patterns
```

**Q: How determine d (differencing order)?**
```
Answer:
- d=0: Already stationary (confirmed by ADF test)
- d=1: Most common (first difference)
- d=2: Rarely needed (second difference)
- Check ADF test after each differencing level
```

## 🌟 Portfolio Value

✅ ARIMA fundamentals
✅ Stationarity testing
✅ ACF/PACF analysis
✅ Auto ARIMA parameter selection
✅ Seasonal ARIMA (SARIMA)
✅ Time-series forecasting
✅ Model diagnostics

---

**Technologies**: Statsmodels, Pandas, NumPy, Scikit-learn, PMDarima

