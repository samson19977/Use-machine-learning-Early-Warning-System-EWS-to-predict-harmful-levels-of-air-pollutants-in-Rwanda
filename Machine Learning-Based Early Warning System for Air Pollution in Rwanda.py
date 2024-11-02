#!/usr/bin/env python
# coding: utf-8

# Here’s a Python workflow to consolidate and analyze my air quality data from both folders ("Air Quality Data for Rural" and "Air Quality Data for Kigali City" Data from 2020-2024 from REMA). The script will read all CSV files, combine them, and conduct basic analysis.

# In[1]:


import pandas as pd
import glob
import os


# In[14]:


rural_path = os.path.expanduser("C:/Users/Francis Musoke/Downloads/Air Quality Data for Rural/*.csv")
rural_files = glob.glob(rural_path)
rural_data = [pd.read_csv(file).assign(Location="Rural") for file in rural_files]
print(rural_data)


# In[21]:


city_path = os.path.expanduser("C:/Users/Francis Musoke/Downloads/Air Quality Data for Kigali City/*.csv")
city_files = glob.glob(city_path)
# Read all files in the city folder and assign a location
city_data = [pd.read_csv(file).assign(Location="Kigali City") for file in city_files]
print(city_data)


# In[23]:


# Combine all files into a single DataFrame
data = pd.concat(rural_data + city_data, ignore_index=True)
print(data)


# Step 3: Clean and Preprocess the Data Ensure the Date column is in a datetime format and handle any missing values.
# Step 4: Exploratory Data Analysis (EDA)
# Perform descriptive statistics and visualizations to understand pollution levels across different locations.
# 
# Basic Descriptive Statistics

# Group by Location and Date for Time Series Analysis
# Calculate average pollutant levels by location over time.

# In[36]:


import glob
import os
import pandas as pd

# Folder paths
rural_path = os.path.expanduser("~/Downloads/Air Quality Data for Rural/*.csv")
city_path = os.path.expanduser("~/Downloads/Air Quality Data for Kigali City/*.csv")

# Read data from each file and print column names
def load_files(path, location):
    files = glob.glob(path)
    dataframes = []
    for file in files:
        try:
            df = pd.read_csv(file)
            # Clean column names by stripping whitespace
            df.columns = df.columns.str.strip()
            
            # Print columns for verification
            print(f"File: {file} Columns: {df.columns.tolist()}")
            
            # Standardize PM column name
            if 'PM25' in df.columns:
                df.rename(columns={'PM25': 'PM2.5'}, inplace=True)
                
            # Assign Location
            df = df.assign(Location=location)
            dataframes.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    return dataframes

# Load data from rural and city paths
rural_data = load_files(rural_path, "Rural")
city_data = load_files(city_path, "Kigali City")

# Concatenate all data
data = pd.concat(rural_data + city_data, ignore_index=True)

# Convert 'Date' column to datetime, coerce errors
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

# Drop rows with missing values in specific columns, if necessary
data = data.dropna(subset=['Date', 'SO2', 'CO', 'PM10', 'NO2', 'O3', 'PM2.5'])

# Check the cleaned data
print(data.info())
print(data.head())

# Descriptive statistics for pollutants
print(data[['SO2', 'CO', 'PM10', 'NO2', 'O3', 'PM2.5']].describe())


# Step 4: Exploratory Data Analysis (EDA)
# Exploratory Data Analysis will help us understand the dataset better. Here are a few tasks you can perform:
# 
# Check for Outliers and Missing Values:
# 
# Plot boxplots for each pollutant (e.g., 'SO2', 'CO', 'PM10', 'NO2', 'O3', 'PM2.5') to see if there are any outliers.
# Calculate the percentage of missing values in each column, if any, after the initial cleanup.
# Data Distribution:
# 
# Plot histograms or KDE plots for each pollutant to understand the distribution.
# Use a time-series plot to visualize trends over time, particularly for each location.
# Pollutant Correlation Analysis:
# 
# Calculate the correlation matrix for the pollutants and create a heatmap to visualize the relationships among them.

# In[38]:


import seaborn as sns
import matplotlib.pyplot as plt

# Set plotting style
sns.set(style="whitegrid")



# In[39]:


# Boxplot for outliers detection
plt.figure(figsize=(12, 6))
sns.boxplot(data=data[['SO2', 'CO', 'PM10', 'NO2', 'O3', 'PM2.5']])
plt.title("Boxplot for Pollutants")
plt.show()


# In[40]:


# Histogram for data distribution
data[['SO2', 'CO', 'PM10', 'NO2', 'O3', 'PM2.5']].hist(bins=30, figsize=(15, 10), layout=(2, 3))
plt.suptitle("Distribution of Pollutants")
plt.show()



# Feature Engineering (Optional)
# To enhance model performance, I might consider adding features like:
# 
# Day of the Week: Extract the day of the week from the 'Date' column.
# Season: Determine the season based on the date, which may influence pollutant levels.
# Lagged Features: For time series data, create lagged versions of pollutants to capture previous values' influence on current pollution levels.

# Spatial Analysis: You can visualize how pollution levels vary between the rural and urban settings.

# In[47]:


# Box plots for pollution levels by location
plt.figure(figsize=(15, 10))
for pollutant in ['SO2', 'CO', 'PM10', 'NO2', 'O3', 'PM2.5']:
    plt.subplot(3, 2, ['SO2', 'CO', 'PM10', 'NO2', 'O3', 'PM2.5'].index(pollutant) + 1)
    sns.boxplot(data=data, x='Location', y=pollutant)
    plt.title(f'{pollutant} Levels by Location')
    plt.xlabel('Location')
    plt.ylabel(f'{pollutant} Concentration')
plt.tight_layout()
plt.show()


# Correlation Analysis: Understand the relationships between different pollutants.

# In[48]:


# Correlation heatmap
correlation_matrix = data[['SO2', 'CO', 'PM10', 'NO2', 'O3', 'PM2.5']].corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Heatmap of Pollutants')
plt.show()


# In[59]:


import pandas as pd

# Paths to the CSV files
rural_file_path ="C:/Users/Francis Musoke/Downloads/Air Quality Data in Kigali From 2020-2024.csv"
city_file_path ="C:/Users/Francis Musoke/Downloads/Air Quality Data in Rural  From 2020-2024.csv"

# Load the rural and city data from CSV files
rural_data = pd.read_csv(rural_file_path)
city_data = pd.read_csv(city_file_path)


# In[61]:


print(city_data)


# In[79]:


import pandas as pd

# File paths to the Excel files
rural_file_path = "C:/Users/Francis Musoke/Downloads/Air Quality Data in Rural  From 2020-2024.xlsx"
city_file_path = "C:/Users/Francis Musoke/Downloads/Air Quality Data in Kigali From 2020-2024.xlsx"

# Function to load all sheets from an Excel file into a single DataFrame with site names
def load_excel_data_with_site_name(file_path):
    # Read all sheets in the Excel file
    all_sheets = pd.read_excel(file_path, sheet_name=None)  # sheet_name=None loads all sheets as a dictionary
    
    # List to store data from each sheet
    data_frames = []
    
    for site_name, data in all_sheets.items():
        # Add a new column for the site name
        data['Site'] = site_name
        # Append the DataFrame to the list
        data_frames.append(data)
    
    # Concatenate all DataFrames into one
    combined_data = pd.concat(data_frames, ignore_index=True)
    return combined_data

# Load rural and city data
rural_data = load_excel_data_with_site_name(rural_file_path)
city_data = load_excel_data_with_site_name(city_file_path)

# Check the combined data
print("Rural Data:\n", rural_data.head())
print("City Data:\n", city_data.head())


# In[67]:


# Adjust the validation function if needed
def validate_site_data(df, location):
    print(f"\nValidating data for {location} data:")
    
    # Convert the 'Date' column to datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Check for missing values
    missing_values = df.isnull().sum()
    print(f"Missing values:\n{missing_values}\n")
    
    # Check for duplicate dates per site
    duplicate_dates = df.duplicated(subset=['Site', 'Date']).sum()
    print(f"Duplicate dates (per site): {duplicate_dates}")
    
    # Display basic statistics
    print(f"Descriptive statistics:\n{df.describe()}\n")

# Run validation for rural and city data
validate_site_data(rural_data, "Rural")
validate_site_data(city_data, "City")


# In[80]:


import pandas as pd

# Assuming rural_data and city_data have been loaded already

def impute_missing_values(df):
    # Mean/Median Imputation
    df['CO'].fillna(df['CO'].mean(), inplace=True)  # For CO
    df['PM10'].fillna(df['PM10'].mean(), inplace=True)  # For PM10
    if 'PM2.5' in df.columns:
        df['PM2.5'].fillna(df['PM2.5'].mean(), inplace=True)  # For PM2.5

    # Forward Fill and Backward Fill
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

    # Linear Interpolation
    df.interpolate(method='linear', inplace=True)

# Impute missing values for both datasets
impute_missing_values(rural_data)
impute_missing_values(city_data)

# Check the imputation result
print(rural_data.isnull().sum())
print(city_data.isnull().sum())


# In[81]:


import pandas as pd

# Adjust the validation function if needed
def validate_site_data(df, location):
    print(f"\nValidating data for {location} data:")
    
    # Convert the 'Date' column to datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Check for missing values
    missing_values = df.isnull().sum()
    print(f"Missing values:\n{missing_values}\n")
    
    # Check for duplicate dates per site
    duplicate_dates = df.duplicated(subset=['Site', 'Date']).sum()
    print(f"Duplicate dates (per site): {duplicate_dates}")
    
    # Display basic statistics
    print(f"Descriptive statistics:\n{df.describe()}\n")

def impute_missing_values(df):
    # Mean/Median Imputation
    df['CO'].fillna(df['CO'].mean(), inplace=True)  # For CO
    df['PM10'].fillna(df['PM10'].mean(), inplace=True)  # For PM10
    if 'PM2.5' in df.columns:
        df['PM2.5'].fillna(df['PM2.5'].mean(), inplace=True)  # For PM2.5

    # Forward Fill and Backward Fill
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

    # Linear Interpolation
    df.interpolate(method='linear', inplace=True)

# Example of how to process each site
def process_site_data(site_data, location):
    validate_site_data(site_data, location)
    
    # Remove duplicates based on 'Site' and 'Date'
    site_data.drop_duplicates(subset=['Site', 'Date'], inplace=True)
    
    # Impute missing values
    impute_missing_values(site_data)

    # Check the imputation result
    print(f"Missing values after imputation for {location}:\n", site_data.isnull().sum())

    # Save the updated DataFrame (you can choose your preferred format)
    site_data.to_csv(f"{location}_data_updated.csv", index=False)  # Save as CSV

# Assuming rural_data and city_data have been loaded already

# Process each site in rural data
rural_sites = rural_data['Site'].unique()  # Get unique site names
for site in rural_sites:
    site_df = rural_data[rural_data['Site'] == site]  # Filter data for the site
    process_site_data(site_df, site)

# Process each site in city data
city_sites = city_data['Site'].unique()  # Get unique site names
for site in city_sites:
    site_df = city_data[city_data['Site'] == site]  # Filter data for the site
    process_site_data(site_df, site)


# In[93]:


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Define pollutant concentration thresholds (in µg/m³ or mg/m³)
thresholds = {
    'SO2': 200,   # µg/m³
    'CO': 10,     # mg/m³
    'PM10': 50,   # µg/m³
    'NO2': 200,   # µg/m³
    'O3': 180,    # µg/m³
    'PM2.5': 15   # µg/m³
}

# Function to plot time series for each site with threshold values in the legend
def plot_time_series(site_data, site_name):
    # Convert 'Date' to datetime and set as index
    site_data['Date'] = pd.to_datetime(site_data['Date'], errors='coerce')
    site_data.set_index('Date', inplace=True)

    # List of pollutants to plot
    pollutants = ['SO2', 'CO', 'PM10', 'NO2', 'O3', 'PM2.5']
    
    # Create subplots for each pollutant
    plt.figure(figsize=(15, 10))
    
    for i, pollutant in enumerate(pollutants):
        plt.subplot(len(pollutants), 1, i + 1)
        plt.plot(site_data.index, site_data[pollutant], label=pollutant, color='blue')
        
        # Add threshold line and show threshold value in legend
        threshold_value = thresholds.get(pollutant, None)
        if threshold_value is not None:
            plt.axhline(y=threshold_value, color='red', linestyle='--', 
                        label=f'Threshold ({threshold_value} µg/m³)' if pollutant != 'CO' else f'Threshold ({threshold_value} mg/m³)')
        
        plt.title(f'{pollutant} Levels at {site_name}', fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel(f'{pollutant} Conc', fontsize=12)  # Shortened to 'Conc'
        plt.legend()
        plt.grid()

        # Format the x-axis for better readability
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))  # Set major ticks to every month
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Format the date to YYYY-MM
        plt.gcf().autofmt_xdate()  # Rotate date labels for better visibility
    
    plt.tight_layout()
    plt.savefig(f'{site_name}_time_series_plot.png')
    plt.show()

# Process each site in rural data
rural_sites = rural_data['Site'].unique()
for site in rural_sites:
    site_df = rural_data[rural_data['Site'] == site]
    plot_time_series(site_df, site)

# Process each site in city data
city_sites = city_data['Site'].unique()
for site in city_sites:
    site_df = city_data[city_data['Site'] == site]
    plot_time_series(site_df, site)


# In[94]:


import pandas as pd
from statsmodels.tsa.stattools import adfuller

# Function to perform ADF test and print results
def adf_test(series, pollutant_name, site_name):
    result = adfuller(series.dropna())  # Drop NaN values for test
    print(f"\nADF Test for {pollutant_name} at {site_name}:")
    print(f"Test Statistic: {result[0]}")
    print(f"P-value: {result[1]}")
    print(f"Critical Values: {result[4]}")
    if result[1] < 0.05:
        print("Result: Stationary (reject null hypothesis)")
    else:
        print("Result: Non-stationary (fail to reject null hypothesis)")

# Loop through each site in the dataset
for site in rural_sites:
    site_data = rural_data[rural_data['Site'] == site]
    pollutants = ['SO2', 'CO', 'PM10', 'NO2', 'O3', 'PM2.5']
    
    # Apply ADF test for each pollutant in the site data
    for pollutant in pollutants:
        if pollutant in site_data.columns:
            adf_test(site_data[pollutant], pollutant, site)

# Repeat the process for city data if required
for site in city_sites:
    site_data = city_data[city_data['Site'] == site]
    for pollutant in pollutants:
        if pollutant in site_data.columns:
            adf_test(site_data[pollutant], pollutant, site)


# The result "Stationary (reject null hypothesis)" for all sites means that the Augmented Dickey-Fuller (ADF) test found each pollutant time series to be stationary. This is an important characteristic in time series analysis and has several implications:
# 
# Statistical Properties are Consistent: A stationary series has consistent statistical properties over time, meaning the mean, variance, and autocorrelation (relationship over time) stay relatively constant. This makes the series more predictable and stable for analysis.
# 
# Modeling Simplification: Since the series is stationary, you won’t need to apply additional transformations like differencing to achieve stationarity before using certain statistical models, like ARIMA, which require stationary data for accurate forecasting.
# 
# Interpretation of ADF Test Result: The ADF test checks if a unit root is present (indicating non-stationarity). By rejecting the null hypothesis, the test suggests there’s no unit root, implying that the series is stationary and doesn’t exhibit strong trends or seasonality patterns that change over time.

# Create Heatmaps for Pollution Levels Across Sites
# This approach uses a heatmap to illustrate average pollution levels per site, allowing for quick identification of regional variations.

# In[ ]:


import numpy as np

# Pivot data to create a format suitable for heatmap visualization
def plot_pollutant_heatmap(data):
    # Pivot the data to get the average concentration of each pollutant per site
    pollution_pivot = data.pivot_table(index='Site', columns='Pollutant', values='Value', aggfunc=np.mean)
    
    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(pollution_pivot, cmap="YlOrRd", annot=True, fmt=".1f", cbar_kws={'label': 'Average Conc (µg/m³)'})
    plt.title("Average Pollution Levels Across Sites")
    plt.xlabel("Pollutant")
    plt.ylabel("Site")
    plt.show()

# Example of calling the function
plot_pollutant_heatmap(all_sites_data)


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

# Calculate correlation matrix for pollutants across all sites
pollutant_data = all_sites_data.pivot_table(index='Date', columns=['Pollutant', 'Site'], values='Value')
correlation_matrix = pollutant_data.corr(method='pearson')  # Use 'spearman' for Spearman correlation if non-linear

# Plot heatmap for correlation matrix
plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
plt.title("Pollutant Correlation Heatmap")
plt.show()


# Time Series Models
# For each pollutant time series, we’ll start with Seasonal ARIMA, Prophet, and LSTM models, as they are suited for handling seasonality and time dependencies. Let’s proceed with time series decomposition first to check for seasonality.

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Decompose the time series for one example pollutant at a specific site
def decompose_series(site_data, pollutant):
    series = site_data[pollutant].dropna()
    decomposition = seasonal_decompose(series, model='additive', period=365)
    
    plt.figure(figsize=(12, 8))
    decomposition.plot()
    plt.suptitle(f"{pollutant} Seasonal Decomposition at Site", fontsize=16)
    plt.show()

# Example call for one pollutant at one site
decompose_series(site_data, 'PM10')


# Seasonal ARIMA Model
# This model will capture seasonality if identified during decomposition.

# In[ ]:


from statsmodels.tsa.statespace.sarimax import SARIMAX

# Fit SARIMA model
def fit_sarima(series):
    model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    sarima_fit = model.fit(disp=False)
    return sarima_fit

# Forecasting with SARIMA
def sarima_forecast(series):
    sarima_fit = fit_sarima(series)
    forecast = sarima_fit.get_forecast(steps=30)  # Forecast for the next 30 days
    forecast_ci = forecast.conf_int()

    plt.figure(figsize=(10, 5))
    plt.plot(series, label='Observed')
    plt.plot(forecast.predicted_mean, label='Forecast')
    plt.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='pink', alpha=0.3)
    plt.title("SARIMA Forecast")
    plt.legend()
    plt.show()

# Example usage
sarima_forecast(site_data['PM10'].dropna())


# Prophet Model
# Prophet is effective for data with trends and seasonal variations. Make sure fbprophet is installed.

# In[ ]:


from fbprophet import Prophet

# Fit and forecast with Prophet
def prophet_forecast(site_data, pollutant):
    df = site_data[['Date', pollutant]].rename(columns={'Date': 'ds', pollutant: 'y'}).dropna()
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    
    model.plot(forecast)
    plt.title(f"{pollutant} Forecast using Prophet")
    plt.show()

# Example call
prophet_forecast(site_data, 'PM10')


# LSTM Model
# LSTM is suitable for handling complex, long-term dependencies in pollutant time series data.

# In[ ]:


from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def lstm_forecast(series):
    # Scale data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(series.values.reshape(-1, 1))

    # Prepare data for LSTM
    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Define LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=5, batch_size=32)

    # Forecast
    predicted_stock_price = model.predict(X)
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
    
    plt.plot(series.values, label='True Value')
    plt.plot(np.arange(60, len(series)), predicted_stock_price, label='LSTM Prediction')
    plt.legend()
    plt.title("LSTM Model Forecast")
    plt.show()

# Example call
lstm_forecast(site_data['PM10'].dropna())


#  Machine Learning Models
# For machine learning models, we’ll train Random Forest, Gradient Boosting, and SVM for pollutant prediction

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Example Random Forest training
def train_random_forest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)
    
    predictions = rf.predict(X_test)
    print("Random Forest Results:")
    print("MAE:", mean_absolute_error(y_test, predictions))
    print("RMSE:", mean_squared_error(y_test, predictions, squared=False))
    print("R²:", r2_score(y_test, predictions))

# Example usage with pollutant and meteorological data
train_random_forest(X, y)


# Cross-Validation and Hyperparameter Tuning
# Cross-validation ensures model generalization. Hyperparameter tuning helps optimize model performance.

# In[ ]:


from sklearn.model_selection import cross_val_score

# Perform K-Fold Cross-Validation
def cross_validate_model(model, X, y):
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
    print("Cross-Validation MAE Scores:", -scores)
    print("Average MAE:", -scores.mean())


# Hyperparameter Tuning (Grid Search)

# In[ ]:


from sklearn.model_selection import GridSearchCV

# Example Grid Search for Random Forest
def tune_random_forest(X, y):
    param_grid = {'n_estimators': [50, 100, 150], 'max_depth': [10, 20, 30]}
    rf = RandomForestRegressor()
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_absolute_error')
    grid_search.fit(X, y)
    print("Best Parameters:", grid_search.best_params_)
    print("Best MAE:", -grid_search.best_score_)

# Example usage
tune_random_forest(X, y)


# Validation and Evaluation
#  Evaluate Model Performance Metrics
# For each model, calculate evaluation metrics like MAE, RMSE, and 𝑅^2
#  for regression tasks.

# In[ ]:


# Model evaluation function
def evaluate_model(y_true, y_pred):
    print("MAE:", mean_absolute_error(y_true, y_pred))
    print("RMSE:", mean_squared_error(y_true, y_pred, squared=False))
    print("R²:", r2_score(y_true, y_pred))

# Example usage
evaluate_model(y_test, predictions)


# In[ ]:





# In[ ]:




