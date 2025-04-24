# Urban Development Time Series Analysis and Forecasting

## Description
This repository contains Jupyter Notebooks and Python scripts for analyzing urban built-up areas and forecasting time series data using machine learning, statistical models, and geospatial data processing. The scripts are designed for urban planning and environmental analysis, focusing on data retrieval, cleaning, clustering, and forecasting. The included scripts are:

- **`LSTM.ipynb`**: Implements a Long Short-Term Memory (LSTM) neural network for univariate time series forecasting, processing Excel data, evaluating model performance, and generating future predictions.
- **`Data_import.ipynb`**: Retrieves built-up area data from Google Earth Engine’s Dynamic World V1 dataset for specified geographic regions and time periods, saving results as CSV files.
- **`Data_clean_save.ipynb`**: Cleans time series data by removing outliers based on quantiles, aggregates data by mean, and consolidates data from multiple districts into Excel files for further analysis.
- **`ARIMA.ipynb`**: Implements an ARIMA model for univariate time series forecasting, including stationarity testing, parameter optimization, and future predictions.
- **`clustered_optimizer.ipynb`**: Implements a clustering-based forecasting pipeline using KMeans and exponential smoothing with damped trend to group and forecast time series for urban zones, optimizing parameters and computing error metrics.
- **`clustered_optimizer_50_kms_included.py`**: A Python script version of the clustering-based forecasting pipeline, combining data from 50 km and non-50 km datasets, performing KMeans clustering, and forecasting with exponential smoothing optimized for MAPE.
- **`clustered_optimizer_50_mse_optimized.py`**: A Python script for clustering-based time series forecasting using KMeans and exponential smoothing with damped trend, optimized for MSE (Mean Squared Error) with grid search and parallel processing.
- **`clustered_optimizer_50_mae_optimized.py`**: A Python script for clustering-based time series forecasting using KMeans and exponential smoothing with damped trend, optimized for MAE (Mean Absolute Error) with grid search and parallel processing.
- **`nonclustered_optimizer_50_mse_optimized.py`**: A Python script for non-clustered time series forecasting using exponential smoothing with damped trend, optimized for MSE with grid search and parallel processing.
- **`nonclustered_optimizer_50_mse_optimized.py`**: An alternative Python script for non-clustered time series forecasting using exponential smoothing with damped trend, optimized for MSE with grid search and parallel processing, with specific data processing for certain Excel files.
- **`nonclustered_optimizer_50_mae_optimized.py`**: A Python script for non-clustered time series forecasting using exponential smoothing with damped trend, optimized for MAE with grid search and parallel processing, with specific data processing for certain Excel files.
- **`nonclustered_optimizer.py`**: A Python script for non-clustered time series forecasting using exponential smoothing with damped trend, optimized for MAPE (Mean Absolute Percentage Error) with grid search and parallel processing.
- **`lm_trial.py`**: A Python script for forecasting urban zone time series using linear regression models with KMeans clustering, leveraging multiple features derived from km² and percentage data, and evaluating errors across different cluster configurations.

## Installation
To run the scripts, set up a Python environment with the required dependencies:

```bash
# Clone the repository
git clone <repository-url>

# Navigate to the repository directory
cd <repository-name>

# Install dependencies
pip install -r requirements.txt
```

The `requirements.txt` should include:
```
numpy
pandas
scikit-learn
tensorflow
matplotlib
seaborn
earthengine-api
statsmodels
scipy
```

Alternatively, install the packages manually:
```bash
pip install numpy pandas scikit-learn tensorflow matplotlib seaborn earthengine-api statsmodels scipy
```

For `Data_import.ipynb`, you also need a Google Earth Engine account and authentication:
1. Sign up at [Google Earth Engine](https://earthengine.google.com/).
2. Run `ee.Authenticate()` in a Python environment or use `earthengine authenticate` in your command line.
3. Ensure your Google Cloud project is set up (e.g., `nth-infusion-437410-b6` as used in the script).

For `clustered_optimizer (1).ipynb`, `clustered_optimizer_50_kms_included.py`, `clustered_optimizer_50_mse_optimized.py`, `clustered_optimizer_50_mae_optimized.py`, `nonclustered_optimizer_50_mse_optimized.py`, `nonclustered_optimizer_50_mse_optimized (1).py`, `nonclustered_optimizer_50_mae_optimized.py`, `nonclustered_optimizer.py`, and `lm_trial.py`, Google Colab is optional for Google Drive integration but not required if running locally.

## Usage
1. Place input files (e.g., Excel files, CSV files, or polygon coordinate files) in the project directory or update file paths in the scripts.
2. For notebooks, open in Jupyter Notebook or JupyterLab:
   ```bash
   jupyter notebook <notebook-name>.ipynb
   ```
3. For Python scripts, run from the command line:
   ```bash
   python <script-name>.py
   ```
4. Run the cells or script sequentially to process data, perform analysis, and generate outputs.
5. Check each script’s section below for specific inputs, outputs, and instructions.

## Script Details

### LSTM.ipynb
**Purpose**: Performs time series forecasting using an LSTM neural network.

- **Functionality**:
  - **Data Preprocessing**: Reads time series data from Excel files (e.g., `all_meandata-50.xlsx`), normalizes it using `MinMaxScaler`, and creates sequences for LSTM input (default lookback: 3 time steps).
  - **LSTM Model**: Builds and trains a univariate LSTM model (50 units, dense output layer) on historical data (first 14 time steps), evaluates performance on test data, and calculates metrics (MAE, MSE, MAPE).
  - **Future Predictions**: Generates future predictions (default: 20 time steps) and supports visualization of historical and forecasted data (commented out).
  - **Batch Processing**: Processes multiple time series (columns 1400–1531 in input Excel) and saves error metrics to Excel files (e.g., `all_lstm_errors_data_1400-1531.xlsx`). Combines multiple error files into a single file (`lstm_errors.xlsx`).

- **Inputs**:
  - Excel files with time series data (e.g., `all_meandata-50.xlsx`, `torbali_veriler.xls`).
  - Configurable parameters: sequence length (3), future predictions (20), LSTM units (50), epochs (100).

- **Outputs**:
  - Excel files with error metrics (MAE, MSE, MAPE).
  - Optional plots of historical and forecasted data (if uncommented).
  - Trained model predictions and actual values.

- **Usage Notes**:
  - Update file paths and column/row selections in `data_capturer` to match your Excel file structure.
  - Uncomment plotting code in `plot_with_future_predictions` to visualize results.

### Data_import.ipynb
**Purpose**: Retrieves built-up area data from Google Earth Engine’s Dynamic World V1 dataset for urban analysis.

- **Functionality**:
  - **Google Earth Engine Integration**: Authenticates and initializes Earth Engine to access the `GOOGLE/DYNAMICWORLD/V1` ImageCollection.
  - **Data Retrieval**: Processes polygon coordinates to calculate built-up areas (in km² and percentage of total area) for specified regions, using semi-annual periods from 2015 to 2025.
  - **Batch Processing**: Analyzes multiple zones within a district (e.g., Torbalı) by reading polygon coordinate files from a directory, processing each zone, and saving results as CSV files.
  - **Data Processing**: Uses `analyze_built_area` to compute built-up area metrics and `ilce_bazli_arama` to iterate over polygon files for a specified district.

- **Inputs**:
  - Directory containing polygon coordinate files (e.g., `/content/drive/MyDrive/torbali/izmir_poligonlar`).
  - District name (e.g., `torbali`) to process specific zones.
  - Google Earth Engine authentication and project ID (e.g., `nth-infusion-437410-b6`).

- **Outputs**:
  - CSV files for each zone (e.g., `<zone-name>_rawdata.csv`) containing built-up area metrics (year, period, built area in km², built percentage, total area).
  - Console output with total area and image counts per period.

- **Usage Notes**:
  - Ensure polygon files are in the correct format (text files with coordinates in `lat,lon` pairs).
  - Update `polygons_path` to point to your directory containing polygon files.
  - Requires a stable internet connection for Earth Engine API calls.

### Data_clean_save.ipynb
**Purpose**: Cleans and aggregates time series data from CSV files, saving consolidated data to Excel files for forecasting.

- **Functionality**:
  - **Data Cleaning**: Uses `data_cleaner` to process CSV files, removing outliers based on quantiles (default: lower quantile = 2, upper quantile = 4). For each time period, it filters `built_percentage` values and computes the mean (configurable aggregation method).
  - **Temporal Consistency**: Ensures data consistency by replacing values in a period with the previous period’s mean if the current period’s maximum is less than or equal to the previous period’s mean.
  - **Batch Processing**: Uses `ilce_to_excel` to process all CSV files in a district’s directory (e.g., `izmir_poligonlar/torbali`), aggregating data into a single Excel file (e.g., `torbali_meandata-50.xlsx`).
  - **Data Consolidation**: Uses `concat_df` to merge data from multiple districts into a single Excel file (`all_kms.xlsx`), splitting data into training (first 14 periods) and testing sets.
  - **Clustering Support**: Imports KMeans and silhouette metrics from `sklearn`, suggesting potential clustering analysis (though not implemented in the provided code).

- **Inputs**:
  - Directory containing CSV files for each district (e.g., `/content/drive/MyDrive/izmir_poligonlar/torbali`).
  - District names (automatically detected from directory structure).
  - CSV files with columns including `date` and `built_percentage`.

- **Outputs**:
  - Excel files for each district (e.g., `<district>_meandata-50.xlsx`) containing cleaned and aggregated time series data.
  - Consolidated Excel file (`all_kms.xlsx`) with data from all districts.
  - Potential clustering outputs (if clustering code is implemented).

- **Usage Notes**:
  - Update `izmir_poligon_path` to point to your directory containing district subdirectories.
  - Ensure CSV files have consistent column names (`date`, `built_percentage`).
  - The script includes a commented-out filter for `class_name == 'built'`, which may need to be re-enabled depending on your data structure.
  - FutureWarning messages indicate a pandas version compatibility issue; consider passing `method="mean"` instead of `np.mean` in `data_cleaner` to suppress warnings.

### ARIMA.ipynb
**Purpose**: Performs time series forecasting using an ARIMA model, focusing on stationarity testing, parameter optimization, and future predictions.

- **Functionality**:
  - **Data Preprocessing**: Reads time series data from Excel files (e.g., `torbali_veriler.xls`), selecting specific columns (e.g., `BE:BW`) and rows for analysis.
  - **Stationarity Testing**: Uses the Augmented Dickey-Fuller (ADF) test to check for stationarity, applying differencing to non-stationary series (p-value > 0.05).
  - **Parameter Optimization**: Tests combinations of ARIMA parameters (p: 0–2, d: 0–1, q: 0–2) using the `optimizer` function, selecting the model with the lowest AIC for each series.
  - **Model Fitting and Forecasting**: Fits ARIMA models to each time series, generates forecasts for 20 future periods (2025–2034, semi-annual), and calculates error metrics (MAE, MSE, MAPE) for test data.
  - **Batch Processing**: Processes multiple zones (e.g., columns like `26G1`, `26H1`) and saves forecasted values and error metrics to Excel files (e.g., `report_df.xlsx`).

- **Inputs**:
  - Excel files with time series data (e.g., `torbali_veriler.xls`).
  - Configurable parameters: ADF test p-value threshold (0.05), ARIMA parameter ranges (p, d, q), forecast horizon (20 periods).

- **Outputs**:
  - Excel file (`report_df.xlsx`) containing model parameters (p, d, q), AIC scores, and error metrics (MAE, MSE, MAPE) for each zone.
  - DataFrame with forecasted values for each zone (e.g., `26Y1_forecasted`, `26H1_forecasted`) for 2025–2034.
  - Optional console output of best parameters and AIC scores for each series.

- **Usage Notes**:
  - Update file paths in `data_capturer` to match your Excel file location and structure.
  - Ensure the input Excel file has the expected column range (e.g., `BE:BW`) and row count (e.g., 45 rows).
  - The script may generate warnings for non-stationary parameters or convergence issues; check `mle_retvals` for debugging.
  - Forecasts assume semi-annual periods; adjust the index in the output DataFrame if a different frequency is needed.

### clustered_optimizer (1).ipynb
**Purpose**: Performs clustering-based time series forecasting using KMeans and exponential smoothing with damped trend for urban zone analysis.

- **Functionality**:
  - **Data Preprocessing**: Reads time series data from Excel files (e.g., `ilce_meandata-50_kms.xlsx`), splitting into training (first 14 rows) and test sets. Transposes data for clustering (zones as rows, time steps as columns).
  - **Clustering**: Uses KMeans to cluster zones based on time series patterns, selecting optimal cluster counts (2–10) via silhouette scores and the elbow method (visualized with inertia plots).
  - **Parameter Optimization**: Optimizes parameters (`alpha`, `beta`, `gamma`) for an exponential smoothing model with damped trend using grid search with adaptive refinement and parallel processing (`ProcessPoolExecutor`) to minimize MAPE.
  - **Forecasting**: Generates multi-step forecasts for test data or future periods using optimized parameters and computes error metrics (MAPE, MAE, MSE) for each zone.
  - **Batch Processing**: Processes multiple districts, combining data into a single Excel file (`all_meandata-50.xlsx`) and saving results (zone assignments, parameters, errors) for each district and cluster configuration.
  - **Output**: Saves results to Excel files (e.g., `ilce_X_clusters_errors_params-50_kms.xlsx`) with zone assignments, parameters, training MAPE, and test errors.

- **Inputs**:
  - Directory containing Excel files for each district (e.g., `izmir_poligon_path/ilce/ilce_meandata-50_kms.xlsx`).
  - District name (`ilce`) or `'all'` for multi-district processing.
  - Configurable parameters: grid sizes for optimization (default: `(10, 5, 3)`), number of parallel jobs (default: `-1` for all cores).

- **Outputs**:
  - Excel files for each district and cluster count (e.g., `ilce_X_clusters_errors_params-50_kms.xlsx`) containing zone assignments, parameters (`alpha`, `beta`, `gamma`), training MAPE, and test errors (MAPE, MAE, MSE).
  - Consolidated Excel file (`all_meandata-50.xlsx`) for multi-district data.
  - Elbow plot visualizations for cluster selection.

- **Usage Notes**:
  - Update `izmir_poligon_path` to point to your directory containing district subdirectories.
  - Ensure Excel files have a `date` column and zone columns with time series data.
  - Google Drive mounting is specific to Colab; modify data loading for local environments.
  - Adjust the training/test split (14 rows) in `info_gatherer` if your data has a different structure.
  - Parallel processing requires sufficient CPU cores for optimal performance.

### clustered_optimizer_50_kms_included.py
**Purpose**: A Python script version of the clustering-based time series forecasting pipeline, combining 50 km and non-50 km datasets for urban zone analysis using KMeans and exponential smoothing with damped trend, optimized for MAPE.

- **Functionality**:
  - **Data Preprocessing**: Reads time series data from Excel files (e.g., `ilce_meandata-50_kms.xlsx` and `ilce_meandata-50.xlsx`), concatenating 50 km and non-50 km datasets. Splits into training (first 14 rows) and test sets, and transposes data for clustering (zones as rows, time steps as columns).
  - **Clustering**: Uses KMeans to cluster zones based on time series patterns, selecting optimal cluster counts (2–10) via silhouette scores and the elbow method (visualized with inertia plots).
  - **Parameter Optimization**: Optimizes parameters (`alpha`, `beta`, `gamma`) for an exponential smoothing model with damped trend using grid search with adaptive refinement and parallel processing (`ProcessPoolExecutor`) to minimize MAPE.
  - **Forecasting**: Generates multi-step forecasts for test data using optimized parameters and computes error metrics (MAPE, MAE, MSE) for each zone.
  - **Output**: Saves results to Excel files (e.g., `ilce_X_clusters_errors_params-50_kms.xlsx`) with zone assignments, parameters (`alpha`, `beta`, `gamma`), training MAPE, and test errors.

- **Inputs**:
  - Directory containing Excel files for each district (e.g., `izmir_poligon_path/ilce/ilce_meandata-50_kms.xlsx` and `ilce_meandata-50.xlsx`).
  - District name (`ilce`) or `'all'` for multi-district processing.
  - Configurable parameters: grid sizes for optimization (default: `(10, 5, 3)`), number of parallel jobs (default: `-1` for all cores).

- **Outputs**:
  - Excel files for each district and cluster count (e.g., `ilce_X_clusters_errors_params-50_kms.xlsx`) containing zone assignments, parameters (`alpha`, `beta`, `gamma`), training MAPE, and test errors (MAPE, MAE, MSE).
  - Elbow plot visualizations for cluster selection.

- **Usage Notes**:
  - Update `izmir_poligon_path` to point to your directory containing district subdirectories.
  - Ensure both `ilce_meandata-50_kms.xlsx` and `ilce_meandata-50.xlsx` files exist for each district.
  - The script includes a commented-out Google Drive mount; modify data loading for local environments (e.g., remove `drive.mount`).
  - Adjust the training/test split (14 rows) in `info_gatherer` if your data has a different structure.
  - Parallel processing requires sufficient CPU cores for optimal performance.
  - Unlike the notebook version, this script does not support multi-district consolidation into a single Excel file (`all_meandata-50.xlsx`).

### clustered_optimizer_50_mse_optimized.py
**Purpose**: A Python script for clustering-based time series forecasting using KMeans and exponential smoothing with damped trend, optimized for Mean Squared Error (MSE) for urban zone analysis.

- **Functionality**:
  - **Data Preprocessing**: Reads time series data from Excel files (e.g., `ilce_meandata-50.xlsx`), splitting into training (first 14 rows) and test sets. Transposes data for clustering (zones as rows, time steps as columns).
  - **Clustering**: Uses KMeans to cluster zones based on time series patterns, selecting the top three cluster counts (2–10) via silhouette scores and visualizing the elbow method with inertia plots.
  - **Parameter Optimization**: Optimizes parameters (`alpha`, `beta`, `gamma`) for an exponential smoothing model with damped trend using a multi-stage grid search (10, 5, 3 points) to minimize MSE, with parallel processing (`ProcessPoolExecutor`).
  - **Forecasting**: Generates multi-step forecasts for test data using optimized parameters and computes error metrics (MAPE, MAE, MSE) for each zone.
  - **Output**: Saves results to Excel files (e.g., `ilce_X_clusters_errors_params-50-MSE-optimized.xlsx`) for each cluster configuration, including zone assignments, parameters (`alpha`, `beta`, `gamma`), training MSE, and test errors.

- **Inputs**:
  - Directory containing Excel files for each district (e.g., `izmir_poligon_path/ilce/ilce_meandata-50.xlsx`).
  - District name (`ilce`) or `'all'` for processing.
  - Configurable parameters: grid sizes for optimization (default: `(10, 5, 3)`), number of parallel jobs (default: 8).

- **Outputs**:
  - Excel files for each district and cluster count (e.g., `ilce_X_clusters_errors_params-50-MSE-optimized.xlsx`) containing zone assignments, parameters (`alpha`, `beta`, `gamma`), training MSE, and test errors (MAPE, MAE, MSE).
  - Elbow plot visualizations for cluster selection.

- **Usage Notes**:
  - Update `izmir_poligon_path` to point to your directory containing district subdirectories.
  - Ensure Excel files have a `date` column and zone columns with time series data.
  - The script includes a commented-out Google Drive mount; modify data loading for local environments (e.g., remove `drive.mount`).
  - Adjust the training/test split (14 rows) in `info_gatherer` if your data has a different structure.
  - The script restricts `gamma` to `[0.05, 0.95]`; adjust ranges in `optimizer_in_clusters` (e.g., to `[0.01, 0.99]`) to remove this constraint, as noted in comments.
  - Parallel processing with `ProcessPoolExecutor` requires sufficient CPU cores; adjust `n_jobs` based on your system.
  - This script is similar to `clustered_optimizer_50_kms_included.py` but optimizes for MSE instead of MAPE and uses only non-50 km data.

### clustered_optimizer_50_mae_optimized.py
**Purpose**: A Python script for clustering-based time series forecasting using KMeans and exponential smoothing with damped trend, optimized for Mean Absolute Error (MAE) for urban zone analysis.

- **Functionality**:
  - **Data Preprocessing**: Reads time series data from Excel files (e.g., `ilce_meandata-50.xlsx`), splitting into training (first 14 rows) and test sets. Transposes data for clustering (zones as rows, time steps as columns).
  - **Clustering**: Uses KMeans to cluster zones based on time series patterns, selecting the top three cluster counts (2–10) via silhouette scores and visualizing the elbow method with inertia plots.
  - **Parameter Optimization**: Optimizes parameters (`alpha`, `beta`, `gamma`) for an exponential smoothing model with damped trend using a multi-stage grid search (10, 5, 3 points) to minimize MAE, with parallel processing (`ProcessPoolExecutor`).
  - **Forecasting**: Generates multi-step forecasts for test data using optimized parameters and computes error metrics (MAPE, MAE, MSE) for each zone.
  - **Output**: Saves results to Excel files (e.g., `ilce_X_clusters_errors_params-50-MAE-optimized.xlsx`) for each cluster configuration, including zone assignments, parameters (`alpha`, `beta`, `gamma`), training MAE, and test errors.

- **Inputs**:
  - Directory containing Excel files for each district (e.g., `izmir_poligon_path/ilce/ilce_meandata-50.xlsx`).
  - District name (`ilce`) or `'all'` for processing.
  - Configurable parameters: grid sizes for optimization (default: `(10, 5, 3)`), number of parallel jobs (default: 8).

- **Outputs**:
  - Excel files for each district and cluster count (e.g., `ilce_X_clusters_errors_params-50-MAE-optimized.xlsx`) containing zone assignments, parameters (`alpha`, `beta`, `gamma`), training MAE, and test errors (MAPE, MAE, MSE).
  - Elbow plot visualizations for cluster selection.

- **Usage Notes**:
  - Update `izmir_poligon_path` to point to your directory containing district subdirectories.
  - Ensure Excel files have a `date` column and zone columns with time series data.
  - The script includes a commented-out Google Drive mount; modify data loading for local environments (e.g., remove `drive.mount`).
  - Adjust the training/test split (14 rows) in `info_gatherer` if your data has a different structure.
  - The script restricts `gamma` to `[0.05, 0.95]`; adjust ranges in `optimizer_in_clusters` (e.g., to `[0.01, 0.99]`) to remove this constraint, as noted in comments.
  - Parallel processing with `ProcessPoolExecutor` requires sufficient CPU cores; adjust `n_jobs` based on your system.
  - This script is similar to `clustered_optimizer_50_mse_optimized.py` but optimizes for MAE instead of MSE, affecting parameter selection and error reporting.

### nonclustered_optimizer_50_mse_optimized.py
**Purpose**: A Python script for non-clustered time series forecasting using exponential smoothing with damped trend, optimized for Mean Squared Error (MSE) for urban zone analysis.

- **Functionality**:
  - **Data Preprocessing**: Reads time series data from Excel files (e.g., `ilce_meandata-50.xlsx`), splitting into training (first 14 rows) and test sets. Optionally processes specific columns (e.g., `BE:BW`) from files like `torbali_veriler.xlsx` via `data_capturer`.
  - **Parameter Optimization**: Optimizes parameters (`alpha`, `beta`, `gamma`) for an exponential smoothing model with damped trend using a three-stage grid search (coarse, refined, fine-tuned) to minimize MSE. Uses parallel processing (`ThreadPoolExecutor`) for efficiency.
  - **Forecasting**: Generates one-step-ahead forecasts for training data and multi-step forecasts for test data, computing error metrics (MAPE, MAE, MSE) for each zone.
  - **Output**: Saves results to Excel files (e.g., `ilce_errors_params-50-MSE-optimized.xlsx`) with zone-specific parameters (`alpha`, `beta`, `gamma`), training MSE, and test errors (MAPE, MAE, MSE).

- **Inputs**:
  - Directory containing Excel files for each district (e.g., `izmir_poligon_path/ilce/ilce_meandata-50.xlsx`).
  - District name (`ilce`) or `'all'` for processing.
  - Configurable parameters: grid sizes for optimization (20, 10, 10 across three stages), number of parallel workers (default: 8).

- **Outputs**:
  - Excel files for each district (e.g., `ilce_errors_params-50-MSE-optimized.xlsx`) containing zone IDs, parameters (`alpha`, `beta`, `gamma`), training MSE, and test errors (MAPE, MAE, MSE).

- **Usage Notes**:
  - Update `izmir_poligon_path` to point to your directory containing district subdirectories.
  - Ensure Excel files have a `date` column and zone columns with time series data.
  - The script includes a commented-out Google Drive mount; modify data loading for local environments (e.g., remove `drive.mount`).
  - Adjust the training/test split (14 rows) in `excel_writer` if your data has a different structure.
  - The `data_capturer` function is specific to `torbali_veriler.xlsx`; modify or remove for general use with `ilce_meandata-50.xlsx`.
  - The script restricts `gamma` to `[0.8, 0.98]`; adjust ranges in `optimizer` (e.g., to `[0.001, 0.999]`) to remove this constraint, as noted in comments.
  - Parallel processing with `ThreadPoolExecutor` is optimized for smaller parameter grids; larger grids use batch processing to manage memory.

### nonclustered_optimizer_50_mse_optimized (1).py
**Purpose**: An alternative Python script for non-clustered time series forecasting using exponential smoothing with damped trend, optimized for Mean Squared Error (MSE) for urban zone analysis, with specific data processing capabilities.

- **Functionality**:
  - **Data Preprocessing**: Reads time series data from Excel files (e.g., `ilce_meandata-50.xlsx`), splitting into training (first 14 rows) and test sets. Includes a `data_capturer` function to process specific columns (e.g., `BE:BW`) from files like `torbali_veriler.xlsx`, extracting 45 rows of data.
  - **Parameter Optimization**: Optimizes parameters (`alpha`, `beta`, `gamma`) for an exponential smoothing model with damped trend using a three-stage grid search (coarse: 20 points, refined: 10 points, fine-tuned: 10 points) to minimize MSE. Uses parallel processing (`ThreadPoolExecutor`) for efficiency, with batch processing for larger parameter grids.
  - **Forecasting**: Generates one-step-ahead forecasts for training data and multi-step forecasts for test data, computing error metrics (MAPE, MAE, MSE) for each zone.
  - **Output**: Saves results to Excel files (e.g., `ilce_errors_params-50-MSE-optimized.xlsx`) with zone-specific parameters (`alpha`, `beta`, `gamma`), training MSE, and test errors (MAPE, MAE, MSE).

- **Inputs**:
  - Directory containing Excel files for each district (e.g., `izmir_poligon_path/ilce/ilce_meandata-50.xlsx`).
  - District name (`ilce`) or `'all'` for processing.
  - Optional: Specific Excel file (`torbali_veriler.xlsx`) for `data_capturer` processing.
  - Configurable parameters: grid sizes for optimization (20, 10, 10 across three stages), number of parallel workers (default: 8).

- **Outputs**:
  - Excel files for each district (e.g., `ilce_errors_params-50-MSE-optimized.xlsx`) containing zone IDs, parameters (`alpha`, `beta`, `gamma`), training MSE, and test errors (MAPE, MAE, MSE).

- **Usage Notes**:
  - Update `izmir_poligon_path` to point to your directory containing district subdirectories.
  - Ensure Excel files have a `date` column and zone columns with time series data for general use, or match the structure expected by `data_capturer` (e.g., `torbali_veriler.xlsx` with columns `BE:BW` and 45 rows).
  - The script includes a commented-out Google Drive mount; modify data loading for local environments (e.g., remove `drive.mount`).
  - Adjust the training/test split (14 rows) in `excel_writer` if your data has a different structure.
  - The `data_capturer` function is tailored for `torbali_veriler.xlsx`; modify or bypass it for use with `ilce_meandata-50.xlsx`.
  - The script restricts `gamma` to `[0.8, 0.98]`; adjust ranges in `optimizer` (e.g., to `[0.001, 0.999]`) to remove this constraint, as noted in comments.
  - Parallel processing with `ThreadPoolExecutor` is optimized for smaller parameter grids (<1000 combinations); larger grids use batch processing with a batch size of 1000.
  - This script is similar to `nonclustered_optimizer_50_mse_optimized.py` but includes additional data processing via `data_capturer` and maintains a consistent `gamma` range restriction.

### nonclustered_optimizer_50_mae_optimized.py
**Purpose**: A Python script for non-clustered time series forecasting using exponential smoothing with damped trend, optimized for Mean Absolute Error (MAE) for urban zone analysis, with specific data processing capabilities.

- **Functionality**:
  - **Data Preprocessing**: Reads time series data from Excel files (e.g., `ilce_meandata-50.xlsx`), splitting into training (first 14 rows) and test sets. Includes a `data_capturer` function to process specific columns (e.g., `BE:BW`) from files like `torbali_veriler.xlsx`, extracting 45 rows of data.
  - **Parameter Optimization**: Optimizes parameters (`alpha`, `beta`, `gamma`) for an exponential smoothing model with damped trend using a three-stage grid search (coarse: 20 points, refined: 10 points, fine-tuned: 10 points) to minimize MAE. Uses parallel processing (`ThreadPoolExecutor`) for efficiency, with batch processing for larger parameter grids.
  - **Forecasting**: Generates one-step-ahead forecasts for training data and multi-step forecasts for test data, computing error metrics (MAPE, MAE, MSE) for each zone.
  - **Output**: Saves results to Excel files (e.g., `ilce_errors_params-50-MAE-optimized.xlsx`) with zone-specific parameters (`alpha`, `beta`, `gamma`), training MAE, and test errors (MAPE, MAE, MSE).

- **Inputs**:
  - Directory containing Excel files for each district (e.g., `izmir_poligon_path/ilce/ilce_meandata-50.xlsx`).
  - District name (`ilce`) or `'all'` for processing.
  - Optional: Specific Excel file (`torbali_veriler.xlsx`) for `data_capturer` processing.
  - Configurable parameters: grid sizes for optimization (20, 10, 10 across three stages), number of parallel workers (default: 8).

- **Outputs**:
  - Excel files for each district (e.g., `ilce_errors_params-50-MAE-optimized.xlsx`) containing zone IDs, parameters (`alpha`, `beta`, `gamma`), training MAE, and test errors (MAPE, MAE, MSE).

- **Usage Notes**:
  - Update `izmir_poligon_path` to point to your directory containing district subdirectories.
  - Ensure Excel files have a `date` column and zone columns with time series data for general use, or match the structure expected by `data_capturer` (e.g., `torbali_veriler.xlsx` with columns `BE:BW` and 45 rows).
  - The script includes a commented-out Google Drive mount; modify data loading for local environments (e.g., remove `drive.mount`).
  - Adjust the training/test split (14 rows) in `excel_writer` if your data has a different structure.
  - The `data_capturer` function is tailored for `torbali_veriler.xlsx`; modify or bypass it for use with `ilce_meandata-50.xlsx`.
  - The script restricts `gamma` to `[0.8, 0.98]`; adjust ranges in `optimizer` (e.g., to `[0.001, 0.999]`) to remove this constraint, as noted in comments.
  - Parallel processing with `ThreadPoolExecutor` is optimized for smaller parameter grids (<1000 combinations); larger grids use batch processing with a batch size of 1000.
  - This script is similar to `nonclustered_optimizer_50_mse_optimized (1).py` but optimizes for MAE instead of MSE, affecting parameter selection and error reporting.

### nonclustered_optimizer.py
**Purpose**: A Python script for non-clustered time series forecasting using exponential smoothing with damped trend, optimized for Mean Absolute Percentage Error (MAPE) for urban zone analysis.

- **Functionality**:
  - **Data Preprocessing**: Reads time series data from Excel files (e.g., `ilce_meandata-50.xlsx`), splitting into training (first 14 rows) and test sets. Optionally processes specific columns (e.g., `BE:BW`) from files like `torbali_veriler.xlsx` via `data_capturer`.
  - **Parameter Optimization**: Optimizes parameters (`alpha`, `beta`, `gamma`) for an exponential smoothing model with damped trend using a three-stage grid search (coarse, refined, fine-tuned) to minimize MAPE. Uses parallel processing (`ThreadPoolExecutor`) for efficiency.
  - **Forecasting**: Generates one-step-ahead forecasts for training data and multi-step forecasts for test data, computing error metrics (MAPE, MAE, MSE) for each zone.
  - **Output**: Saves results to Excel files (e.g., `ilce_errors_params-50.xlsx`) with zone-specific parameters (`alpha`, `beta`, `gamma`), training MAPE, and test errors (MAPE, MAE, MSE).

- **Inputs**:
  - Directory containing Excel files for each district (e.g., `izmir_poligon_path/ilce/ilce_meandata-50.xlsx`).
  - District name (`ilce`) or `'all'` for processing.
  - Configurable parameters: grid sizes for optimization (20, 10, 10 across three stages), number of parallel workers (default: 8).

- **Outputs**:
  - Excel files for each district (e.g., `ilce_errors_params-50.xlsx`) containing zone IDs, parameters (`alpha`, `beta`, `gamma`), training MAPE, and test errors (MAPE, MAE, MSE).

- **Usage Notes**:
  - Update `izmir_poligon_path` to point to your directory containing district subdirectories.
  - Ensure Excel files have a `date` column and zone columns with time series data.
  - The script includes an active Google Drive mount; comment out `drive.mount` or modify data loading for local environments.
  - Adjust the training/test split (14 rows) in `excel_writer` if your data has a different structure.
  - The `data_capturer` function is specific to `torbali_veriler.xlsx`; modify or remove for general use with `ilce_meandata-50.xlsx`.
  - The script restricts `gamma` to `[0.8, 0.98]`; adjust ranges in `optimizer` (e.g., to `[0.001, 0.999]`) to remove this constraint, as noted in comments.
  - Parallel processing with `ThreadPoolExecutor` is optimized for smaller parameter grids; larger grids use batch processing to manage memory.
  - This script is similar to `nonclustered_optimizer_50_mse_optimized.py` but optimizes for MAPE instead of MSE, affecting parameter selection and error reporting.

### lm_trial.py
**Purpose**: A Python script for forecasting urban zone time series using linear regression models combined with KMeans clustering, utilizing multiple features derived from km² and percentage data to predict future built-up areas.

- **Functionality**:
  - **Data Preprocessing**: Reads time series data from Excel files (e.g., `all_meandata-50_kms.xlsx`, `all_meandata-50.xlsx`, `all_3_clusters_errors_params-50_kms.xlsx`). Creates features such as percentage changes, average changes over time, and semi-annual indicators using `prepare_your_data` and `training_prep`.
  - **Clustering**: Uses KMeans to cluster zones based on derived features, creating separate linear regression models for each cluster (3–10 clusters) via `n_model_creator`. Evaluates cluster quality using silhouette scores.
  - **Modeling**: Trains linear regression models for each cluster, predicting one-step-ahead built-up area percentages. Features include previous changes, starting percentages, and exponential smoothing parameters (`alpha`, `beta`, `gamma`).
  - **Forecasting**: Generates multi-step forecasts iteratively using `predictor`, updating input data with predictions for subsequent steps. Handles negative predictions by tracking them for debugging.
  - **Error Evaluation**: Computes error metrics (MAPE, MAE, MSE) for different cluster configurations (3–10 clusters) using `kmean_model_iterator`, saving results to an Excel file (`regression_n_clustered_model_errors.xlsx`).
  - **Output**: Saves error metrics for different cluster counts, comparing model performance across configurations.

- **Inputs**:
  - Directory containing Excel files (e.g., `/content/all_meandata-50_kms.xlsx`, `all_meandata-50.xlsx`, `all_3_clusters_errors_params-50_kms.xlsx`).
  - Configurable parameters: number of clusters (3–10), outlier threshold for percentage change (default: 30000 in `training_prep`, 50 in `one_step_predictor`), test set size (default: 5 periods).

- **Outputs**:
  - Excel file (`regression_n_clustered_model_errors.xlsx`) containing MAE, MAPE, and MSE for each cluster configuration (3–10 clusters).
  - Predicted values and actual observations for test data, stored in DataFrames during processing.

- **Usage Notes**:
  - Update the root path (`/content`) to point to your directory containing input Excel files.
  - Ensure input files have consistent column names and structures (e.g., `date` column and zone columns).
  - The script assumes 1530 zones; adjust the hardcoded value in `prepare_your_data` and `training_prep` if your data differs.
  - The outlier threshold (30000) in `training_prep` is high; consider lowering it (e.g., to 50 as in `one_step_predictor`) for stricter filtering.
  - The script uses the last 4 periods for feature creation; modify `prepare_your_data` if a different time window is needed.
  - Negative predictions are tracked in `less_than_0`; review these for model debugging.
  - The script is designed for multi-step forecasting but may be computationally intensive due to iterative predictions and clustering iterations.

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

Please follow PEP 8 guidelines and include documentation for new features or scripts.

## License
[No License](LICENSE)
