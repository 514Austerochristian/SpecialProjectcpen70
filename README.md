# Data Cleaning & Normalization Tool

A comprehensive web-based application for uploading, cleaning, and normalizing CSV data using pandas, scikit-learn, and numpy. This tool provides an intuitive interface for data preprocessing tasks commonly required in data science and machine learning projects.

## Features

### 🔄 Data Upload & Processing
- **CSV File Upload**: Drag-and-drop or browse to upload CSV files (up to 16MB)
- **Automatic Data Analysis**: Instant dataset statistics and preview
- **Data Type Detection**: Automatic identification of numeric and categorical columns

### 🧹 Data Cleaning
- **Missing Value Handling**: 
  - Drop rows with missing values
  - Fill with mean, median, or mode
  - Custom imputation strategies
- **Duplicate Removal**: Identify and remove duplicate rows
- **Outlier Detection**: 
  - IQR (Interquartile Range) method
  - Z-score method with configurable thresholds

### 📊 Data Normalization
- **Standard Scaler**: Z-score normalization (mean=0, std=1)
- **Min-Max Scaler**: Scale features to [0,1] range
- **Robust Scaler**: Uses median and IQR, robust to outliers
- **Column Selection**: Choose specific columns to normalize

### 🏷️ Categorical Encoding
- **Label Encoding**: Convert categorical variables to numeric labels
- **One-Hot Encoding**: Create binary columns for each category
- **Selective Encoding**: Choose which categorical columns to encode

### 📈 Data Visualization
- **Correlation Matrix**: Heatmap of feature correlations
- **Distribution Plots**: Histograms for data distribution analysis
- **Scatter Plots**: Explore relationships between variables
- **Interactive Plots**: Powered by Plotly for dynamic visualization

### 📝 Processing History
- **Operation Tracking**: Keep track of all data transformations
- **Reset Functionality**: Revert to original dataset at any time
- **Download Cleaned Data**: Export processed data as CSV

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd SpecialProjectcpen70
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python app.py
   ```

4. **Access the application**:
   Open your browser and navigate to `http://localhost:5000`

## Usage

### 1. Upload Your Data
- Click "Choose File" or drag and drop your CSV file
- The application will automatically analyze your data and display:
  - Dataset dimensions (rows × columns)
  - Data types and column information
  - Missing values and duplicate counts
  - Sample data preview

### 2. Clean Your Data
Choose from various cleaning options:
- **Handle Missing Values**: Select how to deal with missing data
- **Remove Duplicates**: Eliminate duplicate rows
- **Handle Outliers**: Remove or treat outliers using statistical methods

### 3. Normalize Your Data
Select normalization method and columns:
- **Standard Scaler**: For normally distributed data
- **Min-Max Scaler**: For bounded ranges
- **Robust Scaler**: For data with outliers

### 4. Encode Categorical Data
Transform categorical variables:
- **Label Encoding**: For ordinal data
- **One-Hot Encoding**: For nominal data

### 5. Visualize Your Data
Create interactive plots:
- **Correlation Matrix**: Understand feature relationships
- **Distribution Plots**: Analyze data distributions
- **Scatter Plots**: Explore variable relationships

### 6. Download Results
- Review processing history
- Download cleaned dataset
- Reset to original data if needed

## Sample Data

A sample dataset (`sample_data.csv`) is included with the following features:
- Employee information with various data types
- Missing values for testing imputation
- Duplicate entries for testing deduplication
- Numeric and categorical columns for comprehensive testing

## Technology Stack

- **Backend**: Flask (Python web framework)
- **Data Processing**: 
  - pandas (data manipulation)
  - numpy (numerical operations)
  - scikit-learn (preprocessing and normalization)
  - scipy (statistical operations)
- **Visualization**: 
  - Plotly (interactive plots)
  - matplotlib & seaborn (statistical plotting)
- **Frontend**: 
  - HTML5, CSS3, JavaScript
  - Bootstrap 5 (responsive design)
  - Font Awesome (icons)

## API Endpoints

- `POST /upload` - Upload CSV file
- `GET /dataset_info` - Get dataset statistics
- `POST /clean_data` - Apply data cleaning operations
- `POST /normalize_data` - Apply normalization
- `POST /encode_categorical` - Encode categorical variables
- `POST /visualize` - Generate visualizations
- `GET /download_cleaned` - Download processed data
- `POST /reset_data` - Reset to original dataset

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues, questions, or contributions, please open an issue on the GitHub repository.