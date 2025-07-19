from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import os
import json
from werkzeug.utils import secure_filename
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variable to store the current dataset
current_data = None
original_data = None
cleaning_history = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global current_data, original_data, cleaning_history
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and file.filename.lower().endswith('.csv'):
        try:
            # Read CSV file
            df = pd.read_csv(file)
            current_data = df.copy()
            original_data = df.copy()
            cleaning_history = []
            
            # Generate basic statistics
            stats = get_dataset_stats(df)
            
            return jsonify({
                'message': 'File uploaded successfully',
                'stats': stats,
                'columns': df.columns.tolist(),
                'shape': df.shape,
                'head': df.head().to_dict('records')
            })
        except Exception as e:
            return jsonify({'error': f'Error reading CSV file: {str(e)}'}), 400
    else:
        return jsonify({'error': 'Please upload a CSV file'}), 400

@app.route('/dataset_info')
def dataset_info():
    global current_data
    if current_data is None:
        return jsonify({'error': 'No dataset loaded'}), 400
    
    stats = get_dataset_stats(current_data)
    return jsonify(stats)

@app.route('/clean_data', methods=['POST'])
def clean_data():
    global current_data, cleaning_history
    
    if current_data is None:
        return jsonify({'error': 'No dataset loaded'}), 400
    
    cleaning_options = request.json
    df = current_data.copy()
    
    try:
        # Handle missing values
        if cleaning_options.get('handle_missing'):
            method = cleaning_options.get('missing_method', 'drop')
            if method == 'drop':
                df = df.dropna()
                cleaning_history.append('Dropped rows with missing values')
            elif method == 'fill_mean':
                numeric_columns = df.select_dtypes(include=[np.number]).columns
                df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
                cleaning_history.append('Filled missing numeric values with mean')
            elif method == 'fill_median':
                numeric_columns = df.select_dtypes(include=[np.number]).columns
                df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
                cleaning_history.append('Filled missing numeric values with median')
            elif method == 'fill_mode':
                for column in df.columns:
                    df[column] = df[column].fillna(df[column].mode()[0] if not df[column].mode().empty else 'Unknown')
                cleaning_history.append('Filled missing values with mode')
        
        # Remove duplicates
        if cleaning_options.get('remove_duplicates'):
            initial_shape = df.shape[0]
            df = df.drop_duplicates()
            removed_count = initial_shape - df.shape[0]
            cleaning_history.append(f'Removed {removed_count} duplicate rows')
        
        # Handle outliers
        if cleaning_options.get('handle_outliers'):
            method = cleaning_options.get('outlier_method', 'iqr')
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            if method == 'iqr':
                for column in numeric_columns:
                    Q1 = df[column].quantile(0.25)
                    Q3 = df[column].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
                cleaning_history.append('Removed outliers using IQR method')
            elif method == 'z_score':
                from scipy import stats
                for column in numeric_columns:
                    z_scores = np.abs(stats.zscore(df[column]))
                    df = df[z_scores < 3]
                cleaning_history.append('Removed outliers using Z-score method (threshold=3)')
        
        current_data = df
        stats = get_dataset_stats(df)
        
        return jsonify({
            'message': 'Data cleaned successfully',
            'stats': stats,
            'cleaning_history': cleaning_history,
            'shape': df.shape
        })
        
    except Exception as e:
        return jsonify({'error': f'Error cleaning data: {str(e)}'}), 400

@app.route('/normalize_data', methods=['POST'])
def normalize_data():
    global current_data, cleaning_history
    
    if current_data is None:
        return jsonify({'error': 'No dataset loaded'}), 400
    
    normalization_options = request.json
    df = current_data.copy()
    
    try:
        method = normalization_options.get('method', 'standard')
        columns_to_normalize = normalization_options.get('columns', [])
        
        if not columns_to_normalize:
            # Default to all numeric columns
            columns_to_normalize = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if method == 'standard':
            scaler = StandardScaler()
            df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
            cleaning_history.append(f'Applied StandardScaler to columns: {columns_to_normalize}')
        elif method == 'minmax':
            scaler = MinMaxScaler()
            df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
            cleaning_history.append(f'Applied MinMaxScaler to columns: {columns_to_normalize}')
        elif method == 'robust':
            scaler = RobustScaler()
            df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
            cleaning_history.append(f'Applied RobustScaler to columns: {columns_to_normalize}')
        
        current_data = df
        stats = get_dataset_stats(df)
        
        return jsonify({
            'message': 'Data normalized successfully',
            'stats': stats,
            'cleaning_history': cleaning_history,
            'normalized_columns': columns_to_normalize
        })
        
    except Exception as e:
        return jsonify({'error': f'Error normalizing data: {str(e)}'}), 400

@app.route('/encode_categorical', methods=['POST'])
def encode_categorical():
    global current_data, cleaning_history
    
    if current_data is None:
        return jsonify({'error': 'No dataset loaded'}), 400
    
    encoding_options = request.json
    df = current_data.copy()
    
    try:
        method = encoding_options.get('method', 'label')
        columns_to_encode = encoding_options.get('columns', [])
        
        if not columns_to_encode:
            # Default to all categorical columns
            columns_to_encode = df.select_dtypes(include=['object']).columns.tolist()
        
        if method == 'label':
            le = LabelEncoder()
            for column in columns_to_encode:
                df[column] = le.fit_transform(df[column].astype(str))
            cleaning_history.append(f'Applied Label Encoding to columns: {columns_to_encode}')
        elif method == 'onehot':
            df = pd.get_dummies(df, columns=columns_to_encode, prefix=columns_to_encode)
            cleaning_history.append(f'Applied One-Hot Encoding to columns: {columns_to_encode}')
        
        current_data = df
        stats = get_dataset_stats(df)
        
        return jsonify({
            'message': 'Categorical data encoded successfully',
            'stats': stats,
            'cleaning_history': cleaning_history,
            'encoded_columns': columns_to_encode
        })
        
    except Exception as e:
        return jsonify({'error': f'Error encoding categorical data: {str(e)}'}), 400

@app.route('/visualize', methods=['POST'])
def visualize():
    global current_data
    
    if current_data is None:
        return jsonify({'error': 'No dataset loaded'}), 400
    
    viz_options = request.json
    plot_type = viz_options.get('type', 'correlation')
    
    try:
        if plot_type == 'correlation':
            numeric_df = current_data.select_dtypes(include=[np.number])
            if numeric_df.empty:
                return jsonify({'error': 'No numeric columns for correlation plot'}), 400
            
            fig = px.imshow(numeric_df.corr(), 
                          title='Correlation Matrix',
                          color_continuous_scale='RdBu_r',
                          aspect='auto')
            
        elif plot_type == 'distribution':
            column = viz_options.get('column')
            if column not in current_data.columns:
                return jsonify({'error': f'Column {column} not found'}), 400
            
            if current_data[column].dtype in ['object', 'category']:
                fig = px.histogram(current_data, x=column, title=f'Distribution of {column}')
            else:
                fig = px.histogram(current_data, x=column, title=f'Distribution of {column}', nbins=30)
        
        elif plot_type == 'scatter':
            x_col = viz_options.get('x_column')
            y_col = viz_options.get('y_column')
            
            if x_col not in current_data.columns or y_col not in current_data.columns:
                return jsonify({'error': 'Invalid column selection'}), 400
            
            fig = px.scatter(current_data, x=x_col, y=y_col, 
                           title=f'{x_col} vs {y_col}')
        
        graphJSON = json.dumps(fig, cls=PlotlyJSONEncoder)
        return jsonify({'plot': graphJSON})
        
    except Exception as e:
        return jsonify({'error': f'Error creating visualization: {str(e)}'}), 400

@app.route('/download_cleaned')
def download_cleaned():
    global current_data
    
    if current_data is None:
        return jsonify({'error': 'No dataset loaded'}), 400
    
    # Create a temporary file
    output = io.StringIO()
    current_data.to_csv(output, index=False)
    output.seek(0)
    
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='cleaned_data.csv'
    )

@app.route('/reset_data', methods=['POST'])
def reset_data():
    global current_data, original_data, cleaning_history
    
    if original_data is None:
        return jsonify({'error': 'No original dataset to reset to'}), 400
    
    current_data = original_data.copy()
    cleaning_history = []
    
    stats = get_dataset_stats(current_data)
    
    return jsonify({
        'message': 'Data reset to original state',
        'stats': stats,
        'cleaning_history': cleaning_history
    })

def get_dataset_stats(df):
    """Generate comprehensive statistics for the dataset"""
    stats = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'missing_values': {k: int(v) for k, v in df.isnull().sum().to_dict().items()},
        'memory_usage': int(df.memory_usage(deep=True).sum()),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
        'duplicate_rows': int(df.duplicated().sum())
    }
    
    # Basic statistics for numeric columns
    if stats['numeric_columns']:
        numeric_stats = df[stats['numeric_columns']].describe().to_dict()
        # Convert numpy types to Python types for JSON serialization
        for col in numeric_stats:
            for stat in numeric_stats[col]:
                if hasattr(numeric_stats[col][stat], 'item'):
                    numeric_stats[col][stat] = numeric_stats[col][stat].item()
        stats['numeric_stats'] = numeric_stats
    
    # Value counts for categorical columns (limited to top 10)
    categorical_stats = {}
    for col in stats['categorical_columns']:
        value_counts = df[col].value_counts().head(10).to_dict()
        # Convert numpy types to Python types
        categorical_stats[col] = {k: int(v) for k, v in value_counts.items()}
    stats['categorical_stats'] = categorical_stats
    
    return stats

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)