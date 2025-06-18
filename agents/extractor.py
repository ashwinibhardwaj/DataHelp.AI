# agents/extractor.py
import pandas as pd
from collections import defaultdict

def extract_data_info(df):
    info = {}

    # Basic structure
    info['shape'] = df.shape
    info['columns'] = list(df.columns)
    info['dtypes'] = df.dtypes.astype(str).to_dict()
    info['nulls'] = df.isnull().sum().to_dict()
    info['nunique'] = df.nunique().to_dict()
    info['describe'] = df.describe(include='all').fillna('').to_dict()
    info['sample'] = df.sample(min(5, len(df)), random_state=42).to_dict(orient='records')

    # Column categorization
    info['column_types'] = {
        'numerical': df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
        'categorical': df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist(),
        'datetime': df.select_dtypes(include=['datetime64']).columns.tolist()
    }

    # Top 3 frequent values for categorical columns
    top_values = defaultdict(dict)
    for col in info['column_types']['categorical']:
        top = df[col].value_counts().head(3).to_dict()
        top_values[col] = top
    info['top_frequent_values'] = dict(top_values)

    # Correlation matrix (rounded for readability)
    try:
        info['correlation'] = df.corr(numeric_only=True).round(2).to_dict()
    except:
        info['correlation'] = {}

    # Suggested target column (optional heuristic)
    potential_targets = [col for col in df.columns if df[col].nunique() <= 10 and df[col].dtype != 'object']
    info['potential_targets'] = potential_targets[-1:] if potential_targets else []

    return info
