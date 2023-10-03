import anvil.server
import pandas as pd
import numpy as np
import shap
from catboost import CatBoostClassifier, Pool
import matplotlib.pyplot as plt
import io

@anvil.server.callable
def process_data(file):
    # Read the uploaded file into a DataFrame
    with file.open() as f:
        df = pd.read_csv(f)

    # Get all categorical features
    all_categorical_features = df.select_dtypes(exclude=['int64', 'float64']).columns.tolist()
    return all_categorical_features

@anvil.server.callable
def train_and_get_shap(file, selected_feature):
    # Read the uploaded file into a DataFrame
    with file.open() as f:
        df = pd.read_csv(f)

    # Data Preparation
    y = df[selected_feature]
    X = df.drop(columns=[selected_feature]).select_dtypes(include=['int64', 'float64'])
    cat_features = list(range(0, X.shape[1]))

    # Model Training
    model = CatBoostClassifier(loss_function='MultiClass', iterations=100, learning_rate=0.3, random_seed=12)
    model.fit(X, y, cat_features=cat_features, verbose=False, plot=False)

    # SHAP Value Calculation
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(Pool(X, y, cat_features=cat_features))

    # SHAP Summary Plot
    sorted_indices = get_shap_order(shap_values[0])
    sorted_shap_values = shap_values[0][:, sorted_indices]
    sorted_features = X.iloc[:, sorted_indices]

    plt.figure(figsize=(20, 10))
    shap.summary_plot(sorted_shap_values, sorted_features, show=False)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    # Convert the plot to a Media object and return
    plot_media = anvil.BlobMedia('image/png', buf)

    # SHAP Summary DataFrame
    shap_summary_df = create_shap_summary_df(sorted_shap_values, sorted_features)
    return plot_media, shap_summary_df.to_dict(orient='records')

def get_shap_order(shap_values):
    abs_shap_values = np.abs(shap_values)
    max_abs_shap_values = np.max(abs_shap_values, axis=0)
    return np.argsort(-max_abs_shap_values)

def create_shap_summary_df(shap_values, features):
    max_abs_indices = np.argmax(np.abs(shap_values), axis=0)
    max_shap_values = shap_values[max_abs_indices, np.arange(shap_values.shape[1])]

    feature_values_at_max_shap = [features[col].iloc[max_abs_indices[i]] for i, col in enumerate(features.columns)]

    percentile_list = []
    for col, max_idx in zip(features.columns, max_abs_indices):
        value = features[col].iloc[max_idx]
        percentile = np.percentile(features[col], 100 * (np.sum(features[col] <= value) / len(features[col])))
        percentile_list.append(percentile)

    df = pd.DataFrame({
        'feature': features.columns,
        'max_shap_value': max_shap_values,
        'feature_value_at_max_shap': feature_values_at_max_shap,
        'percentile': percentile_list
    })

    return df

