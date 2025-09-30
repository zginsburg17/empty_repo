import nbformat as nbf

print("Building comprehensive Jupyter notebook for FR 2052a anomaly detection...")

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
    "# Anomaly Detection for FR 2052a Regulatory Reporting\n\n"
    "## Overview\n"
    "This notebook provides a scalable framework for detecting anomalies in financial data from ClickHouse tables.\n\n"
    "**Focus Areas:**\n"
    "- **Outliers**: Extreme values that deviate significantly from normal patterns\n"
    "- **Miscategorizations**: Data points that don't fit expected patterns for their regulatory bucket\n\n"
    "## Key Features\n"
    "- **Flexible Table Selection**: Easily switch between different tables to analyze\n"
    "- **Flexible Column Selection**: Choose which columns to analyze\n"
    "- **Scalable Processing**: Handles millions of rows through batch processing\n"
    "- **Multiple Detection Methods**: Statistical, ML-based, and deep learning approaches\n"
    "- **Memory Efficient**: Optimized for large datasets\n"
    "- **Well Documented**: Extensive comments throughout\n\n"
    "## Methods Included\n"
    "1. **Statistical Methods**: Z-score and IQR for outlier detection\n"
    "2. **Isolation Forest**: Tree-based anomaly detection\n"
    "3. **Autoencoder (TensorFlow)**: Deep learning reconstruction-based detection\n"
    "4. **DBSCAN**: Density-based clustering for anomaly detection\n\n"
    "## Prerequisites\n"
    "```bash\n"
    "pip install -r requirements.txt\n"
    "```"
))

cells.append(nbf.v4.new_markdown_cell("## 1. Import Required Libraries"))

cells.append(nbf.v4.new_code_cell(
    "import pandas as pd\n"
    "import numpy as np\n"
    "from clickhouse_driver import Client\n"
    "import matplotlib.pyplot as plt\n"
    "import seaborn as sns\n"
    "from sklearn.preprocessing import StandardScaler, RobustScaler\n"
    "from sklearn.ensemble import IsolationForest\n"
    "from sklearn.cluster import DBSCAN\n"
    "from sklearn.decomposition import PCA\n"
    "import tensorflow as tf\n"
    "from tensorflow import keras\n"
    "from tensorflow.keras import layers\n"
    "import warnings\n"
    "warnings.filterwarnings('ignore')\n\n"
    "print(f\"TensorFlow version: {tf.__version__}\")\n"
    "print(f\"Pandas version: {pd.__version__}\")\n"
    "print(f\"NumPy version: {np.__version__}\")"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 2. Configure ClickHouse Connection\n\n"
    "**IMPORTANT**: Replace the placeholder credentials with your actual database credentials."
))

cells.append(nbf.v4.new_code_cell(
    "ch = Client(\n"
    "    host='sd-mccg-kpyd.dam.nsroot.net',\n"
    "    port=9000,\n"
    "    database='dcw_glrs_clean',\n"
    "    user='YOUR_USERNAME',\n"
    "    password='YOUR_PASSWORD',\n"
    "    client_name='anomaly_detection',\n"
    "    alt_hosts='sd-t2i1-7o4d.nam.nsroot.net:9000,sd-ybfb-pfe0.nam.nsroot.net:9000',\n"
    "    settings={'use_numpy': True}\n"
    ")\n\n"
    "print(\"Database connection configured.\")"
))

cells.append(nbf.v4.new_markdown_cell(
    "### 2.1 Configure Target Table\n\n"
    "**Set the table name you want to analyze.** Change this to analyze different tables in the database."
))

cells.append(nbf.v4.new_code_cell(
    "TABLE_NAME = 'tb_elp_dep_dl'\n\n"
    "print(f\"Target table: {TABLE_NAME}\")"
))

print("Adding data exploration functions...")

cells.append(nbf.v4.new_markdown_cell(
    "## 3. Data Exploration Functions\n\n"
    "These helper functions let you explore the table schema and sample data before selecting columns for analysis."
))

cells.append(nbf.v4.new_code_cell(
    "def get_table_schema(client, table_name):\n"
    "    query = f\"DESCRIBE TABLE {table_name}\"\n"
    "    schema_df = client.query_dataframe(query)\n"
    "    return schema_df\n\n"
    "def get_table_count(client, table_name):\n"
    "    query = f\"SELECT COUNT(*) as count FROM {table_name}\"\n"
    "    result = client.query_dataframe(query)\n"
    "    return result['count'].iloc[0]\n\n"
    "def get_sample_data(client, table_name, sample_size=1000):\n"
    "    query = f\"SELECT * FROM {table_name} LIMIT {sample_size}\"\n"
    "    return client.query_dataframe(query)\n\n"
    "print(\"Data exploration functions defined.\")"
))

cells.append(nbf.v4.new_markdown_cell("### 3.1 Explore Table Schema"))

cells.append(nbf.v4.new_code_cell(
    "schema = get_table_schema(ch, TABLE_NAME)\n"
    "print(\"Table Schema:\")\n"
    "print(schema)\n\n"
    "row_count = get_table_count(ch, TABLE_NAME)\n"
    "print(f\"\\nTotal rows in table: {row_count:,}\")"
))

cells.append(nbf.v4.new_markdown_cell("### 3.2 View Sample Data"))

cells.append(nbf.v4.new_code_cell(
    "sample_df = get_sample_data(ch, TABLE_NAME, sample_size=1000)\n"
    "print(f\"Sample data shape: {sample_df.shape}\")\n"
    "print(\"\\nFirst few rows:\")\n"
    "sample_df.head()"
))

cells.append(nbf.v4.new_code_cell(
    "print(\"Basic statistics for sample data:\")\n"
    "sample_df.describe()"
))

print("Adding column configuration...")

cells.append(nbf.v4.new_markdown_cell(
    "## 4. Column Selection & Configuration\n\n"
    "**Configure which columns to analyze for anomalies.**\n\n"
    "Based on the schema above, populate these lists:\n"
    "- **NUMERIC_COLUMNS**: Financial metrics to analyze (amounts, balances, rates, etc.)\n"
    "- **CATEGORICAL_COLUMNS**: Categories for grouping/filtering (bucket types, classifications, etc.)\n"
    "- **ID_COLUMNS**: Identifiers to track anomalies back to source (record IDs, transaction IDs, etc.)"
))

cells.append(nbf.v4.new_code_cell(
    "NUMERIC_COLUMNS = [\n"
    "    \n"
    "]\n\n"
    "CATEGORICAL_COLUMNS = [\n"
    "    \n"
    "]\n\n"
    "ID_COLUMNS = [\n"
    "    \n"
    "]\n\n"
    "BATCH_SIZE = 100000\n\n"
    "print(f\"Numeric columns for analysis: {NUMERIC_COLUMNS}\")\n"
    "print(f\"Categorical columns: {CATEGORICAL_COLUMNS}\")\n"
    "print(f\"Identifier columns: {ID_COLUMNS}\")\n"
    "print(f\"Batch size for processing: {BATCH_SIZE:,} rows\")"
))

print("Adding batch loading function...")

cells.append(nbf.v4.new_markdown_cell(
    "## 5. Scalable Data Loading\n\n"
    "This batch loading function handles millions of rows without memory issues."
))

batch_loader_code = """def load_data_in_batches(client, table_name, 
                        columns=None, batch_size=100000, 
                        max_rows=None, where_clause=None):
    if columns:
        col_str = ', '.join(columns)
    else:
        col_str = '*'
    
    where_sql = f"WHERE {where_clause}" if where_clause else ""
    
    total_query = f"SELECT COUNT(*) as count FROM {table_name} {where_sql}"
    total_rows = client.query_dataframe(total_query)['count'].iloc[0]
    
    if max_rows:
        total_rows = min(total_rows, max_rows)
    
    print(f"Loading {total_rows:,} rows in batches of {batch_size:,}...")
    
    offset = 0
    batch_num = 1
    
    while offset < total_rows:
        current_batch_size = min(batch_size, total_rows - offset)
        
        query = f\"\"\"
        SELECT {col_str}
        FROM {table_name}
        {where_sql}
        LIMIT {current_batch_size}
        OFFSET {offset}
        \"\"\"
        
        batch_df = client.query_dataframe(query)
        
        print(f"Batch {batch_num}: Loaded {len(batch_df):,} rows (offset: {offset:,})")
        
        yield batch_df
        
        offset += current_batch_size
        batch_num += 1

print("Batch loading function defined.")"""

cells.append(nbf.v4.new_code_cell(batch_loader_code))

print("Adding preprocessing function...")

cells.append(nbf.v4.new_markdown_cell(
    "## 6. Data Preprocessing\n\n"
    "Clean and normalize data before anomaly detection."
))

preprocess_code = """def preprocess_data(df, numeric_cols, handle_missing='drop', scaler_type='standard'):
    df_copy = df[numeric_cols].copy()
    
    df_copy.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    initial_rows = len(df_copy)
    
    if handle_missing == 'drop':
        df_copy.dropna(inplace=True)
    elif handle_missing == 'mean':
        df_copy.fillna(df_copy.mean(), inplace=True)
    elif handle_missing == 'median':
        df_copy.fillna(df_copy.median(), inplace=True)
    elif handle_missing == 'zero':
        df_copy.fillna(0, inplace=True)
    
    rows_after = len(df_copy)
    if initial_rows != rows_after:
        print(f"Removed {initial_rows - rows_after:,} rows with missing values")
    
    scaler = None
    if scaler_type == 'standard':
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_copy)
    elif scaler_type == 'robust':
        scaler = RobustScaler()
        scaled_data = scaler.fit_transform(df_copy)
    elif scaler_type == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df_copy)
    else:
        scaled_data = df_copy.values
    
    processed_df = pd.DataFrame(scaled_data, columns=numeric_cols, index=df_copy.index)
    
    return processed_df, scaler

print("Preprocessing function defined.")"""

cells.append(nbf.v4.new_code_cell(preprocess_code))

print("Adding anomaly detection methods...")

cells.append(nbf.v4.new_markdown_cell(
    "## 7. Anomaly Detection Methods\n\n"
    "### Method 1: Statistical Outlier Detection (Z-Score & IQR)\n\n"
    "**Fast, interpretable method for detecting extreme values.**\n\n"
    "- **Z-Score**: Measures how many standard deviations a point is from the mean\n"
    "- **IQR (Interquartile Range)**: Detects outliers based on quartile ranges\n\n"
    "**Best for**: Quick initial screening, univariate outliers"
))

statistical_code = """def detect_statistical_anomalies(df, numeric_cols, method='zscore', threshold=3):
    data = df[numeric_cols]
    
    if method == 'zscore':
        z_scores = np.abs((data - data.mean()) / data.std())
        is_anomaly = (z_scores > threshold).any(axis=1)
        
    elif method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        is_anomaly = ((data < lower_bound) | (data > upper_bound)).any(axis=1)
    
    return is_anomaly

print("Statistical anomaly detection function defined.")"""

cells.append(nbf.v4.new_code_cell(statistical_code))

cells.append(nbf.v4.new_markdown_cell(
    "### Method 2: Isolation Forest\n\n"
    "**Tree-based ensemble method that isolates anomalies efficiently.**\n\n"
    "Works by randomly partitioning data - anomalies require fewer splits to isolate.\n\n"
    "**Best for**: Multi-dimensional outliers, when you know approximate anomaly rate"
))

isolation_forest_code = """def detect_isolation_forest_anomalies(df, numeric_cols, contamination=0.01, 
                                     n_estimators=100, random_state=42):
    data = df[numeric_cols].values
    
    model = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    
    anomaly_labels = model.fit_predict(data)
    
    anomaly_scores = model.score_samples(data)
    
    return anomaly_labels, anomaly_scores, model

print("Isolation Forest function defined.")"""

cells.append(nbf.v4.new_code_cell(isolation_forest_code))

print("Adding autoencoder functions...")

cells.append(nbf.v4.new_markdown_cell(
    "### Method 3: Autoencoder (TensorFlow Deep Learning)\n\n"
    "**Neural network that learns to reconstruct normal data patterns.**\n\n"
    "The autoencoder compresses data to a lower-dimensional representation, then reconstructs it.\n"
    "Normal data reconstructs well; anomalies have high reconstruction error.\n\n"
    "**Architecture:**\n"
    "- Encoder: Compresses input to lower dimensional representation\n"
    "- Decoder: Reconstructs input from compressed representation\n\n"
    "**Best for**: Complex patterns, sufficient training data (10,000+ records)"
))

autoencoder_code = """def build_autoencoder(input_dim, encoding_dim=None, hidden_layers=[64, 32]):
    if encoding_dim is None:
        encoding_dim = max(input_dim // 2, 8)
    
    input_layer = layers.Input(shape=(input_dim,))
    
    encoded = input_layer
    for units in hidden_layers:
        encoded = layers.Dense(units, activation='relu')(encoded)
        encoded = layers.Dropout(0.2)(encoded)
    
    encoded = layers.Dense(encoding_dim, activation='relu', name='encoding')(encoded)
    
    decoded = encoded
    for units in reversed(hidden_layers):
        decoded = layers.Dense(units, activation='relu')(decoded)
        decoded = layers.Dropout(0.2)(decoded)
    
    decoded = layers.Dense(input_dim, activation='linear')(decoded)
    
    autoencoder = keras.Model(inputs=input_layer, outputs=decoded)
    
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return autoencoder

def train_autoencoder(model, X_train, epochs=50, batch_size=256, 
                     validation_split=0.1, verbose=1):
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )
    
    history = model.fit(
        X_train, X_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stopping, reduce_lr],
        verbose=verbose
    )
    
    return history

def detect_autoencoder_anomalies(model, X_test, threshold_percentile=95):
    reconstructions = model.predict(X_test, batch_size=256, verbose=0)
    
    reconstruction_errors = np.mean(np.square(X_test - reconstructions), axis=1)
    
    threshold = np.percentile(reconstruction_errors, threshold_percentile)
    
    anomaly_labels = reconstruction_errors > threshold
    
    return anomaly_labels, reconstruction_errors, threshold

print("Autoencoder functions defined.")"""

cells.append(nbf.v4.new_code_cell(autoencoder_code))

cells.append(nbf.v4.new_markdown_cell(
    "### Method 4: DBSCAN (Density-Based Clustering)\n\n"
    "**Identifies anomalies as points in low-density regions.**\n\n"
    "DBSCAN groups together points that are closely packed and marks isolated points as anomalies.\n\n"
    "**Best for**: When anomalies are isolated clusters or in low-density regions"
))

dbscan_code = """def detect_dbscan_anomalies(df, numeric_cols, eps=0.5, min_samples=5):
    data = df[numeric_cols].values
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    
    cluster_labels = dbscan.fit_predict(data)
    
    is_anomaly = cluster_labels == -1
    
    return cluster_labels, is_anomaly

print("DBSCAN function defined.")"""

cells.append(nbf.v4.new_code_cell(dbscan_code))

print("Adding visualization functions...")

cells.append(nbf.v4.new_markdown_cell(
    "## 8. Visualization Functions\n\n"
    "Functions to visualize and analyze anomaly detection results."
))

viz_code = """def plot_anomaly_distribution(anomaly_labels, scores=None, title="Anomaly Distribution"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    if isinstance(anomaly_labels[0], (bool, np.bool_)):
        anomaly_count = np.sum(anomaly_labels)
        normal_count = len(anomaly_labels) - anomaly_count
    else:
        anomaly_count = np.sum(anomaly_labels == -1)
        normal_count = np.sum(anomaly_labels == 1)
    
    axes[0].bar(['Normal', 'Anomaly'], [normal_count, anomaly_count], 
               color=['green', 'red'], alpha=0.7)
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'{title}\\nTotal: {len(anomaly_labels):,}')
    axes[0].set_yscale('log')
    
    for i, (label, count) in enumerate([('Normal', normal_count), ('Anomaly', anomaly_count)]):
        axes[0].text(i, count, f'{count:,}\\n({count/len(anomaly_labels)*100:.2f}%)', 
                    ha='center', va='bottom')
    
    if scores is not None:
        axes[1].hist(scores, bins=50, alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Anomaly Score')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Anomaly Scores')
        axes[1].axvline(np.percentile(scores, 95), color='red', 
                       linestyle='--', label='95th percentile')
        axes[1].legend()
    else:
        axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_pca_anomalies(df, numeric_cols, anomaly_labels, title="PCA Visualization"):
    data = df[numeric_cols].values
    
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data)
    
    if isinstance(anomaly_labels[0], (bool, np.bool_)):
        is_anomaly = anomaly_labels
    else:
        is_anomaly = anomaly_labels == -1
    
    plt.figure(figsize=(10, 8))
    
    plt.scatter(data_2d[~is_anomaly, 0], data_2d[~is_anomaly, 1], 
               c='green', alpha=0.3, label='Normal', s=10)
    
    plt.scatter(data_2d[is_anomaly, 0], data_2d[is_anomaly, 1], 
               c='red', alpha=0.7, label='Anomaly', s=50, edgecolors='black')
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Training History - Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history.history['mae'], label='Training MAE')
    axes[1].plot(history.history['val_mae'], label='Validation MAE')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('Training History - MAE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

print("Visualization functions defined.")"""

cells.append(nbf.v4.new_code_cell(viz_code))

print("Adding complete analysis pipeline examples...")

cells.append(nbf.v4.new_markdown_cell(
    "## 9. Complete Analysis Pipeline\n\n"
    "### 9.1 Load Data"
))

load_data_code = """all_columns = ID_COLUMNS + CATEGORICAL_COLUMNS + NUMERIC_COLUMNS

print(f"Loading data with columns: {all_columns}")

data_batches = []
for batch in load_data_in_batches(ch, TABLE_NAME, columns=all_columns, batch_size=BATCH_SIZE, max_rows=500000):
    data_batches.append(batch)

full_data = pd.concat(data_batches, ignore_index=True)
print(f"\\nTotal data loaded: {len(full_data):,} rows")
print(f"Memory usage: {full_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")"""

cells.append(nbf.v4.new_code_cell(load_data_code))

cells.append(nbf.v4.new_markdown_cell("### 9.2 Preprocess Data"))

preprocess_example = """processed_data, scaler = preprocess_data(
    full_data, 
    NUMERIC_COLUMNS, 
    handle_missing='drop',
    scaler_type='robust'
)

print(f"Processed data shape: {processed_data.shape}")"""

cells.append(nbf.v4.new_code_cell(preprocess_example))

cells.append(nbf.v4.new_markdown_cell("### 9.3 Statistical Methods"))

zscore_example = """zscore_anomalies = detect_statistical_anomalies(
    processed_data, 
    NUMERIC_COLUMNS, 
    method='zscore', 
    threshold=3
)

print(f"Z-score anomalies detected: {zscore_anomalies.sum():,} ({zscore_anomalies.sum()/len(zscore_anomalies)*100:.2f}%)")

plot_anomaly_distribution(zscore_anomalies, title="Z-Score Anomaly Detection")"""

cells.append(nbf.v4.new_code_cell(zscore_example))

iqr_example = """iqr_anomalies = detect_statistical_anomalies(
    processed_data, 
    NUMERIC_COLUMNS, 
    method='iqr', 
    threshold=1.5
)

print(f"IQR anomalies detected: {iqr_anomalies.sum():,} ({iqr_anomalies.sum()/len(iqr_anomalies)*100:.2f}%)")

plot_anomaly_distribution(iqr_anomalies, title="IQR Anomaly Detection")"""

cells.append(nbf.v4.new_code_cell(iqr_example))

cells.append(nbf.v4.new_markdown_cell("### 9.4 Isolation Forest"))

if_example = """if_labels, if_scores, if_model = detect_isolation_forest_anomalies(
    processed_data,
    NUMERIC_COLUMNS,
    contamination=0.01,
    n_estimators=100
)

print(f"Isolation Forest anomalies: {np.sum(if_labels == -1):,}")

plot_anomaly_distribution(if_labels, if_scores, title="Isolation Forest")"""

cells.append(nbf.v4.new_code_cell(if_example))

if_pca = """plot_pca_anomalies(processed_data, NUMERIC_COLUMNS, if_labels, 
                  title="Isolation Forest Anomalies (PCA Visualization)")"""

cells.append(nbf.v4.new_code_cell(if_pca))

cells.append(nbf.v4.new_markdown_cell("### 9.5 Autoencoder (Deep Learning)"))

ae_build = """input_dim = len(NUMERIC_COLUMNS)
print(f"Building autoencoder with input dimension: {input_dim}")

autoencoder = build_autoencoder(
    input_dim=input_dim,
    encoding_dim=max(input_dim // 2, 8),
    hidden_layers=[64, 32]
)

autoencoder.summary()"""

cells.append(nbf.v4.new_code_cell(ae_build))

ae_train = """X_train = processed_data.values

history = train_autoencoder(
    autoencoder,
    X_train,
    epochs=50,
    batch_size=256,
    validation_split=0.1,
    verbose=1
)

plot_training_history(history)"""

cells.append(nbf.v4.new_code_cell(ae_train))

ae_detect = """ae_anomalies, ae_errors, ae_threshold = detect_autoencoder_anomalies(
    autoencoder,
    X_train,
    threshold_percentile=95
)

print(f"Autoencoder anomalies: {ae_anomalies.sum():,} ({ae_anomalies.sum()/len(ae_anomalies)*100:.2f}%)")
print(f"Reconstruction error threshold: {ae_threshold:.6f}")

plot_anomaly_distribution(ae_anomalies, ae_errors, title="Autoencoder Anomaly Detection")"""

cells.append(nbf.v4.new_code_cell(ae_detect))

ae_pca = """plot_pca_anomalies(processed_data, NUMERIC_COLUMNS, ae_anomalies,
                  title="Autoencoder Anomalies (PCA Visualization)")"""

cells.append(nbf.v4.new_code_cell(ae_pca))

print("Adding result export and analysis...")

cells.append(nbf.v4.new_markdown_cell(
    "## 10. Combine Results & Export\n\n"
    "Combine results from all methods and export flagged anomalies."
))

export_code = """results_df = full_data.loc[processed_data.index].copy()

results_df['zscore_anomaly'] = zscore_anomalies.values
results_df['iqr_anomaly'] = iqr_anomalies.values
results_df['isolation_forest_anomaly'] = if_labels == -1
results_df['isolation_forest_score'] = if_scores
results_df['autoencoder_anomaly'] = ae_anomalies
results_df['autoencoder_error'] = ae_errors

results_df['anomaly_count'] = (
    results_df['zscore_anomaly'].astype(int) +
    results_df['iqr_anomaly'].astype(int) +
    results_df['isolation_forest_anomaly'].astype(int) +
    results_df['autoencoder_anomaly'].astype(int)
)

results_df['is_anomaly'] = results_df['anomaly_count'] >= 2

print(f"\\nAnomalies flagged by multiple methods:")
print(results_df['anomaly_count'].value_counts().sort_index())

anomalies_only = results_df[results_df['is_anomaly']].copy()
print(f"\\nTotal anomalies (2+ methods): {len(anomalies_only):,}")

anomalies_only.to_csv('anomalies_detected.csv', index=False)
print("\\nAnomalies saved to: anomalies_detected.csv")

anomalies_only.head(20)"""

cells.append(nbf.v4.new_code_cell(export_code))

cells.append(nbf.v4.new_markdown_cell("## 11. Analyze Top Anomalies"))

top_anomalies_code = """top_anomalies = anomalies_only.nlargest(20, 'anomaly_count')

print("Top 20 Most Severe Anomalies:")
print("="*80)
for idx, row in top_anomalies.iterrows():
    print(f"\\nRecord: {row[ID_COLUMNS].to_dict() if ID_COLUMNS else idx}")
    print(f"Flagged by {row['anomaly_count']}/4 methods")
    print(f"Isolation Forest Score: {row['isolation_forest_score']:.4f}")
    print(f"Autoencoder Error: {row['autoencoder_error']:.4f}")
    if CATEGORICAL_COLUMNS:
        print(f"Categories: {row[CATEGORICAL_COLUMNS].to_dict()}")
    print(f"Values: {row[NUMERIC_COLUMNS].to_dict()}")
    print("-"*80)"""

cells.append(nbf.v4.new_code_cell(top_anomalies_code))

cells.append(nbf.v4.new_markdown_cell("## 12. Save Models for Future Use"))

save_models = """import joblib

joblib.dump(scaler, 'scaler.pkl')
print("Scaler saved to: scaler.pkl")

joblib.dump(if_model, 'isolation_forest_model.pkl')
print("Isolation Forest model saved to: isolation_forest_model.pkl")

autoencoder.save('autoencoder_model.h5')
print("Autoencoder saved to: autoencoder_model.h5")

print("\\nAll models saved successfully!")"""

cells.append(nbf.v4.new_code_cell(save_models))

cells.append(nbf.v4.new_markdown_cell(
    "## 13. Summary and Recommendations\n\n"
    "### Method Comparison\n\n"
    "| Method | Speed | Interpretability | Best Use Case |\n"
    "|--------|-------|-----------------|---------------|\n"
    "| **Z-Score/IQR** | Very Fast | High | Quick screening, univariate outliers |\n"
    "| **Isolation Forest** | Fast | Medium | Multi-dimensional outliers |\n"
    "| **Autoencoder** | Moderate | Low | Complex patterns, sufficient data |\n"
    "| **DBSCAN** | Slow | Medium | Low-density region anomalies |\n\n"
    "### Recommendations for FR 2052a Compliance\n\n"
    "1. **Multi-Method Approach**: Use consensus from multiple methods\n"
    "   - Flag records detected by 2+ methods as high priority\n"
    "   - Records flagged by all methods warrant immediate investigation\n\n"
    "2. **Threshold Tuning**:\n"
    "   - Start with conservative thresholds (Z-score=3, contamination=0.01)\n"
    "   - Adjust based on investigation capacity and false positive rate\n\n"
    "3. **Regular Retraining**:\n"
    "   - Retrain models monthly or quarterly as patterns evolve\n"
    "   - Track model performance over time\n\n"
    "4. **Domain Expert Review**:\n"
    "   - Always involve subject matter experts for final validation\n"
    "   - Document investigation outcomes to improve future detection\n\n"
    "5. **Workflow Integration**:\n"
    "   - Automate anomaly detection in your reporting pipeline\n"
    "   - Create alerts for new anomalies\n"
    "   - Track metrics: detection rate, false positives, investigation time\n\n"
    "### Parameter Tuning Guide\n\n"
    "**Statistical Methods**:\n"
    "- Z-score threshold: 2 (aggressive) to 4 (conservative)\n"
    "- IQR multiplier: 1.5 (standard) to 3 (conservative)\n\n"
    "**Isolation Forest**:\n"
    "- Contamination: Set to expected anomaly rate (typically 0.01-0.05)\n"
    "- N_estimators: 100-200 (more = slower but more stable)\n\n"
    "**Autoencoder**:\n"
    "- Threshold percentile: 90-99 (higher = fewer anomalies)\n"
    "- Hidden layers: Adjust based on data complexity\n"
    "- Epochs: 20-100 (use early stopping)\n\n"
    "**DBSCAN**:\n"
    "- eps: Requires experimentation (try 0.3-1.0)\n"
    "- min_samples: 5-10 for typical datasets\n\n"
    "### Next Steps\n\n"
    "1. ✅ Fill in NUMERIC_COLUMNS, CATEGORICAL_COLUMNS, ID_COLUMNS\n"
    "2. ✅ Run data exploration cells to understand your data\n"
    "3. ✅ Execute the complete pipeline on a sample (500K rows)\n"
    "4. ✅ Review detected anomalies with compliance team\n"
    "5. ✅ Tune thresholds based on feedback\n"
    "6. ✅ Scale up to full dataset\n"
    "7. ✅ Integrate into regular workflow\n"
    "8. ✅ Monitor and maintain over time"
))

print(f"\nWriting notebook with {len(cells)} cells...")
nb['cells'] = cells

with open('anomaly_detection_fr2052a.ipynb', 'w') as f:
    nbf.write(nb, f)

print(f"\n✓ Complete notebook created successfully!")
print(f"✓ Total cells: {len(cells)}")
print(f"✓ File: anomaly_detection_fr2052a.ipynb")
