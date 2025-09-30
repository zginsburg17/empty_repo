# Anomaly Detection for FR 2052a Regulatory Reporting

A comprehensive, scalable anomaly detection framework for ClickHouse financial data using multiple machine learning approaches.

## Overview

This framework provides flexible anomaly detection for the `tb_elp_dep_dl` table in the `dcw_glrs_clean` database. It's designed to identify:
- **Outliers**: Extreme values in financial metrics
- **Miscategorizations**: Data points that don't fit expected patterns for their category

## Features

- **Multiple Detection Methods**: Statistical, Isolation Forest, Autoencoder (TensorFlow), and DBSCAN
- **Scalable**: Handles millions of rows through efficient batch processing
- **Flexible Column Selection**: Choose which columns to analyze
- **Memory Efficient**: Optimized for large datasets
- **Well Documented**: Extensive comments and explanations throughout
- **Jupyter Ready**: Designed for interactive exploration and analysis

## Prerequisites

### Python Environment
```bash
pip install clickhouse-driver pandas numpy scikit-learn tensorflow matplotlib seaborn joblib
```

### Required Packages
- `clickhouse-driver` - Database connectivity
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `scikit-learn` - Machine learning algorithms (Isolation Forest, DBSCAN, preprocessing)
- `tensorflow` - Deep learning (Autoencoder)
- `matplotlib` & `seaborn` - Visualization
- `joblib` - Model persistence

## Quick Start

1. **Configure Database Connection**
   ```python
   ch = Client(
       host='your-host',
       port=9000,
       database='dcw_glrs_clean',
       user='YOUR_USERNAME',
       password='YOUR_PASSWORD',
       # ... other settings
   )
   ```

2. **Explore Your Data**
   ```python
   # View table schema
   schema = get_table_schema(ch)
   
   # Get row count
   row_count = get_table_count(ch)
   
   # Sample data
   sample = get_sample_data(ch, sample_size=1000)
   ```

3. **Select Columns for Analysis**
   ```python
   NUMERIC_COLUMNS = ['amount', 'balance', 'rate', ...]
   CATEGORICAL_COLUMNS = ['category', 'type', ...]
   ID_COLUMNS = ['record_id', 'transaction_id', ...]
   ```

4. **Run Anomaly Detection**
   - Statistical Methods (fastest)
   - Isolation Forest (good balance)
   - Autoencoder (most sophisticated)
   - DBSCAN (density-based)

## Detection Methods

### 1. Statistical Methods
**Z-Score and IQR-based outlier detection**

- **Pros**: Fast, interpretable, no training required
- **Cons**: Assumes normal distribution
- **Use Case**: Quick screening, univariate outliers
- **Performance**: ~1 second for 100K rows

```python
anomalies = detect_statistical_anomalies(
    data, 
    columns, 
    method='zscore', 
    threshold=3
)
```

### 2. Isolation Forest
**Tree-based ensemble anomaly detection**

- **Pros**: Fast, handles multi-dimensional data, no distribution assumptions
- **Cons**: Requires setting contamination parameter
- **Use Case**: Multi-dimensional outliers
- **Performance**: ~5 seconds for 100K rows

```python
labels, scores, model = detect_isolation_forest_anomalies(
    data,
    columns,
    contamination=0.01,
    n_estimators=100
)
```

### 3. Autoencoder (TensorFlow)
**Deep learning reconstruction-based detection**

- **Pros**: Learns complex patterns, flexible architecture
- **Cons**: Requires training time, needs tuning
- **Use Case**: Complex patterns, sufficient data
- **Performance**: ~2-5 minutes training for 100K rows

```python
# Build and train
model = build_autoencoder(input_dim=len(columns))
history = train_autoencoder(model, X_train, epochs=50)

# Detect
anomalies, errors, threshold = detect_autoencoder_anomalies(
    model, 
    X_test,
    threshold_percentile=95
)
```

### 4. DBSCAN
**Density-based clustering**

- **Pros**: Finds isolated clusters, no cluster count needed
- **Cons**: Sensitive to parameters, computationally intensive
- **Use Case**: Low-density region anomalies
- **Performance**: ~10-30 seconds for 100K rows

```python
labels, is_anomaly = detect_dbscan_anomalies(
    data,
    columns,
    eps=0.5,
    min_samples=5
)
```

## Batch Processing for Large Datasets

The framework efficiently handles millions of rows through batch processing:

```python
# Load data in batches
for batch in load_data_in_batches(
    ch, 
    columns=all_columns,
    batch_size=100000,
    max_rows=1000000
):
    # Process each batch
    process_batch(batch)
```

## Workflow Recommendations

### For FR 2052a Compliance

1. **Initial Screening**
   - Use statistical methods for quick univariate checks
   - Identify obvious outliers in key metrics

2. **Multi-Dimensional Analysis**
   - Apply Isolation Forest for cross-column pattern detection
   - Useful for catching miscategorizations

3. **Deep Learning Validation**
   - Train autoencoder on historical "good" data
   - Flag records with high reconstruction errors

4. **Consensus Approach**
   - Combine results from multiple methods
   - Flag records detected by 2+ methods for review

5. **Expert Review**
   - Always involve domain experts for final validation
   - Track false positives to improve thresholds

## Output and Reporting

The notebook generates:
- **CSV Export**: `anomalies_detected.csv` with all flagged records
- **Visualizations**: Distribution plots, PCA projections, training curves
- **Saved Models**: For future use on new data
  - `scaler.pkl` - Data preprocessing scaler
  - `isolation_forest_model.pkl` - Trained Isolation Forest
  - `autoencoder_model.h5` - Trained Autoencoder

## Example Results Structure

```python
results_df = {
    'record_id': [...],
    'zscore_anomaly': [True/False],
    'iqr_anomaly': [True/False],
    'isolation_forest_anomaly': [True/False],
    'isolation_forest_score': [...],
    'autoencoder_anomaly': [True/False],
    'autoencoder_error': [...],
    'anomaly_count': [0-4],  # Number of methods flagging this record
    'is_anomaly': [True/False]  # Flagged by 2+ methods
}
```

## Best Practices

### Parameter Tuning
- **Statistical Methods**: Adjust z-score threshold (2-4) or IQR multiplier (1.5-3)
- **Isolation Forest**: Set contamination based on expected anomaly rate
- **Autoencoder**: Tune architecture and threshold percentile (90-99)
- **DBSCAN**: Experiment with eps and min_samples based on data density

### Performance Optimization
- Use batch processing for datasets > 1M rows
- Consider sampling for initial model training
- Use `n_jobs=-1` for parallel processing where available
- Monitor memory usage with large datasets

### Model Maintenance
- Retrain models periodically (monthly/quarterly)
- Track detection rates and false positives
- Adjust thresholds based on investigation capacity
- Document all parameter changes and rationale

## Troubleshooting

### Common Issues

**Memory Errors**
- Reduce batch size
- Process data in chunks
- Sample data for initial exploration

**Slow Performance**
- Enable parallel processing (`n_jobs=-1`)
- Reduce sample size for quick tests
- Use statistical methods for initial screening

**Too Many/Few Anomalies**
- Adjust detection thresholds
- Check data preprocessing
- Verify column selection is appropriate

**Model Not Training**
- Check for NaN/infinite values
- Ensure sufficient data for training
- Verify column data types are numeric

## License

This framework is provided as-is for internal use in regulatory compliance workflows.

## Support

For questions or issues:
1. Review the detailed comments in the notebook
2. Check the troubleshooting section
3. Consult with your data science team
4. Review scikit-learn and TensorFlow documentation for specific algorithms
