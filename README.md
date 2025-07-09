# dess-smard-preprocessor
# SMARD Data Preprocessing for Reinforcement Learning

## Overview

This preprocessing pipeline converts raw German energy market data (SMARD) into RL-ready format for Distributed Energy Storage Systems (DESS) optimization. The pipeline handles German CSV formatting, feature engineering, and state space creation for reinforcement learning applications.

## Data Source

- **Input**: `energie_zusammengefasst_mit_aufloesung.csv` (Retrieved from SMARD database)
- **Format**: German CSV (semicolon separated, comma decimal)
- **Content**: Hourly energy market data including wind, solar, load, and prices

## Requirements

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Quick Start

```python
from preprocess_smard import run_preprocessing_pipeline

# Process SMARD data
results = run_preprocessing_pipeline(
    csv_path="energie_zusammengefasst_mit_aufloesung.csv",
    output_dir="./preprocessed_data/",
    n_state_bins=10,
    scaler_type='robust',
    export_format='all'
)

# Load processed data
import pandas as pd
rl_dataset = pd.read_pickle(results['exported_files'][0])
```

## Pipeline Features

### Data Processing
-  German CSV format handling (`;` separator, `,` decimal)
-  Missing value imputation and outlier removal
-  Data validation and quality checks

### Feature Engineering
- **Core Features**: Wind, Solar, Load, Price data
- **Market Features**: Negative prices, renewable surplus, golden hours
- **Temporal Features**: Hour, day, month with cyclical encoding
- **Rolling Windows**: 3h, 6h, 12h, 24h patterns
- **Lag/Lead Features**: Historical and future value patterns

### RL Optimization
- **State Space**: Discretized states for Q-learning
- **Feature Scaling**: Robust scaling for neural networks
- **Multiple Feature Sets**: Minimal, standard, and full feature sets

## Output Files

```
preprocessed_data/
├── smard_processed_YYYYMMDD_HHMMSS.pkl     # Complete RL dataset
├── smard_processed_YYYYMMDD_HHMMSS.csv     # Raw processed data
├── smard_processed_YYYYMMDD_HHMMSS_config.json  # Configuration
├── smard_processed_YYYYMMDD_HHMMSS_minimal_features.csv
├── smard_processed_YYYYMMDD_HHMMSS_standard_features.csv
├── smard_processed_YYYYMMDD_HHMMSS_full_features.csv
└── eda_report_YYYYMMDD_HHMMSS.png          # Data visualization
```

## Key Metrics

The pipeline generates comprehensive statistics including:
- **Golden Hours**: Opportunities with negative prices + renewable surplus
- **Data Quality Score**: Completeness, consistency, validity (0-100)
- **State Space Size**: Total unique states for RL
- **Feature Completeness**: Percentage of non-missing values

## Usage Examples

### Basic Processing
```python
# Standard preprocessing
results = run_preprocessing_pipeline("energie_zusammengefasst_mit_aufloesung.csv")
```

### Custom Configuration
```python
# Custom settings
results = run_preprocessing_pipeline(
    csv_path="energie_zusammengefasst_mit_aufloesung.csv",
    n_state_bins=20,           # More granular states
    scaler_type='standard',    # Standard scaling
    export_format='pickle'     # Pickle only
)
```

### Accessing Results
```python
# Get processed dataset
rl_data = results['rl_dataset']

# Check statistics
stats = rl_data['statistics']
print(f"Golden hours found: {stats['opportunity_analysis']['golden_hours']}")
print(f"Data quality: {stats['rl_readiness']['data_quality_score']}/100")

# Get feature sets
features = rl_data['feature_sets']
minimal_features = features['minimal']    # 11 core features
standard_features = features['standard']  # 20+ features
full_features = features['full']          # 100+ features
```

## Configuration Options

| Parameter | Options | Description |
|-----------|---------|-------------|
| `n_state_bins` | `5-20` | Granularity of state discretization |
| `scaler_type` | `'standard'`, `'minmax'`, `'robust'` | Feature scaling method |
| `export_format` | `'pickle'`, `'csv'`, `'json'`, `'all'` | Output format |

## Data Schema

### Input Columns (German)
- `Datum von` / `Datum bis` → Datetime range
- `Wind Onshore [MWh]` → Wind generation
- `Photovoltaik [MWh]` → Solar generation
- `Netzlast [MWh]` → Grid load
- `Deutschland/Luxemburg [€/MWh]` → Electricity price

### Output Features
- **States**: `price_state`, `renewable_surplus_state`, `load_state`
- **Actions**: Market participation decisions
- **Rewards**: Based on price arbitrage and renewable utilization

## Troubleshooting

### Common Issues
1. **German CSV Format**: Ensure semicolon separator and comma decimal
2. **Missing Data**: Pipeline handles gaps with interpolation
3. **Memory Usage**: Large datasets may require chunked processing

### Data Quality
- Pipeline removes rows with missing critical data (`datetime`, `load`, `price`)
- Applies sanity checks (load > 0, price ∈ [-1000, 3000])
- Interpolates missing renewable generation data

## Technical Details

- **Language**: Python 3.8+
- **Memory**: ~500MB for 1 year of hourly data
- **Processing Time**: ~30 seconds for 8760 hours
- **Output Size**: ~50MB for full feature set

## Contact

**Author**: Energy Systems AI Team  
**Email**: dalim@rptu.de  
**Institution**: RPTU Kaiserslautern-Landau  
**Date**: 2025

---

*This preprocessing pipeline is optimized for DESS-RL applications and German energy market characteristics.*
