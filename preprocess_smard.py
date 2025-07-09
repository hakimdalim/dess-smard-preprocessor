"""
SMARD Data Preprocessing Pipeline for DESS-RL
============================================
Specialized preprocessing pipeline for German energy market data (SMARD)
Converts raw CSV data to RL-ready format for DESS optimization

Author: Energy Systems AI Team
Date: 2025
"""

import pandas as pd
import numpy as np
import json
import datetime
import os
from typing import Dict, List, Any, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class SMARDDataPreprocessor:
    """
    Comprehensive preprocessing pipeline for SMARD energy data
    Optimized for DESS-RL training and analysis
    """
    
    def __init__(self, csv_path: str, output_dir: str = "./preprocessed_data/"):
        """
        Initialize preprocessor
        
        Args:
            csv_path: Path to SMARD CSV file
            output_dir: Directory for output files
        """
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.raw_data = None
        self.processed_data = None
        self.scalers = {}
        self.data_stats = {}
        self.rl_config = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"🔧 SMARD Data Preprocessor initialized")
        print(f"📂 Input: {csv_path}")
        print(f"📁 Output: {output_dir}")
    
    def load_raw_data(self) -> pd.DataFrame:
        """
        Load raw SMARD CSV data with German formatting
        """
        print("\n📊 Loading raw SMARD data...")
        
        try:
            # Read CSV with German format settings
            self.raw_data = pd.read_csv(
                self.csv_path,
                sep=';',           # German CSV uses semicolon
                decimal=',',       # German decimal separator
                encoding='utf-8',
                thousands='.',     # German thousands separator
                na_values=['', ' ', 'null', 'NULL', 'NaN']
            )
            
            print(f"✅ Raw data loaded successfully")
            print(f"   📏 Shape: {self.raw_data.shape}")
            print(f"   📋 Columns: {list(self.raw_data.columns)}")
            print(f"   📅 Date range: {self.raw_data.iloc[0, 0]} to {self.raw_data.iloc[-1, 1]}")
            
            return self.raw_data
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise
    
    def clean_and_validate(self) -> pd.DataFrame:
        """
        Clean and validate raw data
        """
        print("\n🧹 Cleaning and validating data...")
        
        if self.raw_data is None:
            raise ValueError("Raw data not loaded. Call load_raw_data() first.")
        
        df = self.raw_data.copy()
        
        # Standardize column names
        column_mapping = {
            'Datum von': 'datetime_from',
            'Datum bis': 'datetime_to',
            'Wind Onshore [MWh] Berechnete Auflösungen': 'wind_mwh',
            'Photovoltaik [MWh] Berechnete Auflösungen': 'pv_mwh',
            'Erdgas [MWh] Berechnete Auflösungen': 'gas_mwh',
            'Netzlast [MWh] Berechnete Auflösungen': 'load_mwh',
            'Deutschland/Luxemburg [€/MWh] Originalauflösungen': 'price_eur_mwh'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Convert datetime columns
        df['datetime_from'] = pd.to_datetime(df['datetime_from'], format='%d.%m.%Y %H:%M')
        df['datetime_to'] = pd.to_datetime(df['datetime_to'], format='%d.%m.%Y %H:%M')
        
        # Handle numeric columns with German formatting
        numeric_columns = ['wind_mwh', 'pv_mwh', 'gas_mwh', 'load_mwh', 'price_eur_mwh']
        
        for col in numeric_columns:
            if col in df.columns:
                # Convert to numeric, handling German decimal format
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Data validation and cleaning
        initial_rows = len(df)
        
        # Remove rows with missing critical data
        critical_columns = ['datetime_from', 'load_mwh', 'price_eur_mwh']
        df = df.dropna(subset=critical_columns)
        
        # Remove unrealistic values
        df = df[df['load_mwh'] > 0]  # Load must be positive
        df = df[df['price_eur_mwh'] > -1000]  # Price sanity check
        df = df[df['price_eur_mwh'] < 3000]   # Price sanity check
        
        # Fill missing renewable data with 0 (assuming no generation)
        renewable_columns = ['wind_mwh', 'pv_mwh']
        for col in renewable_columns:
            if col in df.columns:
                df[col] = df[col].fillna(0)
                df[col] = df[col].clip(lower=0)  # Ensure non-negative
        
        # Fill missing gas data using interpolation - FIXED
        if 'gas_mwh' in df.columns:
            df['gas_mwh'] = df['gas_mwh'].interpolate(method='linear').fillna(0)
            df['gas_mwh'] = df['gas_mwh'].clip(lower=0)
        
        # Reset index
        df = df.reset_index(drop=True)
        
        print(f"✅ Data cleaning completed")
        print(f"   📉 Rows removed: {initial_rows - len(df)} ({(initial_rows - len(df))/initial_rows*100:.1f}%)")
        print(f"   📊 Clean data shape: {df.shape}")
        
        self.raw_data = df
        return df
    
    def create_features(self) -> pd.DataFrame:
        """
        Create engineered features for RL model
        """
        print("\n⚙️ Creating engineered features...")
        
        df = self.raw_data.copy()
        
        # Basic renewable energy features
        df['total_renewable_mwh'] = df['wind_mwh'] + df['pv_mwh']
        df['renewable_surplus_mwh'] = df['total_renewable_mwh'] - df['load_mwh']
        df['renewable_ratio'] = df['total_renewable_mwh'] / df['load_mwh']
        
        # Market opportunity features
        df['is_negative_price'] = (df['price_eur_mwh'] < 0).astype(int)
        df['is_surplus_renewable'] = (df['renewable_surplus_mwh'] > 0).astype(int)
        df['golden_hour'] = (df['is_negative_price'] & df['is_surplus_renewable']).astype(int)
        
        # Price features
        df['price_abs'] = df['price_eur_mwh'].abs()
        df['price_log'] = np.log1p(df['price_abs'])
        
        # Rolling window features (for temporal patterns)
        window_sizes = [3, 6, 12, 24]  # 3h, 6h, 12h, 24h windows
        
        for window in window_sizes:
            # Price patterns
            df[f'price_rolling_mean_{window}h'] = df['price_eur_mwh'].rolling(window=window, center=True).mean()
            df[f'price_rolling_std_{window}h'] = df['price_eur_mwh'].rolling(window=window, center=True).std()
            
            # Renewable patterns
            df[f'renewable_rolling_mean_{window}h'] = df['total_renewable_mwh'].rolling(window=window, center=True).mean()
            df[f'load_rolling_mean_{window}h'] = df['load_mwh'].rolling(window=window, center=True).mean()
        
        # Temporal features
        df['hour'] = df['datetime_from'].dt.hour
        df['day_of_week'] = df['datetime_from'].dt.dayofweek
        df['month'] = df['datetime_from'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Cyclical encoding for temporal features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Market efficiency features
        df['renewable_efficiency'] = df['total_renewable_mwh'] / (df['total_renewable_mwh'] + df['gas_mwh'] + 1e-6)
        df['price_volatility_6h'] = df['price_eur_mwh'].rolling(window=6).std()
        
        # Lag features (previous values)
        lag_features = ['price_eur_mwh', 'total_renewable_mwh', 'load_mwh']
        for feature in lag_features:
            for lag in [1, 2, 3, 6, 12, 24]:
                df[f'{feature}_lag_{lag}h'] = df[feature].shift(lag)
        
        # Forward features (next values) - for learning future patterns
        for feature in lag_features:
            for lead in [1, 2, 3, 6]:
                df[f'{feature}_lead_{lead}h'] = df[feature].shift(-lead)
        
        print(f"✅ Feature engineering completed")
        print(f"   📊 Total features: {df.shape[1]}")
        print(f"   🆕 New features: {df.shape[1] - len(self.raw_data.columns)}")
        
        self.processed_data = df
        return df
    
    def handle_missing_values(self) -> pd.DataFrame:
        """
        Handle missing values in processed data
        """
        print("\n🔍 Handling missing values...")
        
        df = self.processed_data.copy()
        
        # Check missing values
        missing_info = df.isnull().sum()
        missing_percent = (missing_info / len(df) * 100).round(2)
        
        print("Missing values analysis:")
        for col, count in missing_info[missing_info > 0].items():
            print(f"   {col}: {count} ({missing_percent[col]}%)")
        
        # Strategy for different feature types
        
        # 1. Rolling window features - forward/backward fill - FIXED
        rolling_columns = [col for col in df.columns if 'rolling' in col]
        for col in rolling_columns:
            df[col] = df[col].bfill().ffill()
        
        # 2. Lag features - forward fill (use previous known value) - FIXED
        lag_columns = [col for col in df.columns if 'lag_' in col]
        for col in lag_columns:
            df[col] = df[col].ffill()
        
        # 3. Lead features - backward fill (use next known value) - FIXED
        lead_columns = [col for col in df.columns if 'lead_' in col]
        for col in lead_columns:
            df[col] = df[col].bfill()
        
        # 4. For any remaining missing values, use median imputation
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        imputer = SimpleImputer(strategy='median')
        
        for col in numeric_columns:
            if df[col].isnull().sum() > 0:
                df[col] = imputer.fit_transform(df[[col]]).ravel()
        
        # Final check
        remaining_missing = df.isnull().sum().sum()
        print(f"✅ Missing value handling completed")
        print(f"   🔢 Remaining missing values: {remaining_missing}")
        
        self.processed_data = df
        return df
    
    def create_rl_states(self, n_state_bins: int = 10) -> pd.DataFrame:
        """
        Create discretized state representations for RL
        """
        print(f"\n🎯 Creating RL state representations (bins: {n_state_bins})...")
        
        df = self.processed_data.copy()
        
        # Define key state variables for RL
        state_variables = {
            'price_state': 'price_eur_mwh',
            'renewable_surplus_state': 'renewable_surplus_mwh',
            'load_state': 'load_mwh',
            'renewable_ratio_state': 'renewable_ratio',
            'hour_state': 'hour'
        }
        
        # Create discrete states
        for state_name, feature in state_variables.items():
            if feature in df.columns:
                if feature == 'hour':
                    # Hour is naturally discrete (0-23)
                    df[state_name] = df[feature]
                else:
                    # Use quantile-based binning for other features
                    df[state_name] = pd.qcut(
                        df[feature], 
                        q=n_state_bins, 
                        labels=range(n_state_bins),
                        duplicates='drop'
                    ).astype(int)
        
        # Create combined state representation
        state_cols = list(state_variables.keys())
        df['combined_state'] = df[state_cols].apply(
            lambda row: '_'.join(row.astype(str)), axis=1
        )
        
        # Create state ID (integer representation)
        unique_states = df['combined_state'].unique()
        state_mapping = {state: idx for idx, state in enumerate(unique_states)}
        df['state_id'] = df['combined_state'].map(state_mapping)
        
        print(f"✅ RL states created")
        print(f"   🔢 Total unique states: {len(unique_states)}")
        print(f"   📊 State variables: {list(state_variables.keys())}")
        
        # Store state configuration
        self.rl_config = {
            'n_state_bins': n_state_bins,
            'state_variables': state_variables,
            'state_mapping': state_mapping,
            'n_unique_states': len(unique_states)
        }
        
        self.processed_data = df
        return df
    
    def scale_features(self, scaler_type: str = 'robust') -> pd.DataFrame:
        """
        Scale numerical features for RL training
        
        Args:
            scaler_type: 'standard', 'minmax', or 'robust'
        """
        print(f"\n📏 Scaling features using {scaler_type} scaler...")
        
        df = self.processed_data.copy()
        
        # Select scalers
        scaler_map = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        
        if scaler_type not in scaler_map:
            raise ValueError(f"Scaler type {scaler_type} not supported. Use: {list(scaler_map.keys())}")
        
        # Features to scale (exclude discrete states and datetime)
        exclude_columns = [
            'datetime_from', 'datetime_to', 'combined_state', 'state_id',
            'is_negative_price', 'is_surplus_renewable', 'golden_hour', 'is_weekend'
        ] + [col for col in df.columns if 'state' in col]
        
        scale_columns = [col for col in df.columns if col not in exclude_columns and df[col].dtype in ['float64', 'int64']]
        
        # Apply scaling
        scaler = scaler_map[scaler_type]
        
        if len(scale_columns) > 0:
            df[scale_columns] = scaler.fit_transform(df[scale_columns])
            self.scalers[scaler_type] = scaler
            
            print(f"✅ Scaling completed")
            print(f"   📊 Scaled features: {len(scale_columns)}")
        else:
            print("⚠️ No features to scale")
        
        self.processed_data = df
        return df
    
    def create_rl_dataset(self) -> Dict[str, Any]:
        """
        Create final dataset ready for RL training
        """
        print("\n🤖 Creating RL-ready dataset...")
        
        df = self.processed_data.copy()
        
        # Define feature groups
        feature_groups = {
            'core_features': [
                'wind_mwh', 'pv_mwh', 'load_mwh', 'price_eur_mwh',
                'total_renewable_mwh', 'renewable_surplus_mwh', 'renewable_ratio'
            ],
            'market_features': [
                'is_negative_price', 'is_surplus_renewable', 'golden_hour',
                'price_abs', 'price_log'
            ],
            'temporal_features': [
                'hour', 'day_of_week', 'month', 'is_weekend',
                'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos'
            ],
            'rolling_features': [col for col in df.columns if 'rolling' in col],
            'lag_features': [col for col in df.columns if 'lag_' in col],
            'state_features': [col for col in df.columns if 'state' in col]
        }
        
        # Create different feature sets for different RL approaches
        feature_sets = {
            'minimal': feature_groups['core_features'] + feature_groups['temporal_features'][:4],
            'standard': (feature_groups['core_features'] + 
                        feature_groups['market_features'] + 
                        feature_groups['temporal_features']),
            'full': [col for group in feature_groups.values() for col in group if col in df.columns]
        }
        
        # RL-specific data structures
        rl_dataset = {
            'data': df,
            'feature_groups': feature_groups,
            'feature_sets': feature_sets,
            'rl_config': self.rl_config,
            'scalers': self.scalers,
            'statistics': self._calculate_statistics()
        }
        
        print(f"✅ RL dataset created")
        print(f"   📊 Total samples: {len(df)}")
        print(f"   🎯 Feature sets available: {list(feature_sets.keys())}")
        print(f"   📈 Features per set:")
        for name, features in feature_sets.items():
            print(f"      {name}: {len(features)} features")
        
        return rl_dataset
    
    def _calculate_statistics(self) -> Dict[str, Any]:
        """
        Calculate comprehensive data statistics
        """
        df = self.processed_data
        
        stats = {
            'data_overview': {
                'total_hours': len(df),
                'date_range': {
                    'start': df['datetime_from'].min().isoformat(),
                    'end': df['datetime_to'].max().isoformat()
                },
                'missing_data_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            },
            'price_analysis': {
                'min_price': df['price_eur_mwh'].min(),
                'max_price': df['price_eur_mwh'].max(),
                'mean_price': df['price_eur_mwh'].mean(),
                'negative_price_hours': int(df['is_negative_price'].sum()),
                'negative_price_percentage': float(df['is_negative_price'].mean() * 100)
            },
            'renewable_analysis': {
                'avg_wind': df['wind_mwh'].mean(),
                'avg_pv': df['pv_mwh'].mean(),
                'surplus_hours': int(df['is_surplus_renewable'].sum()),
                'surplus_percentage': float(df['is_surplus_renewable'].mean() * 100),
                'max_surplus': df['renewable_surplus_mwh'].max()
            },
            'opportunity_analysis': {
                'golden_hours': int(df['golden_hour'].sum()),
                'golden_hour_percentage': float(df['golden_hour'].mean() * 100),
                'avg_price_during_golden_hours': df[df['golden_hour'] == 1]['price_eur_mwh'].mean() if df['golden_hour'].sum() > 0 else 0
            },
            'rl_readiness': {
                'state_space_size': self.rl_config.get('n_unique_states', 0),
                'feature_completeness': (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
                'temporal_coverage_hours': len(df),
                'data_quality_score': self._calculate_data_quality_score(df)
            }
        }
        
        return stats
    
    def _calculate_data_quality_score(self, df: pd.DataFrame) -> float:
        """
        Calculate data quality score (0-100)
        """
        # Factors for data quality
        completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns)))
        consistency = 1 - (df.duplicated().sum() / len(df))
        validity = 1 - ((df['price_eur_mwh'] < -1000).sum() + (df['price_eur_mwh'] > 3000).sum()) / len(df)
        
        # Weighted average
        quality_score = (0.4 * completeness + 0.3 * consistency + 0.3 * validity) * 100
        return round(quality_score, 2)
    
    def export_data(self, rl_dataset: Dict[str, Any], format_type: str = 'pickle') -> List[str]:
        """
        Export processed data in various formats
        
        Args:
            rl_dataset: Processed RL dataset
            format_type: 'pickle', 'csv', 'json', or 'all'
        """
        print(f"\n💾 Exporting data in {format_type} format...")
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        exported_files = []
        
        # Base filename
        base_name = f"smard_processed_{timestamp}"
        
        if format_type in ['pickle', 'all']:
            # Pickle format (preserves all Python objects)
            pickle_file = os.path.join(self.output_dir, f"{base_name}.pkl")
            pd.to_pickle(rl_dataset, pickle_file)
            exported_files.append(pickle_file)
            print(f"   📦 Pickle: {pickle_file}")
        
        if format_type in ['csv', 'all']:
            # CSV format (data only)
            csv_file = os.path.join(self.output_dir, f"{base_name}.csv")
            rl_dataset['data'].to_csv(csv_file, index=False)
            exported_files.append(csv_file)
            print(f"   📊 CSV: {csv_file}")
        
        if format_type in ['json', 'all']:
            # JSON format (statistics and config)
            json_file = os.path.join(self.output_dir, f"{base_name}_config.json")
            export_config = {
                'statistics': rl_dataset['statistics'],
                'rl_config': rl_dataset['rl_config'],
                'feature_sets': rl_dataset['feature_sets'],
                'processing_timestamp': datetime.datetime.now().isoformat()
            }
            
            with open(json_file, 'w') as f:
                json.dump(export_config, f, indent=2, default=str)
            exported_files.append(json_file)
            print(f"   ⚙️ Config: {json_file}")
        
        # Feature sets as separate CSV files
        for set_name, features in rl_dataset['feature_sets'].items():
            feature_file = os.path.join(self.output_dir, f"{base_name}_{set_name}_features.csv")
            available_features = [f for f in features if f in rl_dataset['data'].columns]
            if available_features:
                feature_data = rl_dataset['data'][['datetime_from'] + available_features]
                feature_data.to_csv(feature_file, index=False)
                exported_files.append(feature_file)
                print(f"   🎯 {set_name.capitalize()}: {feature_file}")
        
        print(f"✅ Export completed: {len(exported_files)} files")
        return exported_files
    
    def generate_eda_report(self, rl_dataset: Dict[str, Any]) -> str:
        """
        Generate exploratory data analysis report
        """
        print("\n📊 Generating EDA report...")
        
        # Create visualizations
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('SMARD Data - Exploratory Data Analysis', fontsize=16, fontweight='bold')
        
        df = rl_dataset['data']
        
        # 1. Price distribution
        axes[0, 0].hist(df['price_eur_mwh'], bins=50, alpha=0.7, color='blue')
        axes[0, 0].set_title('Electricity Price Distribution')
        axes[0, 0].set_xlabel('Price (EUR/MWh)')
        axes[0, 0].axvline(x=0, color='red', linestyle='--', label='Zero Price')
        axes[0, 0].legend()
        
        # 2. Renewable vs Load
        axes[0, 1].scatter(df['load_mwh'], df['total_renewable_mwh'], alpha=0.5)
        axes[0, 1].plot([df['load_mwh'].min(), df['load_mwh'].max()], 
                       [df['load_mwh'].min(), df['load_mwh'].max()], 'r--', label='Load = Renewable')
        axes[0, 1].set_xlabel('Load (MWh)')
        axes[0, 1].set_ylabel('Total Renewable (MWh)')
        axes[0, 1].set_title('Renewable vs Load')
        axes[0, 1].legend()
        
        # 3. Hourly patterns
        hourly_stats = df.groupby('hour')[['price_eur_mwh', 'total_renewable_mwh', 'load_mwh']].mean()
        axes[1, 0].plot(hourly_stats.index, hourly_stats['price_eur_mwh'], label='Price')
        axes[1, 0].set_xlabel('Hour of Day')
        axes[1, 0].set_ylabel('Average Price (EUR/MWh)')
        axes[1, 0].set_title('Average Hourly Price Pattern')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Golden hours distribution
        golden_hours = df[df['golden_hour'] == 1]['hour'].value_counts().sort_index()
        if len(golden_hours) > 0:
            axes[1, 1].bar(golden_hours.index, golden_hours.values, alpha=0.7, color='gold')
            axes[1, 1].set_xlabel('Hour of Day')
            axes[1, 1].set_ylabel('Count of Golden Hours')
            axes[1, 1].set_title('Golden Hours Distribution')
        else:
            axes[1, 1].text(0.5, 0.5, 'No Golden Hours Found', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Golden Hours Distribution')
        
        # 5. Price vs Renewable Surplus
        axes[2, 0].scatter(df['renewable_surplus_mwh'], df['price_eur_mwh'], alpha=0.5)
        axes[2, 0].set_xlabel('Renewable Surplus (MWh)')
        axes[2, 0].set_ylabel('Price (EUR/MWh)')
        axes[2, 0].set_title('Price vs Renewable Surplus')
        axes[2, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[2, 0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
        
        # 6. Time series overview
        sample_data = df.sample(min(1000, len(df))).sort_values('datetime_from')
        axes[2, 1].plot(sample_data['datetime_from'], sample_data['price_eur_mwh'], alpha=0.7, linewidth=1)
        axes[2, 1].set_xlabel('Date')
        axes[2, 1].set_ylabel('Price (EUR/MWh)')
        axes[2, 1].set_title('Price Time Series (Sample)')
        axes[2, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(self.output_dir, f"eda_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ EDA report saved: {plot_file}")
        return plot_file


def run_preprocessing_pipeline(csv_path: str, 
                             output_dir: str = "./preprocessed_data/",
                             n_state_bins: int = 10,
                             scaler_type: str = 'robust',
                             export_format: str = 'all') -> Dict[str, Any]:
    """
    Run complete preprocessing pipeline
    
    Args:
        csv_path: Path to SMARD CSV file
        output_dir: Output directory
        n_state_bins: Number of bins for state discretization
        scaler_type: Type of scaler ('standard', 'minmax', 'robust')
        export_format: Export format ('pickle', 'csv', 'json', 'all')
    
    Returns:
        Dictionary with processed data and metadata
    """
    
    print("🚀 STARTING SMARD DATA PREPROCESSING PIPELINE")
    print("=" * 60)
    
    try:
        # Initialize preprocessor
        preprocessor = SMARDDataPreprocessor(csv_path, output_dir)
        
        # Step 1: Load raw data
        preprocessor.load_raw_data()
        
        # Step 2: Clean and validate
        preprocessor.clean_and_validate()
        
        # Step 3: Create features
        preprocessor.create_features()
        
        # Step 4: Handle missing values
        preprocessor.handle_missing_values()
        
        # Step 5: Create RL states
        preprocessor.create_rl_states(n_state_bins)
        
        # Step 6: Scale features
        preprocessor.scale_features(scaler_type)
        
        # Step 7: Create RL dataset
        rl_dataset = preprocessor.create_rl_dataset()
        
        # Step 8: Export data
        exported_files = preprocessor.export_data(rl_dataset, export_format)
        
        # Step 9: Generate EDA report
        eda_file = preprocessor.generate_eda_report(rl_dataset)
        
        # Final summary
        print("\n✅ PREPROCESSING PIPELINE COMPLETED!")
        print("=" * 60)
        print(f"📊 Processed {rl_dataset['statistics']['data_overview']['total_hours']} hours of data")
        print(f"💰 Found {rl_dataset['statistics']['opportunity_analysis']['golden_hours']} golden hours")
        print(f"📈 Data quality score: {rl_dataset['statistics']['rl_readiness']['data_quality_score']}/100")
        print(f"🎯 State space size: {rl_dataset['statistics']['rl_readiness']['state_space_size']}")
        print(f"📁 Files exported: {len(exported_files) + 1}")
        print("=" * 60)
        
        return {
            'rl_dataset': rl_dataset,
            'exported_files': exported_files,
            'eda_file': eda_file,
            'preprocessor': preprocessor,
            'pipeline_config': {
                'n_state_bins': n_state_bins,
                'scaler_type': scaler_type,
                'export_format': export_format
            }
        }
        
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        raise


# Example usage
if __name__ == "__main__":
    # Configuration
    csv_path = "/home/student/dalim/hiwiWagner/task_4_1_modelling/drive/energie_zusammengefasst_mit_aufloesung.csv"
    output_dir = "./preprocessed_smard_data/"
    
    # Run preprocessing
    if os.path.exists(csv_path):
        print("🚀 Running preprocessing with real data...")
        
        results = run_preprocessing_pipeline(
            csv_path=csv_path,
            output_dir=output_dir,
            n_state_bins=10,
            scaler_type='robust',
            export_format='all'
        )
        
        print(f"\n🎯 Ready for RL training!")
        print(f"📦 Load data with: pd.read_pickle('{results['exported_files'][0]}')")
        
    else:
        print("📁 CSV file not found. Update csv_path and run again.")
        print(f"Expected path: {csv_path}")