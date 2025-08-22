#!/usr/bin/env python3
"""
FPL GW5 Prediction Script
=========================

This script predicts player scores for Gameweek 5 using historical data from GW0-GW4.
Automatically checks data availability and provides clear requirements.

Usage:
    python fpl_gw5_prediction.py

Requirements:
    - Training data: GW0, GW1, GW2, GW3, GW4 (with playerstats.csv)
    - Prediction target: GW5 (with players.csv, fixtures.csv)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Machine Learning libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class FPLPredictor:
    def __init__(self, data_path=None):
        self.data_path = Path(data_path or '/Users/macbook/Dropbox/GitHub/FPL/FPL-Elo-Insights/data/2025-2026/By Gameweek')
        self.prediction_target = 5  # GW5
        # IMPORTANT: Exclude GW0 (friendlies) - only use official EPL gameweeks
        self.required_training_gws = list(range(1, 5))  # GW1-GW4 (official matches only)
        
    def check_data_availability(self):
        """Check if required training data is available for GW5 prediction (excluding GW0 friendlies)"""
        print("🔍 CHECKING DATA AVAILABILITY FOR GW5 PREDICTION")
        print("📝 Note: Excluding GW0 (friendlies) - Using only official EPL matches")
        print("=" * 60)
        
        available_gws = []
        missing_data = []
        
        # Check training gameweeks (GW1-GW4 - official EPL matches only)
        for gw in self.required_training_gws:
            gw_path = self.data_path / f'GW{gw}'
            required_files = ['playerstats.csv', 'players.csv', 'teams.csv']
            
            if gw_path.exists():
                missing_files = [f for f in required_files if not (gw_path / f).exists()]
                if not missing_files:
                    available_gws.append(gw)
                    print(f"✅ GW{gw}: Complete training data available")
                else:
                    missing_data.append(f"GW{gw}: Missing {missing_files}")
                    print(f"❌ GW{gw}: Missing {missing_files}")
            else:
                missing_data.append(f"GW{gw}: Directory not found")
                print(f"❌ GW{gw}: Directory not found")
        
        # Check prediction target (GW5)
        gw5_path = self.data_path / f'GW{self.prediction_target}'
        gw5_files = ['players.csv', 'fixtures.csv']
        gw5_missing = [f for f in gw5_files if not (gw5_path / f).exists()]
        
        if gw5_path.exists() and not gw5_missing:
            print(f"✅ GW{self.prediction_target}: Prediction target data available")
            has_gw5_results = (gw5_path / 'playerstats.csv').exists()
            if has_gw5_results:
                print(f"⚠️  GW{self.prediction_target}: Already has results - consider predicting GW{self.prediction_target + 1}")
        else:
            missing_data.append(f"GW{self.prediction_target}: Missing {gw5_missing if gw5_missing else 'directory'}")
            print(f"❌ GW{self.prediction_target}: Missing {gw5_missing if gw5_missing else 'directory'}")
        
        print(f"\n📊 SUMMARY:")
        print(f"   Available training gameweeks: {available_gws}")
        print(f"   Required for GW5 prediction: {self.required_training_gws} (official EPL only)")
        print(f"   Data completeness: {len(available_gws)}/{len(self.required_training_gws)} gameweeks")
        
        if len(available_gws) >= 2:  # Minimum viable for official matches
            print(f"✅ Sufficient data for prediction (minimum 2 official GWs)")
            return True, available_gws, missing_data
        else:
            print(f"❌ Insufficient data for reliable prediction")
            print(f"\n🔧 WHAT'S NEEDED:")
            for missing in missing_data:
                print(f"   • {missing}")
            return False, available_gws, missing_data
    
    def load_gameweek_data(self, available_gws):
        """Load and combine data from available gameweeks"""
        print(f"\n📚 LOADING DATA FROM {len(available_gws)} GAMEWEEKS...")
        
        all_data = []
        team_mapping = {}
        
        for gw in available_gws:
            gw_path = self.data_path / f'GW{gw}'
            
            # Load files
            playerstats = pd.read_csv(gw_path / 'playerstats.csv')
            players = pd.read_csv(gw_path / 'players.csv')
            teams = pd.read_csv(gw_path / 'teams.csv')
            
            # Create team mapping (code -> name)
            if gw == max(available_gws):  # Use latest for mapping
                team_mapping = dict(zip(teams['code'], teams['name']))
            
            # Merge player data
            merged = playerstats.merge(players, on='web_name', how='left', suffixes=('', '_player'))
            merged['gameweek'] = gw
            
            # Add position indicators
            position_map = {'Goalkeeper': 1, 'Defender': 2, 'Midfielder': 3, 'Forward': 4}
            merged['element_type'] = merged['position'].map(position_map).fillna(0)
            merged['is_gk'] = (merged['element_type'] == 1).astype(int)
            merged['is_def'] = (merged['element_type'] == 2).astype(int) 
            merged['is_mid'] = (merged['element_type'] == 3).astype(int)
            merged['is_fwd'] = (merged['element_type'] == 4).astype(int)
            
            all_data.append(merged)
            print(f"   ✓ GW{gw}: {len(merged)} players loaded")
        
        combined_data = pd.concat(all_data, ignore_index=True)
        print(f"📊 Combined training data: {len(combined_data)} records")
        
        return combined_data, team_mapping
    
    def create_features(self, df):
        """Create features for model training"""
        print("🔧 CREATING ENHANCED FEATURES...")
        
        # Fill missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        # Basic features
        feature_columns = [
            'minutes', 'goals_scored', 'assists', 'clean_sheets', 'saves',
            'yellow_cards', 'red_cards', 'bonus', 'bps',
            'influence', 'creativity', 'threat', 'ict_index',
            'expected_goals', 'expected_assists', 'expected_goal_involvements',
            'is_gk', 'is_def', 'is_mid', 'is_fwd'
        ]
        
        # Add derived features
        df['points_per_minute'] = np.where(df['minutes'] > 0, df['event_points'] / df['minutes'], 0)
        df['goals_per_90'] = np.where(df['minutes'] > 0, (df['goals_scored'] * 90) / df['minutes'], 0)
        df['assists_per_90'] = np.where(df['minutes'] > 0, (df['assists'] * 90) / df['minutes'], 0)
        
        feature_columns.extend(['points_per_minute', 'goals_per_90', 'assists_per_90'])
        
        # Create form features (if multiple gameweeks available)
        if len(df['gameweek'].unique()) > 1:
            print("   Adding form-based features...")
            df = df.sort_values(['web_name', 'gameweek'])
            
            # Rolling averages
            for feature in ['event_points', 'minutes', 'goals_scored', 'assists']:
                if feature in df.columns:
                    df[f'{feature}_avg_3gw'] = df.groupby('web_name')[feature].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
                    feature_columns.append(f'{feature}_avg_3gw')
            
            # Games played
            df['games_played'] = df.groupby('web_name').cumcount() + 1
            feature_columns.append('games_played')
        
        # Filter to available features
        available_features = [col for col in feature_columns if col in df.columns]
        print(f"   ✓ Created {len(available_features)} features")
        
        return df, available_features
    
    def train_models(self, X, y):
        """Train multiple models and select the best one"""
        print("🤖 TRAINING PREDICTION MODELS...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Initialize models
        models = {
            'Ridge Regression': Ridge(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        model_results = {}
        
        for name, model in models.items():
            print(f"   Training {name}...")
            
            # Use scaled data for Ridge, original for Random Forest
            if name == 'Ridge Regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            model_results[name] = {
                'model': model,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'scaler': scaler if name == 'Ridge Regression' else None
            }
            
            print(f"     RMSE: {rmse:.3f}, MAE: {mae:.3f}, R²: {r2:.3f}")
        
        # Select best model
        best_model_name = min(model_results.keys(), key=lambda x: model_results[x]['rmse'])
        best_model_info = model_results[best_model_name]
        
        print(f"🏆 Best model: {best_model_name} (RMSE: {best_model_info['rmse']:.3f})")
        
        return best_model_info, model_results
    
    def generate_gw5_predictions(self, model_info, feature_names, team_mapping):
        """Generate predictions for GW5"""
        print(f"\n🔮 GENERATING GW5 PREDICTIONS...")
        
        # Load GW5 player data
        gw5_path = self.data_path / f'GW{self.prediction_target}'
        gw5_players = pd.read_csv(gw5_path / 'players.csv')
        
        # Prepare features
        gw5_data = gw5_players.copy()
        
        # Add position indicators
        position_map = {'Goalkeeper': 1, 'Defender': 2, 'Midfielder': 3, 'Forward': 4}
        gw5_data['element_type'] = gw5_data['position'].map(position_map).fillna(0)
        gw5_data['is_gk'] = (gw5_data['element_type'] == 1).astype(int)
        gw5_data['is_def'] = (gw5_data['element_type'] == 2).astype(int)
        gw5_data['is_mid'] = (gw5_data['element_type'] == 3).astype(int)
        gw5_data['is_fwd'] = (gw5_data['element_type'] == 4).astype(int)
        
        # Add default values for missing features
        for feature in feature_names:
            if feature not in gw5_data.columns:
                if 'avg' in feature or 'per_90' in feature or 'games_played' in feature:
                    gw5_data[feature] = 0  # Historical averages start at 0
                else:
                    gw5_data[feature] = 0
        
        # Prepare feature matrix
        X_gw5 = gw5_data[feature_names].fillna(0)
        
        # Make predictions
        if model_info['scaler'] is not None:
            X_gw5_scaled = model_info['scaler'].transform(X_gw5)
            predictions = model_info['model'].predict(X_gw5_scaled)
        else:
            predictions = model_info['model'].predict(X_gw5)
        
        # Add predictions and team names
        gw5_data['predicted_points'] = np.round(np.maximum(0, predictions), 1)
        if 'team_code' in gw5_data.columns:
            gw5_data['team'] = gw5_data['team_code'].map(team_mapping)
        
        # Sort by predicted points
        gw5_predictions = gw5_data.sort_values('predicted_points', ascending=False)
        
        print(f"✅ Generated predictions for {len(gw5_predictions)} players")
        print(f"   Average predicted points: {gw5_predictions['predicted_points'].mean():.2f}")
        print(f"   Top prediction: {gw5_predictions['predicted_points'].max():.1f} points")
        
        return gw5_predictions
    
    def display_top_predictions(self, predictions, top_n=20):
        """Display top predicted performers"""
        print(f"\n🌟 TOP {top_n} PREDICTED PERFORMERS FOR GW5:")
        print("=" * 70)
        print(f"{'#':<3} {'Player':<18} {'Team':<15} {'Pos':<4} {'Points':<6}")
        print("-" * 70)
        
        top_predictions = predictions.head(top_n)
        
        for idx, (_, player) in enumerate(top_predictions.iterrows(), 1):
            name = player.get('web_name', 'Unknown')[:17]
            team = player.get('team', 'Unknown')[:14]
            position = player.get('position', 'Unknown')[:3]
            points = player.get('predicted_points', 0)
            
            print(f"{idx:<3} {name:<18} {team:<15} {position:<4} {points:<6}")
        
        # Position summary
        print(f"\n📊 PREDICTIONS BY POSITION:")
        if 'position' in predictions.columns:
            pos_summary = predictions.groupby('position')['predicted_points'].agg(['count', 'mean', 'max']).round(2)
            print(pos_summary)
    
    def run_prediction(self):
        """Main prediction workflow"""
        print("🚀 FPL GW5 PREDICTION SYSTEM")
        print("=" * 50)
        
        # Step 1: Check data availability
        is_ready, available_gws, missing_data = self.check_data_availability()
        
        if not is_ready:
            print(f"\n❌ CANNOT PROCEED WITH GW5 PREDICTION")
            print(f"💡 ACTION REQUIRED: Ensure GW1-GW4 data is available (official EPL matches only)")
            print(f"📝 Remember: GW0 (friendlies) is excluded from training data")
            return None
        
        # Step 2: Load and prepare data
        combined_data, team_mapping = self.load_gameweek_data(available_gws)
        combined_data, feature_names = self.create_features(combined_data)
        
        # Step 3: Prepare training data
        X = combined_data[feature_names].fillna(0)
        y = combined_data['event_points'].fillna(0)
        
        print(f"📈 Training data: {len(X)} samples, {len(feature_names)} features")
        
        # Step 4: Train models
        best_model_info, all_models = self.train_models(X, y)
        
        # Step 5: Generate GW5 predictions
        gw5_predictions = self.generate_gw5_predictions(best_model_info, feature_names, team_mapping)
        
        # Step 6: Display results
        self.display_top_predictions(gw5_predictions)
        
        print(f"\n✅ GW5 PREDICTION COMPLETE!")
        print(f"🎯 Model used: {list(all_models.keys())[0] if best_model_info else 'N/A'}")
        print(f"📊 Training gameweeks: GW{min(available_gws)}-GW{max(available_gws)}")
        
        return gw5_predictions

if __name__ == "__main__":
    # Run GW5 prediction
    predictor = FPLPredictor()
    predictions = predictor.run_prediction()
    
    if predictions is not None:
        # Save predictions to CSV
        output_path = Path("gw5_predictions.csv")
        predictions.to_csv(output_path, index=False)
        print(f"\n💾 Predictions saved to: {output_path}")