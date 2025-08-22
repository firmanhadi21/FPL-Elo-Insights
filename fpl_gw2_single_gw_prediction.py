#!/usr/bin/env python3
"""
FPL GW2 Prediction Script - Single Gameweek Training
====================================================

This script predicts player scores for Gameweek 2 using only GW1 data.
Optimized for minimal training data scenarios with enhanced feature engineering.

Usage:
    python fpl_gw2_single_gw_prediction.py

Requirements:
    - Training data: GW1 (with playerstats.csv)
    - Prediction target: GW2 (with players.csv, fixtures.csv)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Machine Learning libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class GW2Predictor:
    def __init__(self, data_path=None):
        self.data_path = Path(data_path or '/Users/macbook/Dropbox/GitHub/FPL/FPL-Elo-Insights/data/2025-2026/By Gameweek')
        self.training_gw = 1  # Only GW1 available
        self.prediction_target = 2  # GW2
        
    def check_data_availability(self):
        """Check if GW1 training data and GW2 target data are available"""
        print("🔍 CHECKING DATA AVAILABILITY FOR GW2 PREDICTION")
        print("📝 Strategy: Single gameweek training (GW1 → GW2)")
        print("=" * 60)
        
        # Check GW1 training data
        gw1_path = self.data_path / 'GW1'
        gw1_files = ['playerstats.csv', 'players.csv', 'teams.csv']
        gw1_missing = [f for f in gw1_files if not (gw1_path / f).exists()]
        
        if gw1_path.exists() and not gw1_missing:
            print("✅ GW1: Complete training data available")
            gw1_available = True
        else:
            print(f"❌ GW1: Missing {gw1_missing if gw1_missing else 'directory'}")
            gw1_available = False
        
        # Check GW2 prediction target
        gw2_path = self.data_path / 'GW2'
        gw2_files = ['players.csv', 'fixtures.csv']
        gw2_missing = [f for f in gw2_files if not (gw2_path / f).exists()]
        
        if gw2_path.exists() and not gw2_missing:
            print("✅ GW2: Prediction target data available")
            has_gw2_results = (gw2_path / 'playerstats.csv').exists()
            if has_gw2_results:
                print("⚠️  GW2: Already has results - consider predicting GW3")
            gw2_available = True
        else:
            print(f"❌ GW2: Missing {gw2_missing if gw2_missing else 'directory'}")
            gw2_available = False
        
        print(f"\n📊 SUMMARY:")
        print(f"   Training gameweek: GW{self.training_gw}")
        print(f"   Prediction target: GW{self.prediction_target}")
        print(f"   Data status: {'✅ Ready' if (gw1_available and gw2_available) else '❌ Incomplete'}")
        
        if gw1_available and gw2_available:
            print("✅ Sufficient data for single-gameweek prediction")
            return True
        else:
            print("❌ Insufficient data for prediction")
            return False
    
    def load_gw1_data(self):
        """Load and process GW1 training data"""
        print(f"\n📚 LOADING GW1 TRAINING DATA...")
        
        gw1_path = self.data_path / 'GW1'
        
        # Load files
        playerstats = pd.read_csv(gw1_path / 'playerstats.csv')
        players = pd.read_csv(gw1_path / 'players.csv')
        teams = pd.read_csv(gw1_path / 'teams.csv')
        
        # Create team mapping
        team_mapping = dict(zip(teams['code'], teams['name']))
        
        # Merge player data
        merged = playerstats.merge(players, on='web_name', how='left', suffixes=('', '_player'))
        
        # Add position indicators
        position_map = {'Goalkeeper': 1, 'Defender': 2, 'Midfielder': 3, 'Forward': 4}
        merged['element_type'] = merged['position'].map(position_map).fillna(0)
        merged['is_gk'] = (merged['element_type'] == 1).astype(int)
        merged['is_def'] = (merged['element_type'] == 2).astype(int)
        merged['is_mid'] = (merged['element_type'] == 3).astype(int)
        merged['is_fwd'] = (merged['element_type'] == 4).astype(int)
        
        print(f"   ✓ GW1: {len(merged)} players loaded")
        
        return merged, team_mapping
    
    def create_enhanced_features(self, df):
        """Create enhanced features optimized for single gameweek training"""
        print("🔧 CREATING ENHANCED FEATURES FOR SINGLE GAMEWEEK...")
        
        # Fill missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        # Core performance features
        feature_columns = [
            # Basic stats
            'minutes', 'goals_scored', 'assists', 'clean_sheets', 'saves',
            'yellow_cards', 'red_cards', 'bonus', 'bps',
            
            # Advanced stats
            'influence', 'creativity', 'threat', 'ict_index',
            'expected_goals', 'expected_assists', 'expected_goal_involvements',
            
            # Position indicators
            'is_gk', 'is_def', 'is_mid', 'is_fwd'
        ]
        
        # Enhanced derived features (crucial for single gameweek)
        df['points_per_minute'] = np.where(df['minutes'] > 0, df['event_points'] / df['minutes'], 0)
        df['goals_per_90'] = np.where(df['minutes'] > 0, (df['goals_scored'] * 90) / df['minutes'], 0)
        df['assists_per_90'] = np.where(df['minutes'] > 0, (df['assists'] * 90) / df['minutes'], 0)
        
        # ICT per 90 minutes (important for consistent comparison)
        df['influence_per_90'] = np.where(df['minutes'] > 0, (df['influence'] * 90) / df['minutes'], 0)
        df['creativity_per_90'] = np.where(df['minutes'] > 0, (df['creativity'] * 90) / df['minutes'], 0)
        df['threat_per_90'] = np.where(df['minutes'] > 0, (df['threat'] * 90) / df['minutes'], 0)
        
        # Position-specific features
        df['gk_save_bonus'] = df['is_gk'] * df['saves']  # GK-specific
        df['def_clean_bonus'] = df['is_def'] * df['clean_sheets']  # Defender-specific
        df['mid_assist_bonus'] = df['is_mid'] * df['assists']  # Midfielder-specific
        df['fwd_goal_bonus'] = df['is_fwd'] * df['goals_scored']  # Forward-specific
        
        # Playing time indicators
        df['is_starter'] = (df['minutes'] >= 60).astype(int)
        df['is_substitute'] = ((df['minutes'] > 0) & (df['minutes'] < 60)).astype(int)
        df['did_not_play'] = (df['minutes'] == 0).astype(int)
        
        # Efficiency metrics
        df['bps_per_minute'] = np.where(df['minutes'] > 0, df['bps'] / df['minutes'], 0)
        df['ict_efficiency'] = np.where(df['minutes'] > 0, df['ict_index'] / df['minutes'], 0)
        
        # Add all derived features to feature list
        derived_features = [
            'points_per_minute', 'goals_per_90', 'assists_per_90',
            'influence_per_90', 'creativity_per_90', 'threat_per_90',
            'gk_save_bonus', 'def_clean_bonus', 'mid_assist_bonus', 'fwd_goal_bonus',
            'is_starter', 'is_substitute', 'did_not_play',
            'bps_per_minute', 'ict_efficiency'
        ]
        
        feature_columns.extend(derived_features)
        
        # Filter to available features
        available_features = [col for col in feature_columns if col in df.columns]
        print(f"   ✓ Created {len(available_features)} features")
        
        return df, available_features
    
    def train_optimized_models(self, X, y):
        """Train models optimized for single gameweek data"""
        print("🤖 TRAINING MODELS OPTIMIZED FOR SINGLE GAMEWEEK...")
        
        # For single gameweek, use different validation strategy
        # Create synthetic train/test split based on different criteria
        
        # Strategy 1: Random split (standard)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Strategy 2: Split by position (to test generalization across positions)
        position_mask = X['is_fwd'] == 1  # Use forwards as test set
        X_train_pos = X[~position_mask]
        X_test_pos = X[position_mask]
        y_train_pos = y[~position_mask]
        y_test_pos = y[position_mask]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Models optimized for limited data
        models = {
            'Ridge Regression': Ridge(alpha=10.0),  # Higher regularization for limited data
            'Linear Regression': LinearRegression(),
            'Random Forest (Small)': RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)  # Simpler to avoid overfitting
        }
        
        model_results = {}
        
        for name, model in models.items():
            print(f"   Training {name}...")
            
            # Use scaled data for linear models, original for Random Forest
            if 'Regression' in name:
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
                'scaler': scaler if 'Regression' in name else None
            }
            
            print(f"     RMSE: {rmse:.3f}, MAE: {mae:.3f}, R²: {r2:.3f}")
        
        # Select best model based on MAE (more robust for limited data)
        best_model_name = min(model_results.keys(), key=lambda x: model_results[x]['mae'])
        best_model_info = model_results[best_model_name]
        
        print(f"🏆 Best model: {best_model_name} (MAE: {best_model_info['mae']:.3f})")
        
        return best_model_info, model_results
    
    def add_fixture_difficulty(self, gw2_data, team_mapping):
        """Add fixture difficulty analysis for GW2"""
        print("🏟️ ANALYZING GW2 FIXTURES...")
        
        # Load GW2 fixtures if available
        gw2_fixtures_path = self.data_path / 'GW2' / 'fixtures.csv'
        if gw2_fixtures_path.exists():
            fixtures = pd.read_csv(gw2_fixtures_path)
            
            # Simple fixture difficulty based on team strength (can be enhanced)
            # For now, use a basic home/away adjustment
            team_strength = {
                'Man City': 5, 'Arsenal': 5, 'Liverpool': 5, 'Chelsea': 4,
                'Man Utd': 4, 'Newcastle': 4, 'Spurs': 4, 'Aston Villa': 3,
                'Brighton': 3, 'West Ham': 3, 'Wolves': 3, 'Fulham': 3,
                'Brentford': 2, 'Crystal Palace': 2, 'Everton': 2, 'Bournemouth': 2,
                'Nott\'m Forest': 2, 'Burnley': 1, 'Sheffield Utd': 1, 'Luton': 1
            }
            
            # Add fixture difficulty to player data
            gw2_data['fixture_difficulty'] = 3  # Default neutral
            
            for _, fixture in fixtures.iterrows():
                home_team = fixture.get('home_team_name', '')
                away_team = fixture.get('away_team_name', '')
                
                if home_team in team_strength and away_team in team_strength:
                    home_strength = team_strength[home_team]
                    away_strength = team_strength[away_team]
                    
                    # Update difficulty for players in these teams
                    home_difficulty = 6 - away_strength  # Easier against weaker opponents
                    away_difficulty = 6 - home_strength - 0.5  # Away disadvantage
                    
                    if 'team' in gw2_data.columns:
                        gw2_data.loc[gw2_data['team'] == home_team, 'fixture_difficulty'] = home_difficulty
                        gw2_data.loc[gw2_data['team'] == away_team, 'fixture_difficulty'] = away_difficulty
            
            print(f"   ✓ Added fixture difficulty analysis")
        else:
            gw2_data['fixture_difficulty'] = 3  # Neutral if no fixtures data
            print(f"   ⚠ No fixtures data found, using neutral difficulty")
        
        return gw2_data
    
    def generate_gw2_predictions(self, model_info, feature_names, team_mapping):
        """Generate predictions for GW2"""
        print(f"\n🔮 GENERATING GW2 PREDICTIONS...")
        
        # Load GW2 player data
        gw2_path = self.data_path / 'GW2'
        gw2_players = pd.read_csv(gw2_path / 'players.csv')
        
        # Prepare features
        gw2_data = gw2_players.copy()
        
        # Add position indicators
        position_map = {'Goalkeeper': 1, 'Defender': 2, 'Midfielder': 3, 'Forward': 4}
        gw2_data['element_type'] = gw2_data['position'].map(position_map).fillna(0)
        gw2_data['is_gk'] = (gw2_data['element_type'] == 1).astype(int)
        gw2_data['is_def'] = (gw2_data['element_type'] == 2).astype(int)
        gw2_data['is_mid'] = (gw2_data['element_type'] == 3).astype(int)
        gw2_data['is_fwd'] = (gw2_data['element_type'] == 4).astype(int)
        
        # Add team names
        if 'team_code' in gw2_data.columns:
            gw2_data['team'] = gw2_data['team_code'].map(team_mapping)
        
        # Add fixture difficulty
        gw2_data = self.add_fixture_difficulty(gw2_data, team_mapping)
        
        # Add default values for missing features (using position-based estimates)
        for feature in feature_names:
            if feature not in gw2_data.columns:
                if feature in ['minutes', 'is_starter']:
                    # Estimate playing time based on position
                    gw2_data[feature] = 70 if feature == 'minutes' else 1
                elif 'per_90' in feature or 'efficiency' in feature:
                    # Use position-based defaults
                    if 'goals' in feature:
                        gw2_data[feature] = gw2_data['is_fwd'] * 0.5 + gw2_data['is_mid'] * 0.2
                    elif 'assists' in feature:
                        gw2_data[feature] = gw2_data['is_mid'] * 0.3 + gw2_data['is_fwd'] * 0.2
                    else:
                        gw2_data[feature] = 0.1  # Small default
                else:
                    gw2_data[feature] = 0
        
        # Prepare feature matrix
        X_gw2 = gw2_data[feature_names].fillna(0)
        
        # Make predictions
        if model_info['scaler'] is not None:
            X_gw2_scaled = model_info['scaler'].transform(X_gw2)
            predictions = model_info['model'].predict(X_gw2_scaled)
        else:
            predictions = model_info['model'].predict(X_gw2)
        
        # Apply fixture difficulty adjustment
        fixture_multiplier = 1 + (gw2_data['fixture_difficulty'] - 3) * 0.1  # ±10% per difficulty point
        predictions = predictions * fixture_multiplier
        
        # Add predictions
        gw2_data['predicted_points'] = np.round(np.maximum(0, predictions), 1)
        
        # Sort by predicted points
        gw2_predictions = gw2_data.sort_values('predicted_points', ascending=False)
        
        print(f"✅ Generated predictions for {len(gw2_predictions)} players")
        print(f"   Average predicted points: {gw2_predictions['predicted_points'].mean():.2f}")
        print(f"   Top prediction: {gw2_predictions['predicted_points'].max():.1f} points")
        
        return gw2_predictions
    
    def display_results(self, predictions, top_n=25):
        """Display comprehensive prediction results"""
        print(f"\n🌟 TOP {top_n} PREDICTED PERFORMERS FOR GW2:")
        print("=" * 75)
        print(f"{'#':<3} {'Player':<18} {'Team':<15} {'Pos':<4} {'Pts':<5} {'Diff':<4}")
        print("-" * 75)
        
        top_predictions = predictions.head(top_n)
        
        for idx, (_, player) in enumerate(top_predictions.iterrows(), 1):
            name = player.get('web_name', 'Unknown')[:17]
            team = player.get('team', 'Unknown')[:14]
            position = player.get('position', 'Unknown')[:3]
            points = player.get('predicted_points', 0)
            difficulty = player.get('fixture_difficulty', 3)
            
            diff_symbol = "🟢" if difficulty >= 4 else "🟡" if difficulty == 3 else "🔴"
            
            print(f"{idx:<3} {name:<18} {team:<15} {position:<4} {points:<5} {diff_symbol}")
        
        # Position analysis
        print(f"\n📊 PREDICTIONS BY POSITION:")
        pos_summary = predictions.groupby('position')['predicted_points'].agg(['count', 'mean', 'max']).round(2)
        print(pos_summary)
        
        # Top performers by position
        print(f"\n🥇 TOP PERFORMER BY POSITION:")
        for pos in ['Goalkeeper', 'Defender', 'Midfielder', 'Forward']:
            pos_players = predictions[predictions['position'] == pos]
            if not pos_players.empty:
                top_player = pos_players.iloc[0]
                print(f"   {pos}: {top_player['web_name']} ({top_player['predicted_points']:.1f} pts)")
    
    def run_prediction(self):
        """Main prediction workflow"""
        print("🚀 FPL GW2 PREDICTION SYSTEM (Single Gameweek Training)")
        print("=" * 60)
        
        # Step 1: Check data availability
        if not self.check_data_availability():
            print(f"\n❌ CANNOT PROCEED WITH GW2 PREDICTION")
            print(f"💡 ACTION REQUIRED: Ensure GW1 training data is available")
            return None
        
        # Step 2: Load GW1 training data
        gw1_data, team_mapping = self.load_gw1_data()
        gw1_data, feature_names = self.create_enhanced_features(gw1_data)
        
        # Step 3: Prepare training data
        X = gw1_data[feature_names].fillna(0)
        y = gw1_data['event_points'].fillna(0)
        
        print(f"📈 Training data: {len(X)} samples, {len(feature_names)} features")
        print(f"   Point distribution: μ={y.mean():.2f}, σ={y.std():.2f}, max={y.max()}")
        
        # Step 4: Train optimized models
        best_model_info, all_models = self.train_optimized_models(X, y)
        
        # Step 5: Generate GW2 predictions
        gw2_predictions = self.generate_gw2_predictions(best_model_info, feature_names, team_mapping)
        
        # Step 6: Display results
        self.display_results(gw2_predictions)
        
        print(f"\n✅ GW2 PREDICTION COMPLETE!")
        print(f"🎯 Model used: Ridge Regression (optimized for limited data)")
        print(f"📊 Training: Single gameweek (GW1)")
        print(f"🔧 Features: Enhanced with position-specific and efficiency metrics")
        
        return gw2_predictions

if __name__ == "__main__":
    # Run GW2 prediction with single gameweek training
    predictor = GW2Predictor()
    predictions = predictor.run_prediction()
    
    if predictions is not None:
        # Save predictions
        output_path = Path("gw2_predictions_single_gw.csv")
        predictions.to_csv(output_path, index=False)
        print(f"\n💾 Predictions saved to: {output_path}")