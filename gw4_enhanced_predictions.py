#!/usr/bin/env python3
"""
GW4 Enhanced Predictions for HIGH SCORING
Using GW1-GW3 data to predict GW4 with focus on 65+ point strategy
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

def load_historical_data_gw4():
    """Load GW1-GW3 data for GW4 predictions"""
    print("📊 Loading historical data (GW1-GW3) for enhanced GW4 predictions...")
    
    gw1_stats = pd.read_csv('data/2025-2026/By Gameweek/GW1/playerstats.csv')
    gw2_stats = pd.read_csv('data/2025-2026/By Gameweek/GW2/playerstats.csv')
    gw3_stats = pd.read_csv('data/2025-2026/By Gameweek/GW3/playerstats.csv')
    
    # Combine all data
    historical_stats = []
    for gw, data in [(1, gw1_stats), (2, gw2_stats), (3, gw3_stats)]:
        temp = data.copy()
        temp['gameweek'] = gw
        historical_stats.append(temp)
    
    combined_stats = pd.concat(historical_stats, ignore_index=True)
    print(f"✅ Loaded {len(combined_stats)} player-gameweek records for enhanced analysis")
    
    return combined_stats

def calculate_enhanced_form_score(player_stats):
    """Enhanced form calculation with heavy GW3 weighting"""
    if len(player_stats) == 0:
        return 0
    
    # Weight: GW3 = 50%, GW2 = 30%, GW1 = 20%
    weights = {3: 0.5, 2: 0.3, 1: 0.2}
    
    form_score = 0
    total_weight = 0
    
    for _, game in player_stats.iterrows():
        gw = game['gameweek']
        weight = weights.get(gw, 0)
        form_score += game['event_points'] * weight
        total_weight += weight
    
    return form_score / total_weight if total_weight > 0 else 0

def get_fixture_difficulty_ratings_gw4():
    """Enhanced FDR for GW4 based on actual fixtures and team strength"""
    # Based on GW4 fixtures and team Elo ratings
    fixture_difficulty = {
        # Very Easy (FDR 1-2)
        'Arsenal': 2,  # vs Forest at home
        'Liverpool': 2,  # vs Burnley away
        'Man City': 3,  # vs Man Utd at home (derby factor)
        
        # Easy-Medium (FDR 2-3)
        'Chelsea': 3,  # vs Brentford away
        'Tottenham': 3,  # vs West Ham away
        'Newcastle': 2,  # vs Wolves at home
        'Brighton': 3,  # vs Bournemouth away
        
        # Medium (FDR 3)
        'Aston Villa': 3,  # vs Everton away
        'Crystal Palace': 2,  # vs Sunderland at home
        'Fulham': 2,  # vs Leeds at home
        
        # Harder fixtures (FDR 4-5)
        'Man Utd': 4,  # vs Man City away
        'Brentford': 4,  # vs Chelsea at home
        'West Ham': 4,  # vs Tottenham at home
        'Wolves': 4,  # vs Newcastle away
        'Everton': 3,  # vs Aston Villa at home
        'Bournemouth': 4,  # vs Brighton at home
        'Burnley': 5,  # vs Liverpool at home
        'Leeds': 4,  # vs Fulham away
        'Nott\'m Forest': 4,  # vs Arsenal away
        'Sunderland': 5   # vs Crystal Palace away
    }
    
    return fixture_difficulty

def predict_gw4_enhanced(historical_stats):
    """Enhanced GW4 predictions focusing on high-scoring potential"""
    print("🎯 Generating enhanced GW4 predictions for high scoring...")
    
    # Load GW4 data
    gw4_players = pd.read_csv('data/2025-2026/By Gameweek/GW4/players.csv')
    gw4_teams = pd.read_csv('data/2025-2026/By Gameweek/GW4/teams.csv')
    gw4_fixtures = pd.read_csv('data/2025-2026/By Gameweek/GW4/fixtures.csv')
    
    # Team mappings
    team_mapping = {row['code']: row['name'] for _, row in gw4_teams.iterrows()}
    
    # Fixture mappings
    fixture_map = {}
    for _, fixture in gw4_fixtures.iterrows():
        home_code = int(fixture['home_team'])
        away_code = int(fixture['away_team'])
        
        home_team = team_mapping.get(home_code)
        away_team = team_mapping.get(away_code)
        
        if home_team and away_team:
            fixture_map[home_team] = {
                'opponent': away_team,
                'is_home': True,
                'quality': fixture['home_team_elo'] + fixture['away_team_elo']
            }
            fixture_map[away_team] = {
                'opponent': home_team,
                'is_home': False,
                'quality': fixture['home_team_elo'] + fixture['away_team_elo']
            }
    
    fdr_ratings = get_fixture_difficulty_ratings_gw4()
    predictions = []
    
    for _, player in gw4_players.iterrows():
        player_id = player['player_id']
        player_name = player['web_name']
        team_code = player['team_code']
        team_name = team_mapping.get(team_code, f'Team_{team_code}')
        position = player['position']
        
        # Get historical performance
        player_historical = historical_stats[historical_stats['id'] == player_id].copy()
        
        if len(player_historical) == 0:
            base_prediction = get_position_default_gw4(position)
        else:
            # Enhanced calculation
            enhanced_form = calculate_enhanced_form_score(player_historical)
            avg_points = player_historical['event_points'].mean()
            recent_minutes = player_historical[player_historical['gameweek'] == 3]['minutes'].iloc[0] if len(player_historical[player_historical['gameweek'] == 3]) > 0 else 60
            
            # Weight recent form heavily (70% vs 30% average)
            base_prediction = (enhanced_form * 0.7) + (avg_points * 0.3)
            
            # Minutes adjustment
            if recent_minutes >= 80:
                base_prediction *= 1.0
            elif recent_minutes >= 60:
                base_prediction *= 0.9
            elif recent_minutes >= 30:
                base_prediction *= 0.6
            else:
                base_prediction *= 0.3
        
        # Fixture adjustments
        if team_name in fixture_map:
            fixture_info = fixture_map[team_name]
            
            # Home/Away multiplier
            ha_multiplier = get_home_away_multiplier_gw4(position, fixture_info['is_home'])
            base_prediction *= ha_multiplier
            
            # FDR multiplier
            team_fdr = fdr_ratings.get(team_name, 3)
            fdr_multiplier = get_fdr_multiplier_gw4(position, team_fdr)
            base_prediction *= fdr_multiplier
            
            # High-quality fixture bonus (for goal potential)
            if fixture_info['quality'] > 3600:
                base_prediction *= 1.1  # 10% bonus for high-scoring games
            
            opponent = fixture_info['opponent']
            is_home = fixture_info['is_home']
            fdr = team_fdr
        else:
            opponent = 'Unknown'
            is_home = None
            fdr = 3
        
        # Calculate ceiling (differential potential)
        ceiling_multiplier = 1.0
        if position == 'Forward':
            ceiling_multiplier = 1.4
        elif position == 'Midfielder':
            ceiling_multiplier = 1.3
        elif position == 'Defender':
            ceiling_multiplier = 1.2
        
        ceiling_points = base_prediction * ceiling_multiplier
        
        predictions.append({
            'player_id': player_id,
            'name': player_name,
            'team': team_name,
            'position': position,
            'predicted_points': round(base_prediction, 2),
            'ceiling_points': round(ceiling_points, 2),
            'opponent': opponent,
            'fdr': fdr,
            'is_home': is_home,
            'form_score': round(calculate_enhanced_form_score(player_historical), 2) if len(player_historical) > 0 else 0
        })
    
    predictions_df = pd.DataFrame(predictions)
    print(f"✅ Generated enhanced predictions for {len(predictions_df)} players")
    
    return predictions_df

def get_position_default_gw4(position):
    """Position defaults for new players"""
    defaults = {
        'Forward': 5.0,
        'Midfielder': 4.2,
        'Defender': 3.5,
        'Goalkeeper': 3.0
    }
    return defaults.get(position, 3.5)

def get_home_away_multiplier_gw4(position, is_home):
    """Enhanced home/away multipliers"""
    multipliers = {
        'Forward': {'home': 1.20, 'away': 0.88},
        'Midfielder': {'home': 1.15, 'away': 0.92},
        'Defender': {'home': 1.12, 'away': 0.91},
        'Goalkeeper': {'home': 1.10, 'away': 0.88}
    }
    
    return multipliers[position]['home' if is_home else 'away']

def get_fdr_multiplier_gw4(position, fdr):
    """Enhanced FDR multipliers for high scoring"""
    base_multipliers = {
        1: 1.35,  # Very Easy
        2: 1.20,  # Easy
        3: 1.00,  # Average
        4: 0.80,  # Hard
        5: 0.65   # Very Hard
    }
    
    multiplier = base_multipliers.get(fdr, 1.0)
    
    # Position adjustments
    if position in ['Forward', 'Midfielder'] and fdr <= 2:
        multiplier += 0.05  # Extra boost for attacking players in good fixtures
    
    return multiplier

def analyze_captaincy_gw4(predictions_df):
    """Analyze captaincy options for GW4"""
    print(f"\n👑 GW4 CAPTAINCY ANALYSIS FOR RANK GAINS:")
    print("=" * 55)
    
    # Filter potential captains (high ceiling, good fixtures)
    captain_candidates = predictions_df[
        (predictions_df['predicted_points'] >= 6.0) |
        (predictions_df['ceiling_points'] >= 10.0)
    ].copy()
    
    captain_candidates = captain_candidates.sort_values('ceiling_points', ascending=False)
    
    print("🎯 TOP CAPTAINCY OPTIONS (by ceiling):")
    print("-" * 50)
    print("Player           Team         Fixture      Predicted  Ceiling  FDR  Form")
    print("-" * 75)
    
    for _, player in captain_candidates.head(15).iterrows():
        home_away = "H" if player['is_home'] else "A" if player['is_home'] == False else "N"
        fixture = f"vs {player['opponent']} ({home_away})"
        
        print(f"{player['name']:<15} {player['team']:<12} {fixture:<12} "
              f"{player['predicted_points']:>9.1f} {player['ceiling_points']:>8.1f} "
              f"{player['fdr']:>4} {player['form_score']:>5.1f}")
    
    # Differential captain analysis
    print(f"\n🎲 DIFFERENTIAL CAPTAIN TARGETS:")
    print("-" * 40)
    
    differential_captains = captain_candidates[
        (captain_candidates['ceiling_points'] >= 12.0) &
        (captain_candidates['fdr'] <= 3)
    ].head(8)
    
    for _, player in differential_captains.iterrows():
        risk_level = "🔥 HIGH" if player['fdr'] >= 4 else "⚡ MED" if player['fdr'] == 3 else "✅ LOW"
        print(f"• {player['name']} ({player['team']}) - Ceiling: {player['ceiling_points']:.1f} {risk_level}")

def identify_transfer_targets_gw4(predictions_df):
    """Identify best transfer targets for GW4"""
    print(f"\n🔄 GW4 TRANSFER TARGETS:")
    print("=" * 35)
    
    # High-value targets by position
    positions = ['Forward', 'Midfielder', 'Defender', 'Goalkeeper']
    
    for position in positions:
        pos_players = predictions_df[predictions_df['position'] == position].copy()
        
        # Sort by predicted points for this position
        top_pos_players = pos_players.nlargest(5, 'predicted_points')
        
        print(f"\n🎯 TOP {position.upper()}S:")
        print(f"Player           Team         Fixture      Predicted  Form   FDR")
        print("-" * 65)
        
        for _, player in top_pos_players.iterrows():
            home_away = "H" if player['is_home'] else "A" if player['is_home'] == False else "N"
            fixture = f"vs {player['opponent']} ({home_away})"
            
            print(f"{player['name']:<15} {player['team']:<12} {fixture:<12} "
                  f"{player['predicted_points']:>9.1f} {player['form_score']:>6.1f} {player['fdr']:>4}")

def main():
    print("🚀 GW4 ENHANCED PREDICTIONS FOR HIGH SCORING")
    print("=" * 80)
    print("🎯 Target: 65+ points through data-driven selections")
    print()
    
    # Load and analyze
    historical_stats = load_historical_data_gw4()
    predictions_df = predict_gw4_enhanced(historical_stats)
    
    # Analysis
    analyze_captaincy_gw4(predictions_df)
    identify_transfer_targets_gw4(predictions_df)
    
    # Save predictions
    predictions_df.to_csv('gw4_enhanced_predictions.csv', index=False)
    print(f"\n💾 Enhanced predictions saved to 'gw4_enhanced_predictions.csv'")
    
    print(f"\n🏆 KEY INSIGHTS FOR 65+ POINTS:")
    print("=" * 40)
    
    # Top predicted scorers
    top_scorers = predictions_df.nlargest(10, 'predicted_points')
    print(f"🔥 HIGHEST PREDICTED SCORERS:")
    for _, player in top_scorers.iterrows():
        print(f"• {player['name']} ({player['team']}): {player['predicted_points']:.1f} pts")
    
    print(f"\n⚡ DIFFERENTIAL STRATEGY:")
    print("• Focus on players with high ceiling in good fixtures")
    print("• João Pedro continuation if no clear upgrade")
    print("• Target Arsenal/Liverpool attackers vs weak opposition")
    print("• Consider Man City/Man Utd players for derby goals")
    print("• Balance risk with 2-3 safe premium picks")

if __name__ == "__main__":
    main()