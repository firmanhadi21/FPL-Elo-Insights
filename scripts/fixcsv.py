import os
import pandas as pd
from pathlib import Path
import re

# Utility function to create directories
def create_directory(path):
    Path(path).mkdir(parents=True, exist_ok=True)

# Enhanced team mapping correction based on URL analysis
def create_team_id_mapping():
    """
    Create a mapping from the problematic team IDs to correct team names
    based on URL analysis from our comprehensive data consistency check.
    """
    # Correct team names (standard FPL teams)
    correct_teams = {
        1: 'Arsenal', 2: 'Aston Villa', 3: 'Bournemouth', 4: 'Brentford', 5: 'Brighton',
        6: 'Chelsea', 7: 'Crystal Palace', 8: 'Everton', 9: 'Fulham', 10: 'Ipswich',
        11: 'Leicester', 12: 'Liverpool', 13: 'Man City', 14: 'Man Utd', 15: 'Newcastle',
        16: "Nott'm Forest", 17: 'Southampton', 18: 'Spurs', 19: 'West Ham', 20: 'Wolves'
    }
    
    # Problematic ID to correct team name mapping (from URL analysis)
    problematic_id_mapping = {
        # From 2025-2026 GW1 analysis
        31: 'Crystal Palace',  # Chelsea vs Crystal Palace
        94: "Nott'm Forest",   # Brentford vs Nottingham Forest  
        3: 'Man Utd',          # Arsenal vs Manchester United (in this context)
        11: 'Everton',         # Leeds vs Everton (in this context)
        91: 'Bournemouth',     # Liverpool vs AFC Bournemouth
        54: 'Brighton',        # Fulham vs Brighton
        4: 'Newcastle',        # Aston Villa vs Newcastle (in this context)
        43: 'Wolves',          # Man City vs Wolves
        90: 'Spurs',           # Burnley vs Spurs
        21: 'West Ham',        # Sunderland vs West Ham
        
        # Additional mappings from 2024-2025 data patterns
        36: 'Brighton',
        39: 'Wolves', 
        40: 'Ipswich',
        6: 'Chelsea',
        7: 'Aston Villa',
        8: 'Everton',
        17: 'Southampton',
        20: 'Wolves',
    }
    
    return correct_teams, problematic_id_mapping

def extract_teams_from_url(url):
    """
    Extract team names from match URL to verify correct team assignment.
    """
    if not url or '/matches/' not in url:
        return None, None
        
    try:
        # Extract match part: /matches/team1-vs-team2/id
        match_part = url.split('/matches/')[1].split('/')[0].split('#')[0]
        teams = match_part.split('-vs-')
        
        if len(teams) == 2:
            # Clean team names
            home_team = teams[0].replace('-', ' ').title()
            away_team = teams[1].replace('-', ' ').title()
            
            # Normalize common variations
            normalizations = {
                'Afc Bournemouth': 'Bournemouth',
                'Brighton Hove Albion': 'Brighton', 
                'Manchester United': 'Man Utd',
                'Manchester City': 'Man City',
                'Tottenham Hotspur': 'Spurs',
                'Nottingham Forest': "Nott'm Forest",
                'Wolverhampton Wanderers': 'Wolves',
                'West Ham United': 'West Ham',
                'Newcastle United': 'Newcastle',
                'Leeds United': 'Leeds'
            }
            
            home_team = normalizations.get(home_team, home_team)
            away_team = normalizations.get(away_team, away_team)
            
            return home_team, away_team
    except:
        pass
    
    return None, None

# Enhanced function to fix team ID mapping and update matches
def fix_team_mapping_and_update_matches(season_path, season_name):
    """
    Fixes team ID mapping issues and processes matches by gameweek with correct team assignments.
    """
    print(f"\n🔧 FIXING TEAM MAPPING FOR {season_name.upper()}")
    print("=" * 60)
    
    matches_path = os.path.join(season_path, 'matches', 'matches.csv')
    if not os.path.exists(matches_path):
        print(f"❌ Matches file not found: {matches_path}")
        return None
        
    matches_df = pd.read_csv(matches_path)
    print(f"📊 Loaded {len(matches_df)} matches from {season_name}")
    
    # Get team mappings
    correct_teams, problematic_id_mapping = create_team_id_mapping()
    
    # Create reverse mapping (team name to correct ID)
    team_name_to_id = {v: k for k, v in correct_teams.items()}
    
    # Track corrections made
    corrections_made = 0
    url_matches = 0
    
    # Add columns for corrected team info
    matches_df['home_team_corrected'] = matches_df['home_team']
    matches_df['away_team_corrected'] = matches_df['away_team']
    matches_df['home_team_name'] = 'Unknown'
    matches_df['away_team_name'] = 'Unknown'
    matches_df['correction_applied'] = False
    
    # Process each match
    for idx, row in matches_df.iterrows():
        home_id = row['home_team']
        away_id = row['away_team']
        url = row.get('match_url', '')
        
        # Extract teams from URL
        url_home, url_away = extract_teams_from_url(url)
        
        if url_home and url_away:
            url_matches += 1
            
            # Map URL teams to correct IDs
            correct_home_id = team_name_to_id.get(url_home)
            correct_away_id = team_name_to_id.get(url_away)
            
            if correct_home_id and correct_away_id:
                # Apply correction
                matches_df.at[idx, 'home_team_corrected'] = correct_home_id
                matches_df.at[idx, 'away_team_corrected'] = correct_away_id
                matches_df.at[idx, 'home_team_name'] = url_home
                matches_df.at[idx, 'away_team_name'] = url_away
                matches_df.at[idx, 'correction_applied'] = True
                corrections_made += 1
                
                if home_id != correct_home_id or away_id != correct_away_id:
                    print(f"✓ Fixed Match {idx+1}: {url_home} vs {url_away}")
                    print(f"  Old IDs: {home_id} vs {away_id} → New IDs: {correct_home_id} vs {correct_away_id}")
        
        # Fallback: use problematic mapping if URL parsing failed
        if not matches_df.at[idx, 'correction_applied']:
            home_name = problematic_id_mapping.get(home_id) or correct_teams.get(home_id, f'Unknown_ID_{home_id}')
            away_name = problematic_id_mapping.get(away_id) or correct_teams.get(away_id, f'Unknown_ID_{away_id}')
            
            matches_df.at[idx, 'home_team_name'] = home_name
            matches_df.at[idx, 'away_team_name'] = away_name
    
    print(f"\n📈 CORRECTION SUMMARY:")
    print(f"   • Total matches: {len(matches_df)}")
    print(f"   • URLs processed: {url_matches}")
    print(f"   • Corrections applied: {corrections_made}")
    print(f"   • Success rate: {corrections_made/len(matches_df)*100:.1f}%")
    
    # Save corrected matches
    corrected_path = matches_path.replace('.csv', '_corrected.csv')
    matches_df.to_csv(corrected_path, index=False)
    print(f"💾 Saved corrected matches to: {corrected_path}")
    
    return matches_df

# Update matches for all gameweeks (enhanced version)
def update_matches_by_gameweek(season_path, matches_df=None, use_corrected=True):
    """
    Processes all gameweeks in matches_df and saves matches per gameweek.
    Enhanced to use corrected team mappings.
    """
    if matches_df is None:
        matches_path = os.path.join(season_path, 'matches', 'matches.csv')
        matches_df = pd.read_csv(matches_path)
    
    gw_base_path = os.path.join(season_path, 'matches')
    create_directory(gw_base_path)

    # Use corrected team IDs if available
    if use_corrected and 'home_team_corrected' in matches_df.columns:
        matches_df_output = matches_df.copy()
        matches_df_output['home_team'] = matches_df_output['home_team_corrected']
        matches_df_output['away_team'] = matches_df_output['away_team_corrected']
        print("📊 Using corrected team IDs for gameweek files")
    else:
        matches_df_output = matches_df.copy()

    for gw in matches_df['gameweek'].unique():
        gw_path = os.path.join(gw_base_path, f'GW{gw}')
        create_directory(gw_path)
        gw_matches = matches_df_output[matches_df_output['gameweek'] == gw]
        gw_matches.to_csv(os.path.join(gw_path, 'matches.csv'), index=False)
        print(f"✓ Updated GW{gw} with {len(gw_matches)} matches")

    return matches_df_output

# Update player match stats for all gameweeks
def update_player_match_stats(season_path, matches_df):
    """
    Processes all gameweeks in player match stats, mapping match_id to gameweek.
    """
    stats_path = os.path.join(season_path, 'playermatchstats', 'playermatchstats.csv')
    stats_df = pd.read_csv(stats_path)
    gw_base_path = os.path.join(season_path, 'playermatchstats', 'gameweeks')
    create_directory(gw_base_path)

    # Map match_id to gameweek using matches_df
    match_to_gw = matches_df.set_index('match_id')['gameweek'].to_dict()
    stats_df['gameweek'] = stats_df['match_id'].map(match_to_gw)

    # Warn about any stats that don’t map to a gameweek
    if stats_df['gameweek'].isna().any():
        print(f"Warning: {stats_df['gameweek'].isna().sum()} player stats have no matching gameweek.")

    total_stats = len(stats_df)
    print(f"Found {total_stats} player match stats across all gameweeks")

    for gw in stats_df['gameweek'].unique():
        if pd.isna(gw):
            continue  # Skip if gameweek is missing
        gw_path = os.path.join(gw_base_path, f'GW{gw}')
        create_directory(gw_path)
        gw_stats = stats_df[stats_df['gameweek'] == gw]
        gw_stats.to_csv(os.path.join(gw_path, 'playermatchstats.csv'), index=False)
        print(f"Updated GW{gw} with {len(gw_stats)} player match stats")

# Main execution function (enhanced)
def main():
    """
    Enhanced main function that can handle multiple seasons and fix team mapping issues.
    """
    print("🚀 ENHANCED CSV FIXER - TEAM MAPPING CORRECTION")
    print("=" * 70)
    
    # Process both seasons
    seasons = ["2024-2025", "2025-2026"]
    
    for season in seasons:
        season_path = os.path.join('data', season)
        
        if not os.path.exists(season_path):
            print(f"⚠️ Season directory not found: {season_path}")
            continue
            
        print(f"\n🎯 PROCESSING SEASON: {season}")
        print("-" * 50)
        
        # Fix team mapping and get corrected matches
        corrected_matches_df = fix_team_mapping_and_update_matches(season_path, season)
        
        if corrected_matches_df is not None:
            # Update matches by gameweek with corrected data
            print(f"\n📁 Updating gameweek files for {season}...")
            update_matches_by_gameweek(season_path, corrected_matches_df, use_corrected=True)
            
            # Update player match stats if available
            print(f"\n👥 Updating player match stats for {season}...")
            try:
                update_player_match_stats(season_path, corrected_matches_df)
            except Exception as e:
                print(f"⚠️ Could not update player stats for {season}: {e}")
        
        print(f"\n✅ {season} processing complete!")
    
    print(f"\n🎉 ALL SEASONS PROCESSED SUCCESSFULLY!")
    print("=" * 70)
    print("📋 SUMMARY:")
    print("• Fixed team ID mapping issues using URL analysis")
    print("• Created corrected matches files (*_corrected.csv)")
    print("• Updated gameweek-specific match files")
    print("• Ready for FPL prediction analysis!")

if __name__ == "__main__":
    main()
