"""
GW13 Player Form Analysis and Prediction
Analyzes player form through GW12 and predicts top performers for GW13
"""

import pandas as pd
import os
from pathlib import Path

# Base path
BASE_PATH = Path("/Users/macbook/Dropbox/GitHub/FPL/FPL-Elo-Insights/data/2025-2026/By Gameweek")

# Team code to name mapping (from players.csv)
TEAM_CODES = {
    1: "Man Utd", 2: "Leeds", 3: "Arsenal", 4: "Newcastle", 6: "Spurs",
    7: "Aston Villa", 8: "Chelsea", 11: "Everton", 14: "Liverpool",
    17: "Nott'm Forest", 21: "West Ham", 31: "Crystal Palace", 36: "Brighton",
    39: "Wolves", 43: "Man City", 54: "Fulham", 56: "Sunderland",
    90: "Burnley", 91: "Bournemouth", 94: "Brentford"
}

def load_gameweek_stats(gw):
    """Load player stats for a specific gameweek (cumulative data)"""
    path = BASE_PATH / f"GW{gw}" / "playerstats.csv"
    if path.exists():
        return pd.read_csv(path)
    return None

def load_players():
    """Load player info"""
    path = BASE_PATH / "GW12" / "players.csv"
    return pd.read_csv(path)

def load_fixtures(gw):
    """Load fixtures for a gameweek"""
    path = BASE_PATH / f"GW{gw}" / "fixtures.csv"
    if path.exists():
        return pd.read_csv(path)
    return None

def calculate_form(gameweeks=[8, 9, 10, 11, 12]):
    """Calculate player form over specified gameweeks"""
    all_stats = []

    for gw in gameweeks:
        stats = load_gameweek_stats(gw)
        if stats is not None:
            stats['gameweek'] = gw
            all_stats.append(stats)

    if not all_stats:
        return None

    # Combine all gameweek stats
    combined = pd.concat(all_stats, ignore_index=True)

    # For each player, calculate form metrics
    # Using 'form' column from FPL which is average points over last 30 days
    # Also calculate points from recent gameweeks

    # Get the latest gameweek data for current form
    latest = combined[combined['gameweek'] == 12].copy()

    return latest

def get_gw13_fixtures():
    """Get GW13 Premier League fixtures with difficulty analysis"""
    fixtures = load_fixtures(13)
    if fixtures is None:
        return None

    # Filter Premier League only
    prem_fixtures = fixtures[fixtures['tournament'] == 'prem'].copy()

    return prem_fixtures

def analyze_form_and_predict():
    """Main analysis function"""

    print("=" * 80)
    print("FPL GW13 ANALYSIS: Players in Form & Top Predictions")
    print("=" * 80)

    # Load data
    players = load_players()
    form_stats = calculate_form()
    fixtures = get_gw13_fixtures()

    if form_stats is None:
        print("Error loading form stats")
        return

    # Merge player info with form stats
    form_stats = form_stats.merge(
        players[['player_id', 'team_code', 'position']],
        left_on='id',
        right_on='player_id',
        how='left'
    )

    form_stats['team_name'] = form_stats['team_code'].map(TEAM_CODES)

    # Ensure numeric columns
    form_stats['minutes'] = pd.to_numeric(form_stats['minutes'], errors='coerce').fillna(0)
    form_stats['form'] = pd.to_numeric(form_stats['form'], errors='coerce').fillna(0)
    form_stats['total_points'] = pd.to_numeric(form_stats['total_points'], errors='coerce').fillna(0)
    form_stats['now_cost'] = pd.to_numeric(form_stats['now_cost'], errors='coerce').fillna(0)

    # Filter out players with minimal minutes
    form_stats = form_stats[form_stats['minutes'] >= 270]  # At least 3 full games

    # Calculate value metric
    form_stats['value'] = form_stats['total_points'] / (form_stats['now_cost'] / 10)

    print("\n" + "=" * 80)
    print("GW13 PREMIER LEAGUE FIXTURES")
    print("=" * 80)

    if fixtures is not None:
        for _, match in fixtures.iterrows():
            home = TEAM_CODES.get(int(match['home_team']), f"Team {match['home_team']}")
            away = TEAM_CODES.get(int(match['away_team']), f"Team {match['away_team']}")
            home_elo = match.get('home_team_elo', 'N/A')
            away_elo = match.get('away_team_elo', 'N/A')

            if pd.notna(home_elo) and pd.notna(away_elo):
                print(f"{home} ({home_elo:.0f}) vs {away} ({away_elo:.0f})")
            else:
                print(f"{home} vs {away}")

    # === TOP PLAYERS BY POSITION ===

    print("\n" + "=" * 80)
    print("TOP PLAYERS IN FORM BY POSITION (Based on FPL Form Rating)")
    print("=" * 80)

    for pos in ['Goalkeeper', 'Defender', 'Midfielder', 'Forward']:
        print(f"\n--- {pos.upper()}S ---")
        pos_players = form_stats[form_stats['position'] == pos].nlargest(10, 'form')

        for _, player in pos_players.iterrows():
            print(f"{player['web_name']:20s} | {player['team_name']:15s} | "
                  f"Form: {player['form']:.1f} | Points: {int(player['total_points']):3d} | "
                  f"Price: {player['now_cost']/10:.1f}m")

    # === TOP OVERALL IN FORM ===

    print("\n" + "=" * 80)
    print("TOP 25 PLAYERS IN FORM (All Positions)")
    print("=" * 80)

    top_form = form_stats.nlargest(25, 'form')

    for i, (_, player) in enumerate(top_form.iterrows(), 1):
        print(f"{i:2d}. {player['web_name']:20s} | {player['position']:10s} | "
              f"{player['team_name']:15s} | Form: {player['form']:.1f} | "
              f"Points: {int(player['total_points']):3d} | Price: {player['now_cost']/10:.1f}m")

    # === GW13 PREDICTIONS ===

    print("\n" + "=" * 80)
    print("GW13 TOP PERFORMERS PREDICTION")
    print("=" * 80)

    # Create prediction score based on:
    # 1. Form (40%)
    # 2. Expected goals/assists per 90 (30%)
    # 3. Fixture difficulty based on opponent Elo (30%)

    # Get fixture info for each team
    fixture_info = {}
    if fixtures is not None:
        for _, match in fixtures.iterrows():
            home_team = int(match['home_team'])
            away_team = int(match['away_team'])
            home_elo = match.get('home_team_elo', 1700)
            away_elo = match.get('away_team_elo', 1700)

            # Home team faces away team (difficulty based on opponent Elo)
            fixture_info[home_team] = {'opponent': away_team, 'home': True, 'opp_elo': away_elo if pd.notna(away_elo) else 1700}
            fixture_info[away_team] = {'opponent': home_team, 'home': False, 'opp_elo': home_elo if pd.notna(home_elo) else 1700}

    # Add fixture difficulty score (lower opponent Elo = easier fixture)
    form_stats['fixture_info'] = form_stats['team_code'].map(fixture_info)
    form_stats['has_fixture'] = form_stats['fixture_info'].notna()

    # Filter to players with GW13 fixtures
    gw13_players = form_stats[form_stats['has_fixture']].copy()

    def get_opp_elo(fixture_info):
        if pd.isna(fixture_info) or fixture_info is None:
            return 1700
        return fixture_info.get('opp_elo', 1700)

    def is_home(fixture_info):
        if pd.isna(fixture_info) or fixture_info is None:
            return False
        return fixture_info.get('home', False)

    gw13_players['opp_elo'] = gw13_players['fixture_info'].apply(get_opp_elo)
    gw13_players['is_home'] = gw13_players['fixture_info'].apply(is_home)

    # Ensure numeric types
    gw13_players['form'] = pd.to_numeric(gw13_players['form'], errors='coerce').fillna(0)
    gw13_players['opp_elo'] = pd.to_numeric(gw13_players['opp_elo'], errors='coerce').fillna(1700)

    # Normalize metrics for prediction score
    form_min = gw13_players['form'].min()
    form_max = gw13_players['form'].max()
    if form_max != form_min:
        gw13_players['form_norm'] = (gw13_players['form'] - form_min) / (form_max - form_min)
    else:
        gw13_players['form_norm'] = 0.5

    # Fixture difficulty: easier = lower Elo (invert so lower Elo = higher score)
    max_elo = gw13_players['opp_elo'].max()
    min_elo = gw13_players['opp_elo'].min()
    if max_elo != min_elo:
        gw13_players['fixture_score'] = (max_elo - gw13_players['opp_elo']) / (max_elo - min_elo)
    else:
        gw13_players['fixture_score'] = 0.5

    # Home advantage bonus
    gw13_players['home_bonus'] = gw13_players['is_home'].astype(float) * 0.1

    # Expected goals involvement
    gw13_players['expected_goal_involvements_per_90'] = pd.to_numeric(
        gw13_players['expected_goal_involvements_per_90'], errors='coerce'
    ).fillna(0)

    egi_max = gw13_players['expected_goal_involvements_per_90'].max()
    if egi_max > 0:
        gw13_players['egi_norm'] = gw13_players['expected_goal_involvements_per_90'] / egi_max
    else:
        gw13_players['egi_norm'] = 0.0

    # Final prediction score - ensure numeric
    gw13_players['prediction_score'] = (
        gw13_players['form_norm'].astype(float) * 0.4 +
        gw13_players['egi_norm'].astype(float) * 0.3 +
        gw13_players['fixture_score'].astype(float) * 0.2 +
        gw13_players['home_bonus'].astype(float) * 0.1
    )

    # Top predictions by position
    print("\n--- GW13 TOP PICKS BY POSITION ---\n")

    for pos in ['Goalkeeper', 'Defender', 'Midfielder', 'Forward']:
        print(f"\n{pos.upper()}S:")
        pos_preds = gw13_players[gw13_players['position'] == pos].nlargest(5, 'prediction_score')

        for rank, (_, player) in enumerate(pos_preds.iterrows(), 1):
            opp_team = TEAM_CODES.get(
                player['fixture_info']['opponent'] if player['fixture_info'] else 0,
                "Unknown"
            )
            venue = "H" if player['is_home'] else "A"

            print(f"  {rank}. {player['web_name']:18s} | {player['team_name']:12s} | "
                  f"vs {opp_team:12s} ({venue}) | Form: {player['form']:.1f} | "
                  f"Score: {player['prediction_score']:.2f}")

    # Overall top picks
    print("\n" + "-" * 80)
    print("OVERALL TOP 15 GW13 PREDICTIONS")
    print("-" * 80)

    top_preds = gw13_players.nlargest(15, 'prediction_score')

    for rank, (_, player) in enumerate(top_preds.iterrows(), 1):
        opp_team = TEAM_CODES.get(
            player['fixture_info']['opponent'] if player['fixture_info'] else 0,
            "Unknown"
        )
        venue = "H" if player['is_home'] else "A"

        print(f"{rank:2d}. {player['web_name']:18s} | {player['position']:10s} | "
              f"{player['team_name']:12s} | vs {opp_team:12s} ({venue}) | "
              f"Form: {player['form']:.1f} | Price: {player['now_cost']/10:.1f}m | "
              f"Score: {player['prediction_score']:.2f}")

    # === DIFFERENTIAL PICKS (Low ownership, high form) ===

    print("\n" + "=" * 80)
    print("DIFFERENTIAL PICKS (Low Ownership, High Form)")
    print("=" * 80)

    gw13_players['selected_by_percent'] = pd.to_numeric(
        gw13_players['selected_by_percent'], errors='coerce'
    ).fillna(100)

    differentials = gw13_players[
        (gw13_players['selected_by_percent'] < 10) &
        (gw13_players['form'] >= 5)
    ].nlargest(10, 'prediction_score')

    for rank, (_, player) in enumerate(differentials.iterrows(), 1):
        opp_team = TEAM_CODES.get(
            player['fixture_info']['opponent'] if player['fixture_info'] else 0,
            "Unknown"
        )
        venue = "H" if player['is_home'] else "A"

        print(f"{rank:2d}. {player['web_name']:18s} | {player['position']:10s} | "
              f"{player['team_name']:12s} | vs {opp_team:12s} ({venue}) | "
              f"Form: {player['form']:.1f} | Owned: {player['selected_by_percent']:.1f}% | "
              f"Score: {player['prediction_score']:.2f}")

    # === CAPTAIN PICKS ===

    print("\n" + "=" * 80)
    print("CAPTAIN PICKS FOR GW13")
    print("=" * 80)

    # For captain, prioritize attacking players with high form and good fixtures
    captain_candidates = gw13_players[
        gw13_players['position'].isin(['Midfielder', 'Forward'])
    ].nlargest(10, 'prediction_score')

    print("\nTop Captain Options:")
    for rank, (_, player) in enumerate(captain_candidates.iterrows(), 1):
        opp_team = TEAM_CODES.get(
            player['fixture_info']['opponent'] if player['fixture_info'] else 0,
            "Unknown"
        )
        venue = "H" if player['is_home'] else "A"

        print(f"{rank:2d}. {player['web_name']:18s} | {player['team_name']:12s} | "
              f"vs {opp_team:12s} ({venue}) | Form: {player['form']:.1f} | "
              f"xGI/90: {player['expected_goal_involvements_per_90']:.2f} | "
              f"Score: {player['prediction_score']:.2f}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    analyze_form_and_predict()
