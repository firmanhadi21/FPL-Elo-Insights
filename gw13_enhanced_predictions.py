"""
GW13 Enhanced Predictions
Comprehensive prediction model incorporating:
- Player Form (recent performance)
- Fixture Difficulty Rating (FDR) based on Elo ratings
- Home/Away advantage
- Expected Goal Involvement
- Position-specific factors
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_PATH = Path("/Users/macbook/Dropbox/GitHub/FPL/FPL-Elo-Insights/data/2025-2026/By Gameweek")

# Team code to name mapping
TEAM_CODES = {
    1: "Man Utd", 2: "Leeds", 3: "Arsenal", 4: "Newcastle", 6: "Spurs",
    7: "Aston Villa", 8: "Chelsea", 11: "Everton", 14: "Liverpool",
    17: "Nott'm Forest", 21: "West Ham", 31: "Crystal Palace", 36: "Brighton",
    39: "Wolves", 43: "Man City", 54: "Fulham", 56: "Sunderland",
    90: "Burnley", 91: "Bournemouth", 94: "Brentford"
}

# Elo-based FDR calculation thresholds
def calculate_fdr(opponent_elo, is_home):
    """
    Calculate Fixture Difficulty Rating (1-5) based on opponent Elo
    Adjusted for home/away advantage (~65 Elo points home advantage)
    """
    # Adjust for home/away (home team gets ~65 Elo advantage)
    adjusted_elo = opponent_elo - 65 if is_home else opponent_elo + 65

    # FDR thresholds based on Elo distribution
    if adjusted_elo >= 2000:
        return 5  # Very difficult
    elif adjusted_elo >= 1900:
        return 4  # Difficult
    elif adjusted_elo >= 1800:
        return 3  # Medium
    elif adjusted_elo >= 1700:
        return 2  # Easy
    else:
        return 1  # Very easy


def load_data():
    """Load all required data"""
    stats = pd.read_csv(BASE_PATH / "GW12" / "playerstats.csv")
    players = pd.read_csv(BASE_PATH / "GW12" / "players.csv")
    fixtures = pd.read_csv(BASE_PATH / "GW13" / "fixtures.csv")

    return stats, players, fixtures


def analyze_fixtures(fixtures):
    """Analyze GW13 fixtures with FDR"""
    prem_fixtures = fixtures[fixtures['tournament'] == 'prem'].copy()

    fixture_analysis = []
    for _, match in prem_fixtures.iterrows():
        home_team = int(match['home_team'])
        away_team = int(match['away_team'])
        home_elo = float(match['home_team_elo']) if pd.notna(match['home_team_elo']) else 1750
        away_elo = float(match['away_team_elo']) if pd.notna(match['away_team_elo']) else 1750

        # FDR for home team (facing away team)
        home_fdr = calculate_fdr(away_elo, is_home=True)
        # FDR for away team (facing home team)
        away_fdr = calculate_fdr(home_elo, is_home=False)

        fixture_analysis.append({
            'home_team': home_team,
            'away_team': away_team,
            'home_team_name': TEAM_CODES.get(home_team, f"Team {home_team}"),
            'away_team_name': TEAM_CODES.get(away_team, f"Team {away_team}"),
            'home_elo': home_elo,
            'away_elo': away_elo,
            'home_fdr': home_fdr,  # Difficulty for home team
            'away_fdr': away_fdr,  # Difficulty for away team
        })

    return pd.DataFrame(fixture_analysis)


def calculate_prediction_score(row, fixture_info):
    """
    Calculate comprehensive prediction score for a player

    Weights:
    - Form: 35% (recent performance is crucial)
    - FDR: 30% (fixture difficulty matters a lot)
    - Home/Away: 15% (home advantage is real)
    - xGI per 90: 20% (underlying stats)
    """

    form = float(row['form']) if pd.notna(row['form']) else 0
    xgi_90 = float(row['expected_goal_involvements_per_90']) if pd.notna(row['expected_goal_involvements_per_90']) else 0

    if fixture_info is None:
        return 0

    fdr = fixture_info['fdr']
    is_home = fixture_info['is_home']

    # Normalize form (0-10 scale, cap at 10)
    form_score = min(form, 10) / 10

    # FDR score (lower FDR = higher score)
    # FDR 1 = 1.0, FDR 2 = 0.8, FDR 3 = 0.6, FDR 4 = 0.4, FDR 5 = 0.2
    fdr_score = (6 - fdr) / 5

    # Home advantage score
    home_score = 1.0 if is_home else 0.6

    # xGI score (normalize to 0-1, assuming max ~1.5 xGI/90 for top players)
    xgi_score = min(xgi_90, 1.5) / 1.5

    # Position-specific adjustments
    position = row.get('position', 'Midfielder')

    # Attackers benefit more from easy fixtures
    if position == 'Forward':
        fdr_weight = 0.35
        form_weight = 0.30
        xgi_weight = 0.25
        home_weight = 0.10
    elif position == 'Midfielder':
        fdr_weight = 0.30
        form_weight = 0.35
        xgi_weight = 0.20
        home_weight = 0.15
    elif position == 'Defender':
        # Defenders benefit more from home (clean sheets)
        fdr_weight = 0.25
        form_weight = 0.35
        xgi_weight = 0.15
        home_weight = 0.25
    else:  # Goalkeeper
        # GKs heavily depend on team performance and home advantage
        fdr_weight = 0.30
        form_weight = 0.30
        xgi_weight = 0.05
        home_weight = 0.35

    # Calculate final score
    prediction_score = (
        form_score * form_weight +
        fdr_score * fdr_weight +
        home_score * home_weight +
        xgi_score * xgi_weight
    )

    return prediction_score


def main():
    print("=" * 90)
    print("GW13 ENHANCED PREDICTIONS")
    print("Incorporating Form, FDR (Elo-based), Home/Away Advantage, and xGI")
    print("=" * 90)

    # Load data
    stats, players, fixtures = load_data()

    # Merge player info
    stats = stats.merge(
        players[['player_id', 'team_code', 'position']],
        left_on='id',
        right_on='player_id',
        how='left'
    )

    # Ensure numeric columns
    stats['form'] = pd.to_numeric(stats['form'], errors='coerce').fillna(0)
    stats['minutes'] = pd.to_numeric(stats['minutes'], errors='coerce').fillna(0)
    stats['total_points'] = pd.to_numeric(stats['total_points'], errors='coerce').fillna(0)
    stats['now_cost'] = pd.to_numeric(stats['now_cost'], errors='coerce').fillna(0)
    stats['expected_goal_involvements_per_90'] = pd.to_numeric(
        stats['expected_goal_involvements_per_90'], errors='coerce'
    ).fillna(0)
    stats['selected_by_percent'] = pd.to_numeric(
        stats['selected_by_percent'], errors='coerce'
    ).fillna(0)

    # Filter players with meaningful minutes (at least 270 = 3 full games)
    stats = stats[stats['minutes'] >= 270].copy()
    stats['team_name'] = stats['team_code'].map(TEAM_CODES)

    # Analyze fixtures
    fixture_df = analyze_fixtures(fixtures)

    # Print fixture analysis
    print("\n" + "=" * 90)
    print("GW13 FIXTURE DIFFICULTY ANALYSIS")
    print("=" * 90)
    print(f"\n{'Match':<45} {'Home FDR':<10} {'Away FDR':<10} {'Elo Diff':<10}")
    print("-" * 75)

    # Sort by home FDR (easiest first)
    fixture_df_sorted = fixture_df.sort_values('home_fdr')

    for _, match in fixture_df_sorted.iterrows():
        match_str = f"{match['home_team_name']} vs {match['away_team_name']}"
        elo_diff = match['home_elo'] - match['away_elo']
        print(f"{match_str:<45} {match['home_fdr']:<10} {match['away_fdr']:<10} {elo_diff:+.0f}")

    # Create fixture lookup for each team
    team_fixtures = {}
    for _, match in fixture_df.iterrows():
        team_fixtures[match['home_team']] = {
            'opponent': match['away_team'],
            'opponent_name': match['away_team_name'],
            'opponent_elo': match['away_elo'],
            'is_home': True,
            'fdr': match['home_fdr']
        }
        team_fixtures[match['away_team']] = {
            'opponent': match['home_team'],
            'opponent_name': match['home_team_name'],
            'opponent_elo': match['home_elo'],
            'is_home': False,
            'fdr': match['away_fdr']
        }

    # Add fixture info to players
    stats['fixture_info'] = stats['team_code'].map(team_fixtures)
    stats = stats[stats['fixture_info'].notna()].copy()

    # Calculate prediction scores
    stats['prediction_score'] = stats.apply(
        lambda row: calculate_prediction_score(row, row['fixture_info']),
        axis=1
    )

    # Extract fixture details for display
    stats['opponent'] = stats['fixture_info'].apply(lambda x: x['opponent_name'] if x else '')
    stats['is_home'] = stats['fixture_info'].apply(lambda x: x['is_home'] if x else False)
    stats['fdr'] = stats['fixture_info'].apply(lambda x: x['fdr'] if x else 3)
    stats['venue'] = stats['is_home'].apply(lambda x: 'H' if x else 'A')

    # === PREDICTIONS BY POSITION ===
    print("\n" + "=" * 90)
    print("TOP GW13 PREDICTIONS BY POSITION")
    print("=" * 90)

    for pos in ['Goalkeeper', 'Defender', 'Midfielder', 'Forward']:
        print(f"\n{'─' * 90}")
        print(f"  {pos.upper()}S")
        print(f"{'─' * 90}")
        print(f"{'Rank':<5} {'Player':<18} {'Team':<14} {'Opponent':<14} {'H/A':<4} {'FDR':<4} {'Form':<6} {'xGI/90':<7} {'Score':<6}")
        print("-" * 90)

        pos_players = stats[stats['position'] == pos].nlargest(10, 'prediction_score')

        for rank, (_, player) in enumerate(pos_players.iterrows(), 1):
            print(f"{rank:<5} {player['web_name']:<18} {player['team_name']:<14} "
                  f"{player['opponent']:<14} {player['venue']:<4} {int(player['fdr']):<4} "
                  f"{player['form']:<6.1f} {player['expected_goal_involvements_per_90']:<7.2f} "
                  f"{player['prediction_score']:<6.3f}")

    # === TOP 20 OVERALL ===
    print("\n" + "=" * 90)
    print("TOP 20 OVERALL GW13 PREDICTIONS")
    print("=" * 90)
    print(f"\n{'Rank':<5} {'Player':<18} {'Pos':<10} {'Team':<14} {'Opponent':<14} {'H/A':<4} {'FDR':<4} {'Form':<6} {'Price':<6} {'Score':<6}")
    print("-" * 95)

    top_overall = stats.nlargest(20, 'prediction_score')

    for rank, (_, player) in enumerate(top_overall.iterrows(), 1):
        price = player['now_cost'] / 10
        print(f"{rank:<5} {player['web_name']:<18} {player['position']:<10} "
              f"{player['team_name']:<14} {player['opponent']:<14} {player['venue']:<4} "
              f"{int(player['fdr']):<4} {player['form']:<6.1f} {price:<6.1f} "
              f"{player['prediction_score']:<6.3f}")

    # === BEST BY FDR (Easy Fixtures) ===
    print("\n" + "=" * 90)
    print("BEST PICKS FROM EASY FIXTURES (FDR 1-2)")
    print("=" * 90)

    easy_fixtures = stats[stats['fdr'] <= 2].nlargest(15, 'prediction_score')
    print(f"\n{'Rank':<5} {'Player':<18} {'Pos':<10} {'Team':<14} {'Opponent':<14} {'H/A':<4} {'FDR':<4} {'Form':<6} {'Score':<6}")
    print("-" * 90)

    for rank, (_, player) in enumerate(easy_fixtures.iterrows(), 1):
        print(f"{rank:<5} {player['web_name']:<18} {player['position']:<10} "
              f"{player['team_name']:<14} {player['opponent']:<14} {player['venue']:<4} "
              f"{int(player['fdr']):<4} {player['form']:<6.1f} {player['prediction_score']:<6.3f}")

    # === HOME ADVANTAGE PICKS ===
    print("\n" + "=" * 90)
    print("BEST HOME PICKS (Taking Advantage of Home Ground)")
    print("=" * 90)

    home_picks = stats[stats['is_home']].nlargest(15, 'prediction_score')
    print(f"\n{'Rank':<5} {'Player':<18} {'Pos':<10} {'Team':<14} {'Opponent':<14} {'FDR':<4} {'Form':<6} {'Score':<6}")
    print("-" * 85)

    for rank, (_, player) in enumerate(home_picks.iterrows(), 1):
        print(f"{rank:<5} {player['web_name']:<18} {player['position']:<10} "
              f"{player['team_name']:<14} {player['opponent']:<14} "
              f"{int(player['fdr']):<4} {player['form']:<6.1f} {player['prediction_score']:<6.3f}")

    # === CAPTAIN PICKS ===
    print("\n" + "=" * 90)
    print("CAPTAIN PICKS FOR GW13")
    print("=" * 90)

    # For captain, focus on attackers with easy fixtures
    captain_candidates = stats[
        (stats['position'].isin(['Midfielder', 'Forward'])) &
        (stats['fdr'] <= 3)  # Medium or easier fixtures
    ].nlargest(10, 'prediction_score')

    print(f"\n{'Rank':<5} {'Player':<18} {'Team':<14} {'vs':<14} {'H/A':<4} {'FDR':<4} {'Form':<6} {'xGI/90':<7} {'Score':<6} {'Reason'}")
    print("-" * 110)

    for rank, (_, player) in enumerate(captain_candidates.iterrows(), 1):
        # Generate reason
        reasons = []
        if player['fdr'] <= 2:
            reasons.append("Easy fixture")
        if player['is_home']:
            reasons.append("Home")
        if player['form'] >= 7:
            reasons.append("Hot form")
        if player['expected_goal_involvements_per_90'] >= 0.5:
            reasons.append("High xGI")

        reason = ", ".join(reasons) if reasons else "Solid pick"

        print(f"{rank:<5} {player['web_name']:<18} {player['team_name']:<14} "
              f"{player['opponent']:<14} {player['venue']:<4} {int(player['fdr']):<4} "
              f"{player['form']:<6.1f} {player['expected_goal_involvements_per_90']:<7.2f} "
              f"{player['prediction_score']:<6.3f} {reason}")

    # === DIFFERENTIAL PICKS ===
    print("\n" + "=" * 90)
    print("DIFFERENTIAL PICKS (<10% ownership, high potential)")
    print("=" * 90)

    differentials = stats[
        (stats['selected_by_percent'] < 10) &
        (stats['form'] >= 4) &
        (stats['fdr'] <= 3)
    ].nlargest(10, 'prediction_score')

    print(f"\n{'Rank':<5} {'Player':<18} {'Pos':<10} {'Team':<14} {'Opponent':<14} {'Own%':<6} {'FDR':<4} {'Form':<6} {'Score':<6}")
    print("-" * 95)

    for rank, (_, player) in enumerate(differentials.iterrows(), 1):
        print(f"{rank:<5} {player['web_name']:<18} {player['position']:<10} "
              f"{player['team_name']:<14} {player['opponent']:<14} "
              f"{player['selected_by_percent']:<6.1f} {int(player['fdr']):<4} "
              f"{player['form']:<6.1f} {player['prediction_score']:<6.3f}")

    # === VALUE PICKS ===
    print("\n" + "=" * 90)
    print("VALUE PICKS (Best prediction score relative to price)")
    print("=" * 90)

    stats['value_score'] = stats['prediction_score'] / (stats['now_cost'] / 100)
    value_picks = stats[stats['form'] >= 3].nlargest(10, 'value_score')

    print(f"\n{'Rank':<5} {'Player':<18} {'Pos':<10} {'Team':<14} {'Price':<6} {'FDR':<4} {'Form':<6} {'Score':<6}")
    print("-" * 80)

    for rank, (_, player) in enumerate(value_picks.iterrows(), 1):
        price = player['now_cost'] / 10
        print(f"{rank:<5} {player['web_name']:<18} {player['position']:<10} "
              f"{player['team_name']:<14} {price:<6.1f} {int(player['fdr']):<4} "
              f"{player['form']:<6.1f} {player['prediction_score']:<6.3f}")

    # === SUMMARY ===
    print("\n" + "=" * 90)
    print("GW13 SUMMARY & RECOMMENDATIONS")
    print("=" * 90)

    print("\n🎯 TOP CAPTAIN PICK:")
    top_captain = captain_candidates.iloc[0]
    print(f"   {top_captain['web_name']} ({top_captain['team_name']}) vs {top_captain['opponent']} ({top_captain['venue']})")
    print(f"   FDR: {int(top_captain['fdr'])} | Form: {top_captain['form']:.1f} | xGI/90: {top_captain['expected_goal_involvements_per_90']:.2f}")

    print("\n🏠 BEST HOME PICKS:")
    for i, (_, p) in enumerate(home_picks.head(3).iterrows(), 1):
        print(f"   {i}. {p['web_name']} ({p['team_name']}) - FDR {int(p['fdr'])}, Form {p['form']:.1f}")

    print("\n📊 EASIEST FIXTURES TO TARGET:")
    easy_teams = fixture_df[fixture_df['home_fdr'] <= 2][['home_team_name', 'away_team_name', 'home_fdr']].head(3)
    for _, match in easy_teams.iterrows():
        print(f"   {match['home_team_name']} vs {match['away_team_name']} (FDR {int(match['home_fdr'])} for home)")

    print("\n⚠️ FIXTURES TO AVOID:")
    hard_fixtures = fixture_df[fixture_df['home_fdr'] >= 4][['home_team_name', 'away_team_name', 'home_fdr']]
    for _, match in hard_fixtures.iterrows():
        print(f"   {match['home_team_name']} vs {match['away_team_name']} (FDR {int(match['home_fdr'])} for home)")

    print("\n" + "=" * 90)
    print("ANALYSIS COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()
