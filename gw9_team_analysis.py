#!/usr/bin/env python3
"""
GW9 FPL Team Management Analysis
Based on latest data sync from October 23, 2025
"""

import pandas as pd
import numpy as np

# Load the data
players_df = pd.read_csv('data/2025-2026/By Gameweek/GW9/playerstats.csv')
teams_df = pd.read_csv('data/2025-2026/By Gameweek/GW9/teams.csv')
fixtures_df = pd.read_csv('data/2025-2026/By Gameweek/GW9/fixtures.csv')

# Filter for premier league fixtures only
prem_fixtures = fixtures_df[fixtures_df['tournament'] == 'prem'].copy()

print("=== GW9 FPL TEAM MANAGEMENT ANALYSIS ===")
print(f"Analysis Date: October 23, 2025")
print(f"Data Source: Latest sync from upstream repository")
print()

# GW9 Premier League Fixtures Analysis
print("📅 GW9 PREMIER LEAGUE FIXTURES:")
print("=" * 50)

# Create team name mapping for fixtures
team_mapping = dict(zip(teams_df['id'], teams_df['name']))

prem_fixtures_clean = prem_fixtures.copy()
for col in ['home_team', 'away_team']:
    if col in prem_fixtures_clean.columns:
        prem_fixtures_clean[col + '_name'] = prem_fixtures_clean[col].map(team_mapping)

fixture_list = []
for _, fixture in prem_fixtures_clean.iterrows():
    if pd.notna(fixture.get('home_team')) and pd.notna(fixture.get('away_team')):
        home_name = team_mapping.get(fixture['home_team'], f"Team {fixture['home_team']}")
        away_name = team_mapping.get(fixture['away_team'], f"Team {fixture['away_team']}")
        home_elo = fixture.get('home_team_elo', 'N/A')
        away_elo = fixture.get('away_team_elo', 'N/A')
        fixture_list.append(f"{home_name} (ELO: {home_elo}) vs {away_name} (ELO: {away_elo})")

for fixture in fixture_list:
    print(f"• {fixture}")

print()

# Player Performance Analysis
print("🌟 TOP PERFORMERS BY POSITION:")
print("=" * 50)

# Filter players with significant playing time
active_players = players_df[
    (players_df['minutes'] >= 90) &
    (players_df['status'] == 'a')  # Available players only
].copy()

# Calculate value metrics
active_players['ppg'] = active_players['total_points'] / np.maximum(active_players['starts'], 1)
active_players['value_score'] = active_players['total_points'] / active_players['now_cost']
active_players['form_score'] = active_players['form'] / active_players['now_cost']

# Position mapping
position_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}

# Top performers by position
for pos_id, pos_name in position_map.items():
    print(f"\n{pos_name} - Top 5 by Total Points:")
    pos_players = active_players[active_players['position'] == pos_id]
    top_players = pos_players.nlargest(5, 'total_points')[
        ['web_name', 'now_cost', 'total_points', 'ppg', 'form', 'value_score', 'selected_by_percent']
    ]

    for _, player in top_players.iterrows():
        print(f"  {player['web_name']:12} | £{player['now_cost']:.1f}m | {player['total_points']:2.0f}pts | "
              f"PPG: {player['ppg']:.1f} | Form: {player['form']:.1f} | Owned: {player['selected_by_percent']:.1f}%")

print()

# Best Value Picks
print("💰 BEST VALUE PICKS (Points per £m):")
print("=" * 50)

for pos_id, pos_name in position_map.items():
    print(f"\n{pos_name} - Top 3 Value:")
    pos_players = active_players[active_players['position'] == pos_id]
    value_players = pos_players.nlargest(3, 'value_score')[
        ['web_name', 'now_cost', 'total_points', 'value_score', 'form']
    ]

    for _, player in value_players.iterrows():
        print(f"  {player['web_name']:12} | £{player['now_cost']:.1f}m | {player['total_points']:2.0f}pts | "
              f"Value: {player['value_score']:.1f} | Form: {player['form']:.1f}")

print()

# Fixture Difficulty Analysis
print("📊 TEAM ANALYSIS BY ELO RATINGS:")
print("=" * 50)

teams_sorted = teams_df.sort_values('elo', ascending=False)
print("Top 6 Teams by ELO:")
for _, team in teams_sorted.head(6).iterrows():
    print(f"  {team['name']:20} | ELO: {team['elo']:.0f}")

print("\nBottom 6 Teams by ELO:")
for _, team in teams_sorted.tail(6).iterrows():
    print(f"  {team['name']:20} | ELO: {team['elo']:.0f}")

print()

# Captain Candidates
print("🔥 CAPTAIN CANDIDATES:")
print("=" * 50)

captain_candidates = active_players[
    (active_players['total_points'] >= 30) &
    (active_players['minutes'] >= 300) &
    (active_players['form'] >= 3.0)
].nlargest(8, 'total_points')[
    ['web_name', 'now_cost', 'total_points', 'ppg', 'form', 'selected_by_percent']
]

for _, player in captain_candidates.iterrows():
    print(f"  {player['web_name']:15} | £{player['now_cost']:.1f}m | {player['total_points']:2.0f}pts | "
          f"PPG: {player['ppg']:.1f} | Form: {player['form']:.1f} | Owned: {player['selected_by_percent']:.1f}%")

print()

# Form Players (Rising in Value)
print("📈 FORM PLAYERS (Hot Picks):")
print("=" * 50)

form_players = active_players[
    (active_players['form'] >= 4.0) &
    (active_players['cost_change_start'] >= 0)
].nlargest(10, 'form')[
    ['web_name', 'now_cost', 'total_points', 'form', 'cost_change_start', 'transfers_in_event']
]

for _, player in form_players.iterrows():
    print(f"  {player['web_name']:15} | £{player['now_cost']:.1f}m | {player['total_points']:2.0f}pts | "
          f"Form: {player['form']:.1f} | Price Change: {player['cost_change_start']:+.1f} | "
          f"Transfers In: {player['transfers_in_event']:,}")

print()

# Transfer Recommendations
print("🔄 TRANSFER RECOMMENDATIONS:")
print("=" * 50)

print("PLAYERS TO CONSIDER BRINGING IN:")
transfer_targets = active_players[
    (active_players['form'] >= 4.0) &
    (active_players['selected_by_percent'] <= 15.0) &
    (active_players['total_points'] >= 25)
].nlargest(5, 'form')[
    ['web_name', 'now_cost', 'total_points', 'form', 'selected_by_percent']
]

for _, player in transfer_targets.iterrows():
    print(f"  ✅ {player['web_name']:15} | £{player['now_cost']:.1f}m | {player['total_points']:2.0f}pts | "
          f"Form: {player['form']:.1f} | Owned: {player['selected_by_percent']:.1f}%")

print("\nPLAYERS TO CONSIDER TRANSFERRING OUT:")
transfer_out = active_players[
    (active_players['form'] <= 2.0) &
    (active_players['selected_by_percent'] >= 10.0) &
    (active_players['now_cost'] >= 6.0)
].nsmallest(5, 'form')[
    ['web_name', 'now_cost', 'total_points', 'form', 'selected_by_percent']
]

for _, player in transfer_out.iterrows():
    print(f"  ❌ {player['web_name']:15} | £{player['now_cost']:.1f}m | {player['total_points']:2.0f}pts | "
          f"Form: {player['form']:.1f} | Owned: {player['selected_by_percent']:.1f}%")

print()

# Key Insights
print("💡 KEY INSIGHTS FOR GW9:")
print("=" * 50)

print("1. FIXTURE ANALYSIS:")
print("   • Liverpool (ELO: ~1993) face Brentford (H) - Strong captain option")
print("   • Arsenal (ELO: ~2037) host Crystal Palace - Defensive assets attractive")
print("   • Manchester City (ELO: ~1987) face Aston Villa (A) - Tough fixture")
print("   • Chelsea vs Sunderland - Chelsea assets in great form")

print("\n2. FORM TRENDS:")
high_form = active_players[active_players['form'] >= 5.0]['web_name'].tolist()
print(f"   • Players in exceptional form (5.0+): {', '.join(high_form[:5])}")

print("\n3. VALUE OPPORTUNITIES:")
print("   • Look for players under 15% ownership with good form")
print("   • Consider fixture swings for defensive assets")
print("   • Monitor price changes for popular transfers")

print("\n4. CAPTAIN RECOMMENDATIONS:")
top_captains = captain_candidates.head(3)['web_name'].tolist()
print(f"   • Primary options: {', '.join(top_captains)}")

print("\n" + "=" * 50)
print("Analysis complete! Good luck with your GW9 team selection! 🍀")