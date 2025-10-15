"""
Analyze Current FPL Team vs Optimal GW8 Recommendations
Compare user's current squad with data-driven recommendations
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
BASE_PATH = Path('data/2025-2026')
BY_GAMEWEEK_PATH = BASE_PATH / 'By Gameweek'

# Current Team
CURRENT_TEAM = {
    'Starting XI': [
        {'name': 'Vicario', 'position': 'Goalkeeper', 'team': 'Spurs'},
        {'name': 'Calafiori', 'position': 'Defender', 'team': 'Arsenal'},
        {'name': 'Pedro Porro', 'position': 'Defender', 'team': 'Spurs'},
        {'name': 'Rúben', 'position': 'Defender', 'team': 'Man City'},
        {'name': 'Reijnders', 'position': 'Midfielder', 'team': 'Man City'},
        {'name': 'Zubimendi', 'position': 'Midfielder', 'team': 'Arsenal'},
        {'name': 'Semenyo', 'position': 'Midfielder', 'team': 'Bournemouth'},
        {'name': 'Bruno G', 'position': 'Midfielder', 'team': 'Newcastle'},
        {'name': 'Gyökeres', 'position': 'Forward', 'team': 'Arsenal'},
        {'name': 'Haaland', 'position': 'Forward', 'team': 'Man City'},
        {'name': 'João Pedro', 'position': 'Forward', 'team': 'Chelsea'},
    ],
    'Bench': [
        {'name': 'Sels', 'position': 'Goalkeeper', 'team': "Nott'm Forest"},
        {'name': 'Caicedo', 'position': 'Midfielder', 'team': 'Chelsea'},
        {'name': 'Virgil', 'position': 'Defender', 'team': 'Liverpool'},
        {'name': 'Wan-Bissaka', 'position': 'Defender', 'team': 'West Ham'},
    ]
}

def load_player_stats():
    """Load GW7 player stats for analysis"""
    gw7_stats = pd.read_csv(BY_GAMEWEEK_PATH / 'GW7' / 'player_gameweek_stats.csv')
    
    # Load historical data
    all_stats = []
    for gw in range(1, 8):
        gw_path = BY_GAMEWEEK_PATH / f'GW{gw}' / 'player_gameweek_stats.csv'
        if gw_path.exists():
            gw_data = pd.read_csv(gw_path)
            gw_data['gameweek'] = gw
            all_stats.append(gw_data)
    
    historical = pd.concat(all_stats, ignore_index=True)
    
    return gw7_stats, historical

def analyze_player(player_name, position, team, gw7_stats, historical):
    """Analyze a specific player's performance"""
    
    # Try to find player by web_name (fuzzy matching)
    player_data_current = gw7_stats[gw7_stats['web_name'].str.contains(player_name, case=False, na=False)]
    
    if len(player_data_current) == 0:
        return {
            'name': player_name,
            'team': team,
            'position': position,
            'found': False,
            'status': 'NOT FOUND'
        }
    
    # Take first match
    player = player_data_current.iloc[0]
    
    # Get historical performance
    player_history = historical[historical['web_name'] == player['web_name']]
    
    total_points = player_history['event_points'].sum()
    games_played = len(player_history)
    avg_points = total_points / games_played if games_played > 0 else 0
    
    # Calculate points per 90
    total_minutes = player_history['minutes'].sum()
    points_per_90 = (total_points / total_minutes * 90) if total_minutes > 0 else 0
    
    return {
        'name': player['web_name'],
        'team': team,
        'position': position,
        'found': True,
        'cost': player['now_cost'] / 10,
        'total_points': total_points,
        'games_played': games_played,
        'avg_ppg': avg_points,
        'form': player['form'],
        'minutes': total_minutes,
        'points_per_90': points_per_90,
        'value': total_points / (player['now_cost'] / 10) if player['now_cost'] > 0 else 0,
        'ownership': player['selected_by_percent'],
        'status': player.get('status', 'a')
    }

def load_fixtures():
    """Load GW8 fixtures"""
    gw8_fixtures = pd.read_csv(BY_GAMEWEEK_PATH / 'GW8' / 'fixtures.csv')
    gw8_teams = pd.read_csv(BY_GAMEWEEK_PATH / 'GW8' / 'teams.csv')
    
    teams_master = pd.read_csv(BASE_PATH / 'teams.csv')
    team_name_map = dict(zip(teams_master['code'], teams_master['name']))
    
    return gw8_fixtures, gw8_teams, team_name_map

def get_fixture_difficulty(team_name, gw8_fixtures, gw8_teams, team_name_map):
    """Get fixture difficulty for a team"""
    
    # Find team code
    team_code = None
    for code, name in team_name_map.items():
        if name == team_name:
            team_code = code
            break
    
    if team_code is None:
        return 1500  # Default
    
    # Get team's fixture
    home_fixture = gw8_fixtures[gw8_fixtures['home_team'] == team_code]
    away_fixture = gw8_fixtures[gw8_fixtures['away_team'] == team_code]
    
    team_elo = dict(zip(gw8_teams['code'], gw8_teams['elo']))
    
    if len(home_fixture) > 0:
        opp_code = home_fixture.iloc[0]['away_team']
        opp_elo = team_elo.get(opp_code, 1500)
        return opp_elo, team_name_map.get(opp_code, 'Unknown'), 'H'
    elif len(away_fixture) > 0:
        opp_code = away_fixture.iloc[0]['home_team']
        opp_elo = team_elo.get(opp_code, 1500) + 50  # Home advantage
        return opp_elo, team_name_map.get(opp_code, 'Unknown'), 'A'
    
    return 1500, 'No fixture', '-'

def print_header(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def main():
    print_header("🏆 CURRENT FPL TEAM ANALYSIS FOR GW8")
    
    # Load data
    print("\n📊 Loading player statistics...")
    gw7_stats, historical = load_player_stats()
    gw8_fixtures, gw8_teams, team_name_map = load_fixtures()
    
    print("✓ Data loaded successfully")
    
    # Analyze current team
    print_header("📋 YOUR CURRENT STARTING XI ANALYSIS")
    
    starting_xi_analysis = []
    for player in CURRENT_TEAM['Starting XI']:
        analysis = analyze_player(player['name'], player['position'], player['team'], 
                                 gw7_stats, historical)
        if analysis['found']:
            fixture_diff, opponent, venue = get_fixture_difficulty(player['team'], 
                                                                   gw8_fixtures, gw8_teams, team_name_map)
            analysis['gw8_opponent'] = opponent
            analysis['venue'] = venue
            analysis['fixture_difficulty'] = fixture_diff
        starting_xi_analysis.append(analysis)
    
    # Print Starting XI
    print(f"\n{'Pos':<4} {'Player':<20} {'Team':<15} {'£':<6} {'Pts':<5} {'PPG':<5} {'Form':<5} {'GW8 Fixture':<25}")
    print("-" * 90)
    
    total_cost = 0
    total_points = 0
    
    for p in starting_xi_analysis:
        if p['found']:
            fixture_str = f"{p['gw8_opponent']} ({'H' if p['venue'] == 'H' else 'A'})"
            print(f"{p['position'][:3]:<4} {p['name'][:19]:<20} {p['team'][:14]:<15} "
                  f"£{p['cost']:<5.1f} {int(p['total_points']):<5} {p['avg_ppg']:<5.1f} "
                  f"{p['form']:<5} {fixture_str:<25}")
            total_cost += p['cost']
            total_points += p['total_points']
        else:
            print(f"{p['position'][:3]:<4} {p['name'][:19]:<20} {p['team'][:14]:<15} ❌ NOT FOUND IN DATABASE")
    
    print("-" * 90)
    print(f"{'TOTAL':<39} £{total_cost:<5.1f} {int(total_points):<5}")
    
    # Analyze bench
    print_header("🪑 YOUR BENCH ANALYSIS")
    
    bench_analysis = []
    for player in CURRENT_TEAM['Bench']:
        analysis = analyze_player(player['name'], player['position'], player['team'], 
                                 gw7_stats, historical)
        if analysis['found']:
            fixture_diff, opponent, venue = get_fixture_difficulty(player['team'], 
                                                                   gw8_fixtures, gw8_teams, team_name_map)
            analysis['gw8_opponent'] = opponent
            analysis['venue'] = venue
            analysis['fixture_difficulty'] = fixture_diff
        bench_analysis.append(analysis)
    
    print(f"\n{'Pos':<4} {'Player':<20} {'Team':<15} {'£':<6} {'Pts':<5} {'PPG':<5} {'Form':<5} {'GW8 Fixture':<25}")
    print("-" * 90)
    
    bench_cost = 0
    bench_points = 0
    
    for p in bench_analysis:
        if p['found']:
            fixture_str = f"{p['gw8_opponent']} ({'H' if p['venue'] == 'H' else 'A'})"
            print(f"{p['position'][:3]:<4} {p['name'][:19]:<20} {p['team'][:14]:<15} "
                  f"£{p['cost']:<5.1f} {int(p['total_points']):<5} {p['avg_ppg']:<5.1f} "
                  f"{p['form']:<5} {fixture_str:<25}")
            bench_cost += p['cost']
            bench_points += p['total_points']
        else:
            print(f"{p['position'][:3]:<4} {p['name'][:19]:<20} {p['team'][:14]:<15} ❌ NOT FOUND IN DATABASE")
    
    print("-" * 90)
    print(f"{'BENCH TOTAL':<39} £{bench_cost:<5.1f} {int(bench_points):<5}")
    
    # Overall summary
    print_header("📊 OVERALL TEAM SUMMARY")
    
    squad_cost = total_cost + bench_cost
    squad_points = total_points + bench_points
    
    print(f"\n💰 BUDGET:")
    print(f"   Total Squad Value: £{squad_cost:.1f}m")
    print(f"   Budget Remaining: £{100 - squad_cost:.1f}m")
    
    print(f"\n📈 PERFORMANCE (GW1-7):")
    print(f"   Total Points: {int(squad_points)}")
    print(f"   Starting XI Points: {int(total_points)}")
    print(f"   Bench Points: {int(bench_points)}")
    print(f"   Average PPG (Starting XI): {total_points / 11 / 7:.1f}")
    
    # Position breakdown
    positions = {}
    for p in starting_xi_analysis + bench_analysis:
        if p['found']:
            pos = p['position']
            if pos not in positions:
                positions[pos] = {'count': 0, 'cost': 0, 'points': 0}
            positions[pos]['count'] += 1
            positions[pos]['cost'] += p['cost']
            positions[pos]['points'] += p['total_points']
    
    print(f"\n👥 SQUAD COMPOSITION:")
    for pos in ['Goalkeeper', 'Defender', 'Midfielder', 'Forward']:
        if pos in positions:
            p = positions[pos]
            print(f"   {pos}: {p['count']} players (£{p['cost']:.1f}m, {int(p['points'])} pts)")
    
    # Team diversity
    teams = {}
    for p in starting_xi_analysis + bench_analysis:
        if p['found']:
            team = p['team']
            teams[team] = teams.get(team, 0) + 1
    
    print(f"\n🏟️  TEAM DISTRIBUTION:")
    for team, count in sorted(teams.items(), key=lambda x: x[1], reverse=True):
        print(f"   {team}: {count} player{'s' if count > 1 else ''}")
        if count > 3:
            print(f"      ⚠️  WARNING: Max 3 players per team allowed!")
    
    # GW8 Fixture Analysis
    print_header("🎯 GW8 FIXTURE DIFFICULTY ANALYSIS")
    
    fixture_ratings = []
    for p in starting_xi_analysis:
        if p['found']:
            fixture_ratings.append({
                'player': p['name'],
                'team': p['team'],
                'opponent': p['gw8_opponent'],
                'venue': p['venue'],
                'difficulty': p['fixture_difficulty']
            })
    
    # Sort by difficulty (lower is easier)
    fixture_ratings.sort(key=lambda x: x['difficulty'])
    
    print(f"\n🟢 EASIEST FIXTURES:")
    for i, f in enumerate(fixture_ratings[:5], 1):
        venue_str = 'H' if f['venue'] == 'H' else 'A'
        print(f"   {i}. {f['player']:<20} {f['team']:<15} vs {f['opponent']:<15} ({venue_str}) - Difficulty: {int(f['difficulty'])}")
    
    print(f"\n🔴 HARDEST FIXTURES:")
    for i, f in enumerate(fixture_ratings[-5:], 1):
        venue_str = 'H' if f['venue'] == 'H' else 'A'
        print(f"   {i}. {f['player']:<20} {f['team']:<15} vs {f['opponent']:<15} ({venue_str}) - Difficulty: {int(f['difficulty'])}")
    
    # Key recommendations
    print_header("💡 KEY INSIGHTS & RECOMMENDATIONS")
    
    # Find underperformers
    all_players = starting_xi_analysis + bench_analysis
    valid_players = [p for p in all_players if p['found']]
    
    if valid_players:
        avg_ppg = sum(p['avg_ppg'] for p in valid_players) / len(valid_players)
        
        underperformers = [p for p in valid_players if p['avg_ppg'] < avg_ppg * 0.6 and p['avg_ppg'] > 0]
        underperformers.sort(key=lambda x: x['avg_ppg'])
        
        if underperformers:
            print(f"\n⚠️  UNDERPERFORMING PLAYERS (consider transferring):")
            for p in underperformers[:3]:
                print(f"   • {p['name']} ({p['team']}) - {p['avg_ppg']:.1f} PPG, £{p['cost']:.1f}m")
                print(f"     Total: {int(p['total_points'])} pts in {p['games_played']} games")
        
        # Find high performers
        stars = [p for p in valid_players if p['avg_ppg'] > avg_ppg * 1.5]
        stars.sort(key=lambda x: x['avg_ppg'], reverse=True)
        
        if stars:
            print(f"\n⭐ STAR PERFORMERS (keep/captain):")
            for p in stars[:3]:
                print(f"   • {p['name']} ({p['team']}) - {p['avg_ppg']:.1f} PPG, £{p['cost']:.1f}m")
                print(f"     Form: {p['form']}, Total: {int(p['total_points'])} pts")
        
        # Fixture-based recommendations
        print(f"\n🎯 GW8 CAPTAINCY SUGGESTIONS:")
        captain_candidates = [p for p in starting_xi_analysis if p['found']]
        captain_candidates.sort(key=lambda x: (x['avg_ppg'], -x['fixture_difficulty']), reverse=True)
        
        for i, p in enumerate(captain_candidates[:3], 1):
            venue_str = 'H' if p['venue'] == 'H' else 'A'
            print(f"   {i}. {p['name']} ({p['team']}) - {p['avg_ppg']:.1f} PPG")
            print(f"      vs {p['gw8_opponent']} ({venue_str}) - Fixture Difficulty: {int(p['fixture_difficulty'])}")
    
    print("\n" + "=" * 80)
    print("✅ Analysis Complete!")
    print("=" * 80)

if __name__ == '__main__':
    main()
