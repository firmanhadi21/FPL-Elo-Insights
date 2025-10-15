#!/usr/bin/env python3
"""
FPL GW8 Team Builder & Strategy
Complete Analysis Based on GW1-GW7 Performance

This script analyzes all player performance data from GW1-GW7 and recommends
the optimal team for GW8 using official FPL scoring rules.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_PATH = Path('/Users/macbook/Dropbox/GitHub/FPL/FPL-Elo-Insights/data/2025-2026')
BY_GAMEWEEK_PATH = BASE_PATH / 'By Gameweek'

# FPL Budget and Squad Rules
TOTAL_BUDGET = 100.0  # £100 million
SQUAD_SIZE = 15
MIN_GKP, MAX_GKP = 2, 2
MIN_DEF, MAX_DEF = 5, 5
MIN_MID, MAX_MID = 5, 5
MIN_FWD, MAX_FWD = 3, 3
MAX_PLAYERS_PER_TEAM = 3

# Position name mapping
POSITION_NAMES = {
    'Goalkeeper': 'GKP',
    'Defender': 'DEF', 
    'Midfielder': 'MID',
    'Forward': 'FWD'
}

# Valid formations: (GK, DEF, MID, FWD)
VALID_FORMATIONS = [
    (1, 3, 4, 3),  # 3-4-3
    (1, 3, 5, 2),  # 3-5-2
    (1, 4, 3, 3),  # 4-3-3
    (1, 4, 4, 2),  # 4-4-2
    (1, 4, 5, 1),  # 4-5-1
    (1, 5, 3, 2),  # 5-3-2
    (1, 5, 4, 1),  # 5-4-1
]

# ============================================================================
# SECTION 1: DATA LOADING
# ============================================================================

def print_section_header(title):
    """Print a formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def load_master_data():
    """Load master player, team, and gameweek data"""
    print_section_header("📂 LOADING MASTER DATA")
    
    # Use GW7 players.csv as it has the most recent player info with positions
    players_master = pd.read_csv(BY_GAMEWEEK_PATH / 'GW7' / 'players.csv')
    teams_master = pd.read_csv(BASE_PATH / 'teams.csv')
    gameweek_summaries = pd.read_csv(BASE_PATH / 'gameweek_summaries.csv')
    
    print(f"✓ Master players data: {len(players_master)} players")
    print(f"✓ Master teams data: {len(teams_master)} teams")
    print(f"✓ Gameweek summaries: {len(gameweek_summaries)} gameweeks")
    
    return players_master, teams_master, gameweek_summaries


def load_gw8_data():
    """Load GW8 fixtures and planning data"""
    print("\n📅 Loading GW8 Planning Data...")
    
    gw8_path = BY_GAMEWEEK_PATH / 'GW8'
    gw8_fixtures = pd.read_csv(gw8_path / 'fixtures.csv')
    gw8_players = pd.read_csv(gw8_path / 'players.csv')
    gw8_teams = pd.read_csv(gw8_path / 'teams.csv')
    
    print(f"✓ GW8 fixtures: {len(gw8_fixtures)} matches")
    print(f"✓ GW8 players: {len(gw8_players)} players")
    print(f"✓ GW8 teams: {len(gw8_teams)} teams")
    
    return gw8_fixtures, gw8_players, gw8_teams


def load_historical_performance():
    """Load historical performance data from GW1-GW7"""
    print_section_header("📊 LOADING HISTORICAL PERFORMANCE (GW1-GW7)")
    
    all_gw_stats = []
    
    for gw in range(1, 8):
        gw_path = BY_GAMEWEEK_PATH / f'GW{gw}'
        
        # Prefer discrete stats over cumulative
        discrete_stats_path = gw_path / 'player_gameweek_stats.csv'
        
        if discrete_stats_path.exists():
            gw_data = pd.read_csv(discrete_stats_path)
            gw_data['gameweek'] = gw
            all_gw_stats.append(gw_data)
            print(f"✓ GW{gw}: {len(gw_data)} player records (discrete stats)")
        else:
            playerstats_path = gw_path / 'playerstats.csv'
            if playerstats_path.exists():
                gw_data = pd.read_csv(playerstats_path)
                gw_data['gameweek'] = gw
                all_gw_stats.append(gw_data)
                print(f"✓ GW{gw}: {len(gw_data)} player records (cumulative stats)")
    
    historical_stats = pd.concat(all_gw_stats, ignore_index=True)
    
    print(f"\n✅ Total historical records: {len(historical_stats)}")
    print(f"✅ Gameweeks loaded: GW1-GW7")
    print(f"✅ Unique players: {historical_stats['id'].nunique()}")
    
    return historical_stats


# ============================================================================
# SECTION 2: DATA ENRICHMENT
# ============================================================================

def enrich_historical_data(historical_stats, players_master, teams_master):
    """Add position and team information to historical data"""
    print_section_header("🔧 ENRICHING HISTORICAL DATA")
    
    # Create mapping from id to player_id (to link historical stats with master data)
    # Historical stats use 'id', players.csv uses 'player_id'
    # But both should have web_name, so we can merge on that
    
    # Position mapping via web_name
    position_map = dict(zip(players_master['web_name'], players_master['position']))
    historical_stats['position'] = historical_stats['web_name'].map(position_map)
    
    # Team mapping via web_name
    team_map = dict(zip(players_master['web_name'], players_master['team_code']))
    historical_stats['team_code'] = historical_stats['web_name'].map(team_map)
    
    team_name_map = dict(zip(teams_master['code'], teams_master['name']))
    historical_stats['team_name'] = historical_stats['team_code'].map(team_name_map)
    
    print("✓ Position mapping added")
    print("✓ Team mapping added")
    print("\nPosition distribution:")
    print(historical_stats['position'].value_counts())
    
    return historical_stats, team_name_map


# ============================================================================
# SECTION 3: PERFORMANCE ANALYSIS
# ============================================================================

def calculate_player_performance(historical_stats):
    """Calculate aggregate performance metrics for each player"""
    print_section_header("📊 CALCULATING PLAYER PERFORMANCE METRICS")
    
    # Group by player and calculate aggregate stats
    player_performance = historical_stats.groupby(['id', 'web_name', 'position', 'team_name']).agg({
        'total_points': 'sum',
        'event_points': 'mean',
        'minutes': 'sum',
        'goals_scored': 'sum',
        'assists': 'sum',
        'clean_sheets': 'sum',
        'goals_conceded': 'sum',
        'saves': 'sum',
        'bonus': 'sum',
        'bps': 'mean',
        'yellow_cards': 'sum',
        'red_cards': 'sum',
        'gameweek': 'count'
    }).reset_index()
    
    # Rename columns
    player_performance.rename(columns={
        'total_points': 'total_points_sum',
        'event_points': 'avg_points_per_gw',
        'gameweek': 'games_played'
    }, inplace=True)
    
    # Calculate per-90 metrics
    player_performance['minutes_per_game'] = player_performance['minutes'] / player_performance['games_played']
    player_performance['points_per_90'] = (player_performance['total_points_sum'] / player_performance['minutes']) * 90
    player_performance['goals_per_90'] = (player_performance['goals_scored'] / player_performance['minutes']) * 90
    player_performance['assists_per_90'] = (player_performance['assists'] / player_performance['minutes']) * 90
    
    # Handle division by zero
    player_performance = player_performance.replace([np.inf, -np.inf], 0)
    player_performance = player_performance.fillna(0)
    
    print(f"✅ Performance metrics calculated for {len(player_performance)} players")
    print("\nTop 10 performers:")
    print(player_performance.nlargest(10, 'total_points_sum')[
        ['web_name', 'position', 'team_name', 'total_points_sum', 'avg_points_per_gw', 'games_played']
    ])
    
    return player_performance


def add_current_data(player_performance, gw8_players):
    """Add current cost, ownership, and form from GW7 data (most recent)"""
    print_section_header("💰 ADDING CURRENT COST & OWNERSHIP DATA")
    
    # Load GW7 player stats to get current prices and form
    gw7_stats = pd.read_csv(BY_GAMEWEEK_PATH / 'GW7' / 'player_gameweek_stats.csv')
    
    # Create mappings using web_name as the key
    cost_map = dict(zip(gw7_stats['web_name'], gw7_stats['now_cost']))
    ownership_map = dict(zip(gw7_stats['web_name'], gw7_stats['selected_by_percent']))
    form_map = dict(zip(gw7_stats['web_name'], gw7_stats['form']))
    
    player_performance['now_cost'] = player_performance['web_name'].map(cost_map)
    player_performance['selected_by_percent'] = player_performance['web_name'].map(ownership_map)
    player_performance['form'] = player_performance['web_name'].map(form_map)
    
    # Calculate value metrics
    player_performance['cost_millions'] = player_performance['now_cost'] / 10
    player_performance['value_score'] = player_performance['total_points_sum'] / player_performance['cost_millions']
    player_performance['form_numeric'] = pd.to_numeric(player_performance['form'], errors='coerce').fillna(0)
    
    print("✓ Cost data added (from GW7)")
    print("✓ Ownership data added")
    print("✓ Form data added")
    print("✓ Value scores calculated")
    
    print("\n💎 Best Value Players (Points per £million):")
    print(player_performance.nlargest(10, 'value_score')[
        ['web_name', 'position', 'team_name', 'total_points_sum', 'cost_millions', 'value_score']
    ])
    
    return player_performance


# ============================================================================
# SECTION 4: FIXTURE ANALYSIS
# ============================================================================

def analyze_fixtures(gw8_fixtures, gw8_teams, team_name_map):
    """Analyze GW8 fixtures and calculate difficulty"""
    print_section_header("🎯 ANALYZING GW8 FIXTURES")
    
    # Display fixtures - home_team and away_team are numeric codes
    print("\n📅 GW8 Fixtures:")
    print("-" * 70)
    
    # Map team codes to names
    for idx, row in gw8_fixtures.iterrows():
        home_name = team_name_map.get(row['home_team'], f"Team {row['home_team']}")
        away_name = team_name_map.get(row['away_team'], f"Team {row['away_team']}")
        print(f"{home_name} vs {away_name}")
    
    print(f"\n✅ Total GW8 matches: {len(gw8_fixtures)}")
    
    # Calculate fixture difficulty using Elo
    team_fixture_difficulty = {}
    
    # Create team code to Elo mapping
    team_elo = dict(zip(gw8_teams['code'], gw8_teams['elo']))
    
    for team_code, team_name in team_name_map.items():
        home_fixtures = gw8_fixtures[gw8_fixtures['home_team'] == team_code]
        away_fixtures = gw8_fixtures[gw8_fixtures['away_team'] == team_code]
        
        opponents_elo = []
        
        for _, match in home_fixtures.iterrows():
            opp_elo = team_elo.get(match['away_team'], 1500)
            opponents_elo.append(opp_elo)
        
        for _, match in away_fixtures.iterrows():
            opp_elo = team_elo.get(match['home_team'], 1500)
            opponents_elo.append(opp_elo + 50)  # Home advantage
        
        if opponents_elo:
            team_fixture_difficulty[team_name] = np.mean(opponents_elo)
        else:
            team_fixture_difficulty[team_name] = 1500  # Default
    
    print("\n🟢 Easiest Fixtures:")
    sorted_fixtures = sorted(team_fixture_difficulty.items(), key=lambda x: x[1])
    for team, difficulty in sorted_fixtures[:5]:
        print(f"  {team}: {difficulty:.0f}")
    
    print("\n🔴 Hardest Fixtures:")
    for team, difficulty in sorted_fixtures[-5:]:
        print(f"  {team}: {difficulty:.0f}")
    
    return team_fixture_difficulty


# ============================================================================
# SECTION 5: PLAYER RANKING
# ============================================================================

def rank_players(player_performance, team_fixture_difficulty):
    """Create comprehensive player rankings"""
    print_section_header("🏅 CREATING PLAYER RANKINGS")
    
    # Add fixture difficulty to player data
    player_performance['fixture_difficulty'] = player_performance['team_name'].map(team_fixture_difficulty)
    
    # Filter active players (minimum 180 minutes)
    active_players = player_performance[player_performance['minutes'] >= 180].copy()
    
    print(f"Active players (180+ minutes): {len(active_players)}")
    
    # Normalize metrics
    scaler = MinMaxScaler()
    
    positive_metrics = ['total_points_sum', 'avg_points_per_gw', 'value_score', 'form_numeric', 'points_per_90']
    negative_metrics = ['now_cost', 'fixture_difficulty']
    
    for metric in positive_metrics:
        if metric in active_players.columns:
            active_players[f'{metric}_norm'] = scaler.fit_transform(active_players[[metric]])
    
    for metric in negative_metrics:
        if metric in active_players.columns:
            active_players[f'{metric}_norm'] = 1 - scaler.fit_transform(active_players[[metric]])
    
    # Calculate composite score
    weights = {
        'total_points_sum_norm': 0.25,
        'avg_points_per_gw_norm': 0.25,
        'value_score_norm': 0.15,
        'form_numeric_norm': 0.15,
        'points_per_90_norm': 0.10,
        'now_cost_norm': 0.05,
        'fixture_difficulty_norm': 0.05
    }
    
    active_players['composite_score'] = sum(
        active_players[metric] * weight 
        for metric, weight in weights.items() 
        if metric in active_players.columns
    )
    
    print(f"✅ Composite rankings calculated")
    print("\n🏆 Top 10 Players Overall:")
    print(active_players.nlargest(10, 'composite_score')[
        ['web_name', 'position', 'team_name', 'total_points_sum', 'cost_millions', 'composite_score']
    ])
    
    return active_players


def show_position_recommendations(active_players):
    """Display top recommendations by position"""
    print_section_header("🎯 TOP RECOMMENDATIONS BY POSITION")
    
    for position in ['Goalkeeper', 'Defender', 'Midfielder', 'Forward']:
        abbr = POSITION_NAMES.get(position, position)
        print(f"\n{'=' * 70}")
        print(f"🔹 {abbr} - Top 15 Recommendations")
        print(f"{'=' * 70}")
        
        pos_players = active_players[active_players['position'] == position].nlargest(15, 'composite_score')
        
        print(f"\n{'Rank':<5} {'Player':<20} {'Team':<15} {'Pts':<6} {'£':<6} {'PPG':<6} {'Val':<6} {'Score':<8}")
        print("-" * 70)
        
        for rank, (idx, player) in enumerate(pos_players.iterrows(), 1):
            print(f"{rank:<5} {player['web_name'][:19]:<20} {player['team_name'][:14]:<15} "
                  f"{int(player['total_points_sum']):<6} {player['cost_millions']:<6.1f} "
                  f"{player['avg_points_per_gw']:<6.1f} {player['value_score']:<6.1f} "
                  f"{player['composite_score']:<8.3f}")

# ============================================================================
# SECTION 6: SQUAD BUILDING
# ============================================================================

def build_optimal_squad(active_players, budget=100.0):
    """Build optimal FPL squad using greedy algorithm"""
    print_section_header("🤖 BUILDING OPTIMAL GW8 SQUAD")
    
    selected_squad = []
    remaining_budget = budget
    team_counts = {}
    
    position_requirements = {
        'Goalkeeper': {'min': MIN_GKP, 'max': MAX_GKP, 'selected': 0},
        'Defender': {'min': MIN_DEF, 'max': MAX_DEF, 'selected': 0},
        'Midfielder': {'min': MIN_MID, 'max': MAX_MID, 'selected': 0},
        'Forward': {'min': MIN_FWD, 'max': MAX_FWD, 'selected': 0}
    }
    
    sorted_players = active_players.sort_values('composite_score', ascending=False)
    
    for position in ['Goalkeeper', 'Defender', 'Midfielder', 'Forward']:
        pos_req = position_requirements[position]
        pos_players = sorted_players[sorted_players['position'] == position]
        
        for _, player in pos_players.iterrows():
            if pos_req['selected'] >= pos_req['max']:
                break
            
            if player['cost_millions'] > remaining_budget:
                continue
            
            team = player['team_name']
            if team_counts.get(team, 0) >= MAX_PLAYERS_PER_TEAM:
                continue
            
            selected_squad.append(player)
            remaining_budget -= player['cost_millions']
            pos_req['selected'] += 1
            team_counts[team] = team_counts.get(team, 0) + 1
    
    squad_df = pd.DataFrame(selected_squad)
    
    print(f"\n✅ Squad Built Successfully!")
    print(f"💰 Total Cost: £{budget - remaining_budget:.1f}m")
    print(f"💵 Remaining Budget: £{remaining_budget:.1f}m")
    print(f"👥 Squad Size: {len(squad_df)} players")
    
    print(f"\n📊 Position Breakdown:")
    for pos in ['Goalkeeper', 'Defender', 'Midfielder', 'Forward']:
        count = len(squad_df[squad_df['position'] == pos]) if len(squad_df) > 0 else 0
        abbr = POSITION_NAMES.get(pos, pos)
        print(f"  {abbr}: {count} players")
    
    return squad_df


def display_squad(squad_df):
    """Display the optimal squad"""
    print_section_header("🏆 YOUR OPTIMAL GW8 SQUAD")
    
    # Create position order for sorting
    position_order = {'Goalkeeper': 1, 'Defender': 2, 'Midfielder': 3, 'Forward': 4}
    squad_df['pos_order'] = squad_df['position'].map(position_order)
    squad_sorted = squad_df.sort_values(['pos_order', 'composite_score'], ascending=[True, False])
    
    print(f"\n{'Pos':<4} {'Player':<20} {'Team':<15} {'Cost':<7} {'GW Pts':<7} {'Total':<7} {'Form':<6}")
    print("-" * 70)
    
    for idx, player in squad_sorted.iterrows():
        abbr = POSITION_NAMES.get(player['position'], player['position'])
        print(f"{abbr:<4} {player['web_name'][:19]:<20} {player['team_name'][:14]:<15} "
              f"£{player['cost_millions']:<6.1f} {player['avg_points_per_gw']:<7.1f} "
              f"{int(player['total_points_sum']):<7} {player['form_numeric']:<6.1f}")
    
    print("-" * 70)
    print(f"{'TOTAL':<39} £{squad_sorted['cost_millions'].sum():<6.1f} "
          f"{squad_sorted['avg_points_per_gw'].sum():<7.1f} "
          f"{int(squad_sorted['total_points_sum'].sum()):<7}")
    
    return squad_sorted


# ============================================================================
# SECTION 7: STARTING XI SELECTION
# ============================================================================

def select_starting_xi(squad_sorted):
    """Select best starting XI with valid formation"""
    print_section_header("👑 SELECTING OPTIMAL STARTING XI")
    
    best_xi = None
    best_score = 0
    best_formation = None
    
    for formation in VALID_FORMATIONS:
        gk_count, def_count, mid_count, fwd_count = formation
        
        xi_gkp = squad_sorted[squad_sorted['position'] == 'Goalkeeper'].nlargest(gk_count, 'composite_score')
        xi_def = squad_sorted[squad_sorted['position'] == 'Defender'].nlargest(def_count, 'composite_score')
        xi_mid = squad_sorted[squad_sorted['position'] == 'Midfielder'].nlargest(mid_count, 'composite_score')
        xi_fwd = squad_sorted[squad_sorted['position'] == 'Forward'].nlargest(fwd_count, 'composite_score')
        
        xi = pd.concat([xi_gkp, xi_def, xi_mid, xi_fwd])
        
        if len(xi) == 11:
            xi_score = xi['composite_score'].sum()
            if xi_score > best_score:
                best_score = xi_score
                best_xi = xi
                best_formation = formation
    
    print(f"\n✅ Optimal Formation: {best_formation[1]}-{best_formation[2]}-{best_formation[3]}")
    print(f"✅ Team Strength Score: {best_score:.3f}")
    
    print(f"\n🎯 STARTING XI FOR GW8:")
    print("=" * 70)
    print(f"\n{'Pos':<4} {'Player':<20} {'Team':<15} {'PPG':<7} {'Total':<7} {'Form':<6}")
    print("-" * 70)
    
    for idx, player in best_xi.iterrows():
        print(f"{player['position']:<4} {player['web_name'][:19]:<20} {player['team_name'][:14]:<15} "
              f"{player['avg_points_per_gw']:<7.1f} {int(player['total_points_sum']):<7} "
              f"{player['form_numeric']:<6.1f}")
    
    bench = squad_sorted[~squad_sorted['id'].isin(best_xi['id'])]
    
    print(f"\n🪑 BENCH:")
    print("-" * 70)
    for idx, player in bench.iterrows():
        print(f"{player['position']:<4} {player['web_name'][:19]:<20} {player['team_name'][:14]:<15} "
              f"{player['avg_points_per_gw']:<7.1f} {int(player['total_points_sum']):<7} "
              f"{player['form_numeric']:<6.1f}")
    
    return best_xi, best_formation, bench


def select_captain(best_xi):
    """Select captain and vice-captain"""
    print_section_header("👑 CAPTAIN RECOMMENDATIONS")
    
    captain_candidates = best_xi.nlargest(5, 'composite_score')
    
    print(f"\n🎯 Top 5 Captain Options:")
    print(f"\n{'Rank':<6} {'Player':<20} {'Team':<15} {'PPG':<7} {'Form':<7} {'Total':<7}")
    print("-" * 70)
    
    for rank, (idx, player) in enumerate(captain_candidates.iterrows(), 1):
        print(f"{rank:<6} {player['web_name'][:19]:<20} {player['team_name'][:14]:<15} "
              f"{player['avg_points_per_gw']:<7.1f} {player['form_numeric']:<7.1f} "
              f"{int(player['total_points_sum']):<7}")
    
    captain = captain_candidates.iloc[0]
    vice_captain = captain_candidates.iloc[1]
    
    print(f"\n⭐ RECOMMENDED CAPTAIN: {captain['web_name']} ({captain['team_name']})")
    print(f"   Average Points/GW: {captain['avg_points_per_gw']:.1f}")
    print(f"   Current Form: {captain['form_numeric']:.1f}")
    print(f"   Total Points (GW1-7): {int(captain['total_points_sum'])}")
    
    print(f"\n🥈 RECOMMENDED VICE-CAPTAIN: {vice_captain['web_name']} ({vice_captain['team_name']})")
    print(f"   Average Points/GW: {vice_captain['avg_points_per_gw']:.1f}")
    print(f"   Current Form: {vice_captain['form_numeric']:.1f}")
    print(f"   Total Points (GW1-7): {int(vice_captain['total_points_sum'])}")
    
    return captain, vice_captain


# ============================================================================
# SECTION 8: VISUALIZATIONS
# ============================================================================

def create_visualizations(squad_sorted):
    """Create performance visualizations"""
    print_section_header("📈 CREATING PERFORMANCE VISUALIZATIONS")
    
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Points by position
        ax1 = axes[0, 0]
        squad_sorted.groupby('position')['total_points_sum'].sum().plot(kind='bar', ax=ax1, color='skyblue')
        ax1.set_title('Total Points by Position (Your Squad)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Position')
        ax1.set_ylabel('Total Points')
        ax1.grid(True, alpha=0.3)
        
        # 2. Cost vs Points
        ax2 = axes[0, 1]
        colors = {'GKP': 'gold', 'DEF': 'blue', 'MID': 'green', 'FWD': 'red'}
        for pos in ['GKP', 'DEF', 'MID', 'FWD']:
            pos_data = squad_sorted[squad_sorted['position'] == pos]
            ax2.scatter(pos_data['cost_millions'], pos_data['total_points_sum'], 
                       label=pos, alpha=0.7, s=100, color=colors[pos])
        ax2.set_title('Cost vs Total Points (Your Squad)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Cost (£m)')
        ax2.set_ylabel('Total Points')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Form comparison
        ax3 = axes[1, 0]
        squad_sorted.nlargest(10, 'form_numeric')[['web_name', 'form_numeric']].set_index('web_name').plot(
            kind='barh', ax=ax3, color='orange', legend=False
        )
        ax3.set_title('Top 10 Players by Form (Your Squad)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Form Score')
        ax3.grid(True, alpha=0.3)
        
        # 4. Team representation
        ax4 = axes[1, 1]
        team_counts = squad_sorted['team_name'].value_counts()
        team_counts.plot(kind='bar', ax=ax4, color='coral')
        ax4.set_title('Players per Team (Your Squad)', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Team')
        ax4.set_ylabel('Number of Players')
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')
        ax4.axhline(y=3, color='red', linestyle='--', label='Max Limit (3)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('gw8_squad_analysis.png', dpi=300, bbox_inches='tight')
        print("\n✅ Visualizations saved as 'gw8_squad_analysis.png'")
        
    except Exception as e:
        print(f"\n⚠️  Could not create visualizations: {e}")


# ============================================================================
# SECTION 9: EXPORT RESULTS
# ============================================================================

def export_results(squad_sorted, best_xi):
    """Export squad and starting XI to CSV files"""
    print_section_header("💾 EXPORTING RESULTS")
    
    # Export full squad
    squad_export = squad_sorted[[
        'web_name', 'position', 'team_name', 'cost_millions', 
        'total_points_sum', 'avg_points_per_gw', 'form_numeric', 
        'value_score', 'composite_score'
    ]].copy()
    
    squad_export.columns = [
        'Player', 'Position', 'Team', 'Cost (£m)', 
        'Total Points', 'Avg PPG', 'Form', 'Value', 'Score'
    ]
    
    squad_export.to_csv('gw8_optimal_squad.csv', index=False)
    print("✓ Squad exported to 'gw8_optimal_squad.csv'")
    
    # Export starting XI
    xi_export = best_xi[[
        'web_name', 'position', 'team_name', 'cost_millions', 
        'total_points_sum', 'avg_points_per_gw', 'form_numeric'
    ]].copy()
    
    xi_export.columns = [
        'Player', 'Position', 'Team', 'Cost (£m)', 
        'Total Points', 'Avg PPG', 'Form'
    ]
    
    xi_export.to_csv('gw8_starting_xi.csv', index=False)
    print("✓ Starting XI exported to 'gw8_starting_xi.csv'")
    
    print("\n✅ All files saved successfully!")


# ============================================================================
# SECTION 10: FINAL SUMMARY
# ============================================================================

def print_final_summary(squad_sorted, best_xi, best_formation, captain, vice_captain):
    """Print comprehensive final summary"""
    print_section_header("🏆 GW8 TEAM STRATEGY - FINAL SUMMARY")
    
    print(f"\n💰 BUDGET SUMMARY:")
    print(f"  Total Budget: £{TOTAL_BUDGET}m")
    print(f"  Squad Cost: £{squad_sorted['cost_millions'].sum():.1f}m")
    print(f"  Remaining: £{TOTAL_BUDGET - squad_sorted['cost_millions'].sum():.1f}m")
    
    print(f"\n📊 SQUAD COMPOSITION:")
    print(f"  GKP: {len(squad_sorted[squad_sorted['position'] == 'GKP'])} players")
    print(f"  DEF: {len(squad_sorted[squad_sorted['position'] == 'DEF'])} players")
    print(f"  MID: {len(squad_sorted[squad_sorted['position'] == 'MID'])} players")
    print(f"  FWD: {len(squad_sorted[squad_sorted['position'] == 'FWD'])} players")
    print(f"  TOTAL: {len(squad_sorted)} players")
    
    print(f"\n🎯 STARTING XI:")
    print(f"  Formation: {best_formation[1]}-{best_formation[2]}-{best_formation[3]}")
    print(f"  Expected Avg Points/GW: {best_xi['avg_points_per_gw'].sum():.1f}")
    print(f"  Historical Total (GW1-7): {int(best_xi['total_points_sum'].sum())}")
    
    print(f"\n👑 CAPTAINCY:")
    print(f"  Captain: {captain['web_name']} ({captain['team_name']})")
    print(f"  Vice-Captain: {vice_captain['web_name']} ({vice_captain['team_name']})")
    print(f"  Expected Captain Points: {captain['avg_points_per_gw'] * 2:.1f} (with armband)")
    
    print(f"\n📈 SQUAD STRENGTH METRICS:")
    print(f"  Average Cost per Player: £{squad_sorted['cost_millions'].mean():.1f}m")
    print(f"  Average Points per Player (GW): {squad_sorted['avg_points_per_gw'].mean():.1f}")
    print(f"  Average Value Score: {squad_sorted['value_score'].mean():.1f} pts/£m")
    print(f"  Total Historical Points: {int(squad_sorted['total_points_sum'].sum())}")
    
    print(f"\n🎲 KEY INSIGHTS:")
    most_expensive = squad_sorted.nlargest(1, 'cost_millions').iloc[0]
    best_value = squad_sorted.nlargest(1, 'value_score').iloc[0]
    highest_scorer = squad_sorted.nlargest(1, 'total_points_sum').iloc[0]
    best_form = squad_sorted.nlargest(1, 'form_numeric').iloc[0]
    
    print(f"  Most Expensive: {most_expensive['web_name']} (£{most_expensive['cost_millions']:.1f}m)")
    print(f"  Best Value: {best_value['web_name']} ({best_value['value_score']:.1f} pts/£m)")
    print(f"  Highest Scorer (GW1-7): {highest_scorer['web_name']} ({int(highest_scorer['total_points_sum'])} pts)")
    print(f"  Best Form: {best_form['web_name']} (Form: {best_form['form_numeric']:.1f})")
    
    print("\n" + "=" * 70)
    print("✅ GW8 TEAM BUILDING COMPLETE!")
    print("=" * 70)
    print("\n💡 NEXT STEPS:")
    print("  1. Review the recommended squad and starting XI above")
    print("  2. Make any adjustments based on your preferences")
    print("  3. Check for injury news and press conferences")
    print("  4. Set your captain and vice-captain")
    print("  5. Confirm your team before the GW8 deadline")
    print("\n🍀 Good luck with GW8!")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("\n" + "=" * 70)
    print("  🏆 FPL GW8 TEAM BUILDER & STRATEGY")
    print("  Complete Analysis Based on GW1-GW7 Performance")
    print("=" * 70)
    
    # 1. Load all data
    players_master, teams_master, gameweek_summaries = load_master_data()
    gw8_fixtures, gw8_players, gw8_teams = load_gw8_data()
    historical_stats = load_historical_performance()
    
    # 2. Enrich data
    historical_stats, team_name_map = enrich_historical_data(historical_stats, players_master, teams_master)
    
    # 3. Calculate performance metrics
    player_performance = calculate_player_performance(historical_stats)
    player_performance = add_current_data(player_performance, gw8_players)
    
    # 4. Analyze fixtures
    team_fixture_difficulty = analyze_fixtures(gw8_fixtures, gw8_teams, team_name_map)
    
    # 5. Rank players
    active_players = rank_players(player_performance, team_fixture_difficulty)
    show_position_recommendations(active_players)
    
    # 6. Build squad
    squad_df = build_optimal_squad(active_players, TOTAL_BUDGET)
    squad_sorted = display_squad(squad_df)
    
    # 7. Select starting XI
    best_xi, best_formation, bench = select_starting_xi(squad_sorted)
    captain, vice_captain = select_captain(best_xi)
    
    # 8. Create visualizations
    create_visualizations(squad_sorted)
    
    # 9. Export results
    export_results(squad_sorted, best_xi)
    
    # 10. Print final summary
    print_final_summary(squad_sorted, best_xi, best_formation, captain, vice_captain)


if __name__ == "__main__":
    main()
