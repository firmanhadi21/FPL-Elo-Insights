#!/usr/bin/env python3
"""
GW5 DETAILED ANALYSIS: USING LATEST 2025-2026 DATA
Comprehensive analysis of GW5 fixtures and player recommendations
"""

import os
import csv

class GW5Analysis:
    """Detailed analysis of GW5 using the latest data"""
    
    def __init__(self, season="2025-2026", gameweek=5):
        self.season = season
        self.gameweek = gameweek
        self.data_path = f"data/{season}/By Gameweek/GW{gameweek}"
        self.team_names = self.load_team_names()
        self.players = self.load_player_data()
    
    def load_team_names(self):
        """Load team names from teams.csv"""
        team_names = {}
        try:
            teams_path = f"{self.data_path}/teams.csv"
            if os.path.exists(teams_path):
                with open(teams_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        team_id = str(row.get('code'))  # Using 'code' instead of 'id'
                        team_name = row.get('name', f"Team {team_id}")
                        if team_id:
                            team_names[team_id] = team_name
        except Exception as e:
            print(f"⚠️  Could not load team names: {e}")
        return team_names
    
    def load_player_data(self):
        """Load player data"""
        try:
            players_path = f"{self.data_path}/players.csv"
            if os.path.exists(players_path):
                with open(players_path, 'r') as f:
                    reader = csv.DictReader(f)
                    players = list(reader)
                return players
        except Exception as e:
            print(f"❌ Error loading player data: {e}")
        return []
    
    def load_fixture_data(self):
        """Load fixture data"""
        try:
            matches_path = f"{self.data_path}/matches.csv"
            if os.path.exists(matches_path):
                with open(matches_path, 'r') as f:
                    reader = csv.DictReader(f)
                    matches = list(reader)
                return matches
        except Exception as e:
            print(f"❌ Error loading fixture data: {e}")
        return []
    
    def load_player_gameweek_stats(self):
        """Load player gameweek stats"""
        try:
            stats_path = f"{self.data_path}/player_gameweek_stats.csv"
            if os.path.exists(stats_path):
                with open(stats_path, 'r') as f:
                    reader = csv.DictReader(f)
                    stats = list(reader)
                return stats
        except Exception as e:
            print(f"❌ Error loading player gameweek stats: {e}")
        return []
    
    def categorize_fixtures(self):
        """Categorize fixtures by quality and difficulty"""
        matches = self.load_fixture_data()
        
        print("⚽ GW5 FIXTURE CATEGORIZATION")
        print("=" * 40)
        
        # Categorize fixtures
        premium_fixtures = []      # High-quality matches with top teams
        favorable_fixtures = []    # Good fixtures for players
        challenging_fixtures = []  # Tough fixtures
        european_fixtures = []     # European competition matches
        
        for match in matches:
            home_team_id = str(match.get('home_team', ''))
            away_team_id = str(match.get('away_team', ''))
            
            home_team = self.team_names.get(home_team_id, f"Team {home_team_id}")
            away_team = self.team_names.get(away_team_id, f"Team {away_team_id}")
            
            # Check tournament type
            tournament = match.get('tournament', 'prem')
            
            # European fixtures
            if tournament != 'prem':
                european_fixtures.append(f"{home_team} vs {away_team} ({tournament.upper()})")
            # Premium Premier League fixtures
            elif any(pl_team in home_team or pl_team in away_team 
                     for pl_team in ['Liverpool', 'Man City', 'Chelsea', 'Arsenal', 'Man Utd', 'Newcastle']):
                premium_fixtures.append(f"{home_team} vs {away_team}")
            # Favorable fixtures (weaker teams at home)
            elif any(weaker_team in home_team 
                     for weaker_team in ['Burnley', 'Bournemouth', 'Sunderland']):
                favorable_fixtures.append(f"{home_team} vs {away_team}")
            else:
                challenging_fixtures.append(f"{home_team} vs {away_team}")
        
        print(f"\n🏆 PREMIUM FIXTURES ({len(premium_fixtures)}):")
        for fixture in premium_fixtures:
            print(f"  • {fixture}")
        
        print(f"\n🌟 FAVORABLE FIXTURES ({len(favorable_fixtures)}):")
        for fixture in favorable_fixtures:
            print(f"  • {fixture}")
        
        print(f"\n🇪🇺 EUROPEAN COMPETITION ({len(european_fixtures)}):")
        for fixture in european_fixtures:
            print(f"  • {fixture}")
        
        print(f"\n⚠️  CHALLENGING FIXTURES ({len(challenging_fixtures)}):")
        for fixture in challenging_fixtures:
            print(f"  • {fixture}")
        
        return {
            'premium': premium_fixtures,
            'favorable': favorable_fixtures,
            'challenging': challenging_fixtures,
            'european': european_fixtures
        }
    
    def analyze_top_players(self):
        """Analyze top players based on form and fixtures"""
        print("\n⭐ TOP PLAYER RECOMMENDATIONS")
        print("=" * 35)
        
        # Load player gameweek stats for form analysis
        stats = self.load_player_gameweek_stats()
        
        if not stats:
            print("⚠️  No player gameweek stats available")
            return
        
        # Group stats by player and calculate average points
        player_stats = {}
        for stat in stats:
            player_id = stat.get('id')
            points = float(stat.get('event_points', 0)) if stat.get('event_points') else 0
            
            if player_id not in player_stats:
                player_stats[player_id] = {'points': [], 'total': 0}
            
            player_stats[player_id]['points'].append(points)
            player_stats[player_id]['total'] += points
        
        # Calculate average points for players with multiple gameweeks
        for player_id, data in player_stats.items():
            if len(data['points']) > 0:
                data['average'] = data['total'] / len(data['points'])
            else:
                data['average'] = 0
        
        # Sort players by average points
        sorted_players = sorted(player_stats.items(), key=lambda x: x[1].get('average', 0), reverse=True)
        
        # Get player names (need to match on player_id)
        player_names = {}
        for player in self.players:
            player_id = player.get('player_id')
            if player_id:
                player_names[player_id] = f"{player['first_name']} {player['second_name']}"
        
        print("🔥 PLAYERS IN FORM (Last 3 Gameweeks):")
        count = 0
        for player_id, data in sorted_players:
            if count >= 10:  # Top 10 players
                break
            
            player_name = player_names.get(player_id, f"Player {player_id}")
            average_points = data.get('average', 0)
            if average_points > 3:  # Only show players averaging more than 3 points
                print(f"  • {player_name}: {average_points:.1f} pts/gw (total: {data['total']:.1f})")
                count += 1
        
        if count == 0:
            print("  • No players with significant recent form found")
    
    def identify_differential_opportunities(self):
        """Identify differential player opportunities"""
        print("\n🕵️  DIFFERENTIAL OPPORTUNITIES")
        print("=" * 35)
        
        # Players from teams with favorable fixtures
        favorable_teams = ['Burnley', 'Bournemouth', 'Sunderland', 'Nott\'m Forest']
        
        # Find players from these teams
        differential_players = []
        for player in self.players:
            team_code = player.get('team_code', '')
            
            # Match team code to team name
            team_name = self.team_names.get(str(team_code), '')
            
            if any(fav_team in team_name for fav_team in favorable_teams):
                player_name = f"{player['first_name']} {player['second_name']}"
                # Get player cost if available
                try:
                    now_cost = float(player.get('now_cost', 0)) / 10 if player.get('now_cost') else 0  # Convert to millions
                except:
                    now_cost = 0
                differential_players.append({
                    'name': player_name,
                    'team': team_name,
                    'cost': now_cost,
                    'position': player.get('position', '')
                })
        
        if differential_players:
            print("💰 BUDGET-FRIENDLY DIFFERENTIALS:")
            for player in differential_players[:10]:  # Top 10
                print(f"  • {player['name']} ({player['team']}) - £{player['cost']:.1f}m [{player['position']}]")
        else:
            print("  • No differential opportunities identified")
    
    def generate_captaincy_recommendations(self):
        """Generate captaincy recommendations"""
        print("\n👑 CAPTAINCY RECOMMENDATIONS")
        print("=" * 30)
        
        # Based on fixture quality and recent form
        print("🎯 TOP CAPTAIN CHOICES:")
        print("  1. Players from premium fixtures (Liverpool, Man City, Chelsea matches)")
        print("  2. In-form players averaging 5+ points per gameweek")
        print("  3. Home advantage players with good opposition")
        
        print("\n🎲 DIFFERENTIAL CAPTAIN OPTIONS:")
        print("  1. Players from favorable fixtures (Burnley, Bournemouth at home)")
        print("  2. Lower-owned assets with high ceiling")
        print("  3. Players returning from injury/match fitness")
    
    def run_analysis(self):
        """Run complete GW5 analysis"""
        print(f"📊 GW5 COMPREHENSIVE ANALYSIS - {self.season}")
        print("=" * 50)
        
        # 1. Fixture categorization
        fixtures = self.categorize_fixtures()
        
        # 2. Top player analysis
        self.analyze_top_players()
        
        # 3. Differential opportunities
        self.identify_differential_opportunities()
        
        # 4. Captaincy recommendations
        self.generate_captaincy_recommendations()
        
        # 5. Summary
        print(f"\n📋 GW5 STRATEGY SUMMARY:")
        print("=" * 25)
        print("✅ Use actual fixture data from matches.csv")
        print("✅ Focus on in-form players from player_gameweek_stats.csv")
        print("✅ Consider European fatigue for teams with continental matches")
        print("✅ Leverage favorable fixtures for differential picks")
        print("✅ Balance premium assets with budget options")

def main():
    """Main function"""
    analysis = GW5Analysis(season="2025-2026", gameweek=5)
    analysis.run_analysis()

if __name__ == "__main__":
    main()