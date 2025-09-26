#!/usr/bin/env python3
"""
COMPREHENSIVE FPL ANALYSIS FRAMEWORK
A complete framework for data-driven FPL strategy development with verification
"""

import os
import csv
import re

class FPLAnalysisFramework:
    """Comprehensive framework for FPL analysis with data verification"""
    
    def __init__(self, season="2025-2026", gameweek=5):
        self.season = season
        self.gameweek = gameweek
        self.data_path = f"data/{season}/By Gameweek/GW{gameweek}"
        self.team_names = self.load_team_names()
        self.players = self.load_player_data()
        self.verification_log = []
    
    def verify_data_availability(self):
        """Verify that all required data files are available"""
        print("🔍 DATA AVAILABILITY VERIFICATION")
        print("=" * 40)
        
        required_paths = [
            f"{self.data_path}/players.csv",
            f"{self.data_path}/teams.csv",
            f"{self.data_path}/matches.csv"
        ]
        
        all_available = True
        for path in required_paths:
            if os.path.exists(path):
                print(f"✅ {path}")
                self.verification_log.append(f"AVAILABLE: {path}")
            else:
                print(f"❌ {path} - MISSING")
                self.verification_log.append(f"MISSING: {path}")
                all_available = False
        
        return all_available
    
    def load_team_names(self):
        """Load team names from teams.csv"""
        team_names = {}
        try:
            teams_path = f"{self.data_path}/teams.csv"
            if os.path.exists(teams_path):
                with open(teams_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        team_code = str(row.get('code'))  # Using 'code' as the key
                        team_name = row.get('name', f"Team {team_code}")
                        if team_code:
                            team_names[team_code] = team_name
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
                print(f"✅ Loaded {len(players)} players")
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
    
    def categorize_fixtures(self):
        """Categorize fixtures by quality and difficulty"""
        matches = self.load_fixture_data()
        
        if not matches:
            print("❌ No fixture data available")
            return {}
        
        print("\n⚽ FIXTURE CATEGORIZATION")
        print("=" * 30)
        
        # Categorize fixtures
        premium_fixtures = []      # High-quality PL matches with top teams
        favorable_fixtures = []    # Good fixtures for players
        challenging_fixtures = []  # Tough fixtures
        european_fixtures = []     # European competition matches
        
        for match in matches:
            home_team_code = str(match.get('home_team', ''))
            away_team_code = str(match.get('away_team', ''))
            
            home_team = self.team_names.get(home_team_code, f"Team {home_team_code}")
            away_team = self.team_names.get(away_team_code, f"Team {away_team_code}")
            
            # Check tournament type
            tournament = match.get('tournament', 'prem')
            
            # European fixtures
            if tournament != 'prem':
                european_fixtures.append({
                    'fixture': f"{home_team} vs {away_team}",
                    'tournament': tournament.upper(),
                    'home_team': home_team,
                    'away_team': away_team
                })
            # Premium Premier League fixtures
            elif any(pl_team in home_team or pl_team in away_team 
                     for pl_team in ['Liverpool', 'Man City', 'Chelsea', 'Arsenal', 'Man Utd', 'Newcastle']):
                premium_fixtures.append({
                    'fixture': f"{home_team} vs {away_team}",
                    'home_team': home_team,
                    'away_team': away_team
                })
            # Favorable fixtures (weaker teams at home)
            elif any(weaker_team in home_team 
                     for weaker_team in ['Burnley', 'Bournemouth', 'Sunderland']):
                favorable_fixtures.append({
                    'fixture': f"{home_team} vs {away_team}",
                    'home_team': home_team,
                    'away_team': away_team
                })
            else:
                challenging_fixtures.append({
                    'fixture': f"{home_team} vs {away_team}",
                    'home_team': home_team,
                    'away_team': away_team
                })
        
        print(f"\n🏆 PREMIUM FIXTURES ({len(premium_fixtures)}):")
        for fixture in premium_fixtures:
            print(f"  • {fixture['fixture']}")
        
        print(f"\n🌟 FAVORABLE FIXTURES ({len(favorable_fixtures)}):")
        for fixture in favorable_fixtures:
            print(f"  • {fixture['fixture']}")
        
        print(f"\n🇪🇺 EUROPEAN COMPETITION ({len(european_fixtures)}):")
        for fixture in european_fixtures:
            print(f"  • {fixture['fixture']} ({fixture['tournament']})")
        
        print(f"\n⚠️  CHALLENGING FIXTURES ({len(challenging_fixtures)}):")
        for fixture in challenging_fixtures:
            print(f"  • {fixture['fixture']}")
        
        return {
            'premium': premium_fixtures,
            'favorable': favorable_fixtures,
            'challenging': challenging_fixtures,
            'european': european_fixtures
        }
    
    def identify_fatigue_risk(self, fixtures):
        """Identify players at risk from European fatigue"""
        if not fixtures or 'european' not in fixtures:
            return []
        
        european_teams = set()
        for fixture in fixtures['european']:
            european_teams.add(fixture['home_team'])
            european_teams.add(fixture['away_team'])
        
        print(f"\n⚠️  FATIGUE RISK TEAMS:")
        for team in european_teams:
            if team and not team.startswith("Team "):
                print(f"  • {team}")
        
        return list(european_teams)
    
    def analyze_form_players(self):
        """Analyze players based on recent form"""
        print("\n📈 PLAYER FORM ANALYSIS")
        print("=" * 25)
        
        # Load player gameweek stats for form analysis
        try:
            stats_path = f"{self.data_path}/player_gameweek_stats.csv"
            if os.path.exists(stats_path):
                with open(stats_path, 'r') as f:
                    reader = csv.DictReader(f)
                    stats = list(reader)
                
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
                
                print("🔥 TOP IN-FORM PLAYERS:")
                count = 0
                for player_id, data in sorted_players:
                    if count >= 8:  # Top 8 players
                        break
                    
                    player_name = player_names.get(player_id, f"Player {player_id}")
                    average_points = data.get('average', 0)
                    if average_points > 5:  # Only show players averaging more than 5 points
                        print(f"  • {player_name}: {average_points:.1f} pts/gw")
                        count += 1
                
                if count == 0:
                    print("  • No high-form players identified")
            else:
                print("  • No player gameweek stats available")
        except Exception as e:
            print(f"  • Error analyzing form players: {e}")
    
    def recommend_transfers(self, fixtures, fatigue_teams):
        """Provide transfer recommendations"""
        print("\n🔄 TRANSFER RECOMMENDATIONS")
        print("=" * 30)
        
        # Players from teams with favorable fixtures (and not in European competition)
        favorable_teams = ['Burnley', 'Bournemouth', 'Sunderland']
        
        # Remove teams that have European matches
        safe_favorable_teams = [team for team in favorable_teams if team not in fatigue_teams]
        
        if safe_favorable_teams:
            print("✅ SAFE DIFFERENTIAL OPTIONS:")
            print("   (Favorable fixtures without European fatigue risk)")
            for team in safe_favorable_teams:
                print(f"  • {team} players")
        else:
            print("⚠️  Limited safe differential options this gameweek")
        
        # Premium picks from well-rested teams
        premium_teams = ['Liverpool', 'Man City', 'Chelsea', 'Arsenal', 'Man Utd', 'Newcastle']
        safe_premium_teams = [team for team in premium_teams if team not in fatigue_teams]
        
        if safe_premium_teams:
            print("\n🏆 PREMIUM OPTIONS:")
            print("   (Top teams without European fatigue risk)")
            for team in safe_premium_teams:
                print(f"  • {team} players")
        else:
            print("\n⚠️  Premium options may be affected by European fatigue")
    
    def captaincy_advice(self, fixtures, fatigue_teams):
        """Provide captaincy recommendations"""
        print("\n👑 CAPTAINCY RECOMMENDATIONS")
        print("=" * 30)
        
        # Best captain options: Home advantage + No European fatigue
        print("🥇 TOP CAPTAIN CHOICES:")
        print("  1. Players from premium fixtures with home advantage")
        print("  2. In-form players from teams without European matches")
        print("  3. Assets from favorable fixtures (Burnley, Bournemouth, Sunderland)")
        
        print("\n⚠️  AVOID AS CAPTAIN:")
        print("  1. Players from teams with European competition")
        for team in fatigue_teams:
            if team and not team.startswith("Team "):
                print(f"     • {team}")
        print("  2. Players from challenging away fixtures")
    
    def run_complete_analysis(self):
        """Run complete FPL analysis"""
        print(f"📊 COMPREHENSIVE FPL ANALYSIS - GW{self.gameweek} {self.season}")
        print("=" * 60)
        
        # Step 1: Verify data availability
        data_ok = self.verify_data_availability()
        
        if not data_ok:
            print("\n❌ CRITICAL: Missing required data files")
            print("Please ensure the repository is up to date")
            return
        
        # Step 2: Categorize fixtures
        fixtures = self.categorize_fixtures()
        
        # Step 3: Identify fatigue risk
        fatigue_teams = self.identify_fatigue_risk(fixtures)
        
        # Step 4: Analyze form players
        self.analyze_form_players()
        
        # Step 5: Provide transfer recommendations
        self.recommend_transfers(fixtures, fatigue_teams)
        
        # Step 6: Captaincy advice
        self.captaincy_advice(fixtures, fatigue_teams)
        
        # Step 7: Summary
        print(f"\n📋 STRATEGY SUMMARY:")
        print("=" * 20)
        print("✅ Focus on actual fixture data from matches.csv")
        print("✅ Consider European fatigue for teams with continental matches")
        print("✅ Balance premium assets with safe differential picks")
        print("✅ Prioritize well-rested players over historical reputation")
        
        # Verification summary
        print(f"\n🔍 VERIFICATION SUMMARY:")
        print("=" * 22)
        print(f"Total checks performed: {len(self.verification_log)}")
        issues = [log for log in self.verification_log if "MISSING" in log]
        if issues:
            print(f"⚠️  Issues found: {len(issues)}")
            for issue in issues:
                print(f"  • {issue}")
        else:
            print("✅ All data verification checks passed")

def main():
    """Main function"""
    print("🚀 FPL COMPREHENSIVE ANALYSIS FRAMEWORK")
    print("=" * 50)
    print("Data-driven strategy development with verification\n")
    
    # Create and run analysis
    framework = FPLAnalysisFramework(season="2025-2026", gameweek=5)
    framework.run_complete_analysis()
    
    print(f"\n💡 TIPS FOR FUTURE GAMEWEEKS:")
    print("=" * 30)
    print("1. ALWAYS verify fixture data against matches.csv")
    print("2. Check for European competition matches")
    print("3. Cross-reference team names between files")
    print("4. Validate assumptions with raw data")
    print("5. Document all findings for future reference")

if __name__ == "__main__":
    main()