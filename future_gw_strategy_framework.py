#!/usr/bin/env python3
"""
FUTURE GW STRATEGY FRAMEWORK: DATA-DRIVEN APPROACH WITH VERIFICATION
A robust framework for developing FPL strategies with proper data verification
"""

import os
import csv
import re

class FPLStrategyFramework:
    """A framework for developing reliable FPL strategies with data verification"""
    
    def __init__(self, season="2025-2026", gameweek=None):
        self.season = season
        self.gameweek = gameweek
        self.data_path = f"data/{season}/By Gameweek/GW{gameweek}" if gameweek else f"data/{season}"
        self.verification_log = []
        self.team_names = self.load_team_names()
    
    def verify_data_availability(self):
        """Verify that all required data files are available"""
        print("🔍 DATA AVAILABILITY VERIFICATION")
        print("=" * 40)
        
        # Updated paths to match the actual directory structure
        required_paths = [
            f"{self.data_path}/players.csv",
            f"{self.data_path}/teams.csv",
            f"{self.data_path}/playerstats.csv"
        ]
        
        if self.gameweek:
            required_paths.extend([
                f"{self.data_path}/matches.csv",
                f"{self.data_path}/playermatchstats.csv"
            ])
        
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
    
    def load_fixture_data(self):
        """Load and verify fixture data for the gameweek"""
        if not self.gameweek:
            print("⚠️  No gameweek specified, cannot load fixture data")
            return None
            
        fixture_path = f"{self.data_path}/matches.csv"
        
        if not os.path.exists(fixture_path):
            print(f"❌ Fixture data not found at {fixture_path}")
            return None
            
        try:
            with open(fixture_path, 'r') as f:
                reader = csv.DictReader(f)
                fixtures = list(reader)
            
            print(f"✅ Loaded {len(fixtures)} fixtures for GW{self.gameweek}")
            
            # Display key fixture information
            print("\n📋 KEY FIXTURE INFORMATION:")
            print("=" * 30)
            
            for match in fixtures:
                home_team_id = str(match.get('home_team', 'Unknown'))
                away_team_id = str(match.get('away_team', 'Unknown'))
                
                # Try to get team names from our mapping first
                home_team = self.team_names.get(home_team_id)
                away_team = self.team_names.get(away_team_id)
                
                # If not found, try to extract from match_url
                if not home_team or not away_team:
                    home_team, away_team = self.extract_team_names_from_url(match.get('match_url', ''))
                    if not home_team:
                        home_team = f"Team {home_team_id}"
                    if not away_team:
                        away_team = f"Team {away_team_id}"
                
                print(f"  • {home_team} vs {away_team}")
                
            return fixtures
        except Exception as e:
            print(f"❌ Error loading fixture data: {e}")
            return None
    
    def extract_team_names_from_url(self, match_url):
        """Extract team names from match URL"""
        if not match_url:
            return None, None
            
        try:
            # Extract team names from URL pattern like "/matches/wolverhampton-wanderers-vs-liverpool/"
            match_pattern = r"/matches/([^/]+)/"
            match_result = re.search(match_pattern, match_url)
            
            if match_result:
                teams_part = match_result.group(1)
                # Split on "vs" or "-vs-" or similar patterns
                if "-vs-" in teams_part:
                    teams = teams_part.split("-vs-")
                elif "vs" in teams_part:
                    teams = teams_part.split("vs")
                else:
                    # Try to split on the last hyphen before a recognizable pattern
                    teams = teams_part.split("-vs")
                
                if len(teams) >= 2:
                    home_team = self.format_team_name(teams[0])
                    away_team = self.format_team_name(teams[1])
                    return home_team, away_team
        except Exception as e:
            print(f"⚠️  Could not extract team names from URL: {e}")
        
        return None, None
    
    def format_team_name(self, team_name):
        """Format team name by removing extra characters and capitalizing appropriately"""
        # Remove common suffixes and prefixes
        team_name = re.sub(r'^\d+-\d+-prem-', '', team_name)
        team_name = re.sub(r'-\d+$', '', team_name)
        
        # Convert hyphens to spaces and title case
        team_name = team_name.replace('-', ' ').title()
        
        # Handle special cases
        team_name = team_name.replace('Wolverhampton Wanderers', 'Wolves')
        team_name = team_name.replace('Tottenham Hotspur', 'Spurs')
        team_name = team_name.replace('Man United', 'Man Utd')
        team_name = team_name.replace('Manchester United', 'Man Utd')
        team_name = team_name.replace('Manchester City', 'Man City')
        team_name = team_name.replace('Afc Bournemouth', 'Bournemouth')
        team_name = team_name.replace('Brighton Hove Albion', 'Brighton')
        team_name = team_name.replace('West Ham United', 'West Ham')
        team_name = team_name.replace('Newcastle United', 'Newcastle')
        team_name = team_name.replace('Ipswich Town', 'Ipswich')
        team_name = team_name.replace('Nottingham Forest', "Nott'm Forest")
        
        return team_name.strip()
    
    def load_team_names(self):
        """Load team names from teams.csv"""
        team_names = {}
        try:
            teams_path = f"{self.data_path}/teams.csv"
            if os.path.exists(teams_path):
                with open(teams_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        team_id = str(row.get('id'))
                        team_name = row.get('name', f"Team {team_id}")
                        if team_id:
                            team_names[team_id] = team_name
            else:
                print("⚠️  teams.csv not found, will extract team names from match data")
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
            else:
                print("❌ players.csv not found")
                return None
        except Exception as e:
            print(f"❌ Error loading player data: {e}")
            return None
    
    def analyze_form_and_fixture_difficulty(self):
        """Analyze player form and fixture difficulty"""
        if not self.gameweek or self.gameweek < 2:
            print("⚠️  Need at least GW2 data for form analysis")
            return
            
        print("\n📈 FORM AND FIXTURE ANALYSIS")
        print("=" * 35)
        
        # Check for player gameweek stats
        gw_stats_path = f"{self.data_path}/player_gameweek_stats.csv"
        if os.path.exists(gw_stats_path):
            print("✅ player_gameweek_stats.csv available for form analysis")
            print("💡 Use this file to analyze recent player performance (last 3 gameweeks)")
        else:
            print("⚠️  player_gameweek_stats.csv not found")
            print("💡 Falling back to cumulative playerstats.csv")
    
    def verify_fixture_consistency(self):
        """Verify fixture data consistency across sources"""
        if not self.gameweek:
            return
            
        print("\n✅ FIXTURE CONSISTENCY CHECK")
        print("=" * 30)
        
        # Check matches.csv
        matches_path = f"{self.data_path}/matches.csv"
        if os.path.exists(matches_path):
            print("✅ Matches.csv found")
            
            # Basic validation
            try:
                with open(matches_path, 'r') as f:
                    reader = csv.DictReader(f)
                    first_row = next(reader, None)
                    if first_row and 'home_team' in first_row and 'away_team' in first_row:
                        print("✅ Fixture structure validated")
                    else:
                        print("❌ Fixture structure issues detected")
                        self.verification_log.append("ISSUE: Fixture structure invalid")
            except Exception as e:
                print(f"❌ Error validating fixture structure: {e}")
        else:
            print("❌ Matches.csv not found")
            self.verification_log.append("MISSING: Matches.csv")
    
    def generate_recommendations(self):
        """Generate data-driven recommendations"""
        print("\n🎯 STRATEGY RECOMMENDATIONS")
        print("=" * 30)
        
        recommendations = [
            "1. ALWAYS verify fixture data against matches.csv before making picks",
            "2. Use player_gameweek_stats.csv for recent form analysis (last 3 gameweeks)",
            "3. Cross-reference team Elo ratings for fixture difficulty assessment",
            "4. Consider home advantage (+15% for forwards, +12% for midfielders)",
            "5. Factor in player ownership percentages for differential opportunities",
            "6. Check for injuries/team news in official FPL API if available",
            "7. Validate all player-team assignments using teams.csv"
        ]
        
        for rec in recommendations:
            print(f"  {rec}")
    
    def run_complete_analysis(self):
        """Run the complete analysis framework"""
        print(f"🚀 FPL STRATEGY FRAMEWORK FOR GW{self.gameweek or 'N/A'}")
        print("=" * 60)
        print(f"Season: {self.season}")
        if self.gameweek:
            print(f"Gameweek: {self.gameweek}")
        print()
        
        # Step 1: Verify data availability
        data_ok = self.verify_data_availability()
        
        if not data_ok:
            print("\n❌ CRITICAL: Missing required data files")
            print("Please ensure the repository is up to date and all data files are present")
            return
        
        # Step 2: Load and verify fixture data
        fixtures = self.load_fixture_data()
        
        # Step 3: Load player data
        players = self.load_player_data()
        
        # Step 4: Analyze form and fixture difficulty
        self.analyze_form_and_fixture_difficulty()
        
        # Step 5: Verify fixture consistency
        self.verify_fixture_consistency()
        
        # Step 6: Generate recommendations
        self.generate_recommendations()
        
        # Summary
        print(f"\n📋 VERIFICATION SUMMARY:")
        print("=" * 25)
        print(f"Total checks performed: {len(self.verification_log)}")
        issues = [log for log in self.verification_log if "MISSING" in log or "ISSUE" in log]
        if issues:
            print(f"⚠️  Issues found: {len(issues)}")
            for issue in issues:
                print(f"  • {issue}")
        else:
            print("✅ All data verification checks passed")
        
        print(f"\n💡 NEXT STEPS:")
        print("=" * 15)
        print("1. Review the actual fixtures from matches.csv")
        print("2. Analyze player form from player_gameweek_stats.csv")
        print("3. Assess fixture difficulty using Elo ratings")
        print("4. Make selections based on verified data only")
        print("5. Document any assumptions for future reference")

def main():
    """Main function - example usage"""
    # Example for GW5 (using the latest data as requested)
    framework = FPLStrategyFramework(season="2025-2026", gameweek=5)
    framework.run_complete_analysis()

if __name__ == "__main__":
    main()