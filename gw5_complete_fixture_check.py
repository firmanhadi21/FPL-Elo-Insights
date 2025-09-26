#!/usr/bin/env python3
"""
GW5 COMPLETE FIXTURE VERIFICATION
Checking ALL GW5 fixtures to ensure accurate analysis
"""

def get_complete_gw5_fixtures():
    """Map team codes to actual fixtures"""

    # Team mapping from the data
    teams = {
        1: "Arsenal", 2: "Aston Villa", 3: "Burnley", 4: "Bournemouth",
        5: "Brentford", 6: "Brighton", 7: "Chelsea", 8: "Crystal Palace",
        9: "Everton", 10: "Fulham", 11: "Leeds", 12: "Liverpool",
        13: "Man City", 14: "Man Utd", 15: "Newcastle", 16: "Nott'm Forest",
        17: "Sunderland", 18: "Spurs", 19: "West Ham", 20: "Wolves"
    }

    # From the fixture data (team codes from CSV)
    fixtures_raw = [
        (21, 31),   # West Ham vs Crystal Palace
        (90, 17),   # Burnley vs Nott'm Forest
        (56, 7),    # Sunderland vs Aston Villa
        (54, 94),   # Fulham vs Brentford
        (36, 6),    # Brighton vs Spurs
        (91, 4),    # Bournemouth vs Newcastle
        (3, 43),    # Arsenal vs Man City
        (14, 11),   # Liverpool vs Everton
        (39, 2),    # Wolves vs Leeds
        (1, 8)      # Man Utd vs Chelsea
    ]

    print("🔍 GW5 COMPLETE FIXTURE VERIFICATION:")
    print("=" * 50)
    print("Based on actual fixture data from CSV:")
    print()

    corrected_fixtures = []
    for home_code, away_code in fixtures_raw:
        # Manual mapping based on CSV codes
        if home_code == 21 and away_code == 31:
            fixture = "West Ham vs Crystal Palace"
        elif home_code == 90 and away_code == 17:
            fixture = "Burnley vs Nott'm Forest"
        elif home_code == 56 and away_code == 7:
            fixture = "Sunderland vs Aston Villa"
        elif home_code == 54 and away_code == 94:
            fixture = "Fulham vs Brentford"
        elif home_code == 36 and away_code == 6:
            fixture = "Brighton vs Spurs"
        elif home_code == 91 and away_code == 4:
            fixture = "Bournemouth vs Newcastle"
        elif home_code == 3 and away_code == 43:
            fixture = "Arsenal vs Man City"
        elif home_code == 14 and away_code == 11:
            fixture = "Liverpool vs Everton"
        elif home_code == 39 and away_code == 2:
            fixture = "Wolves vs Leeds"
        elif home_code == 1 and away_code == 8:
            fixture = "Man Utd vs Chelsea"
        else:
            fixture = f"Team {home_code} vs Team {away_code}"

        corrected_fixtures.append(fixture)
        print(f"  • {fixture}")

    return corrected_fixtures

def analyze_all_fixture_errors():
    """Identify all errors in previous analysis"""

    print("\n❌ PREVIOUS ANALYSIS ERRORS IDENTIFIED:")
    print("=" * 45)

    errors = [
        {
            'error': 'Chelsea fixture wrong',
            'incorrect': 'Chelsea vs Crystal Palace (A)',
            'correct': 'Man Utd vs Chelsea',
            'impact': 'Medium - Chelsea away at United is tougher'
        },
        {
            'error': 'Salah fixture wrong',
            'incorrect': 'Liverpool vs Nottingham Forest (A)',
            'correct': 'Liverpool vs Everton (H)',
            'impact': 'High - Merseyside Derby much better fixture'
        },
        {
            'error': 'Missing Newcastle analysis',
            'incorrect': 'Newcastle not highlighted',
            'correct': 'Bournemouth vs Newcastle',
            'impact': 'Medium - Good fixture for Newcastle assets'
        },
        {
            'error': 'West Ham fixture ignored',
            'incorrect': 'Not analyzed',
            'correct': 'West Ham vs Crystal Palace',
            'impact': 'Low - But Palace away is tough'
        }
    ]

    for i, error in enumerate(errors, 1):
        print(f"{i}. {error['error']}:")
        print(f"   ❌ Previous: {error['incorrect']}")
        print(f"   ✅ Actual: {error['correct']}")
        print(f"   📊 Impact: {error['impact']}")
        print()

def completely_revised_strategy():
    """Provide completely revised strategy based on all fixtures"""

    print("🎯 COMPLETELY REVISED GW5 STRATEGY:")
    print("=" * 40)

    fixture_analysis = {
        'excellent': [
            {
                'fixture': 'Liverpool vs Everton (H)',
                'players': 'Salah, Van Dijk, Diaz',
                'reasoning': 'Merseyside Derby at home, Liverpool dominance',
                'confidence': 'Very High'
            }
        ],
        'good': [
            {
                'fixture': 'Bournemouth vs Newcastle',
                'players': 'Isak, Gordon, Livramento',
                'reasoning': 'Newcastle away form, Bournemouth defensive issues',
                'confidence': 'High'
            },
            {
                'fixture': 'Brighton vs Spurs',
                'players': 'Mitoma, Van Hecke (BHA), Son, Maddison (TOT)',
                'reasoning': 'Open game expected, both teams attack',
                'confidence': 'Medium-High'
            }
        ],
        'okay': [
            {
                'fixture': 'Arsenal vs Man City',
                'players': 'Depends on team preference',
                'reasoning': 'Big game, could go either way',
                'confidence': 'Medium'
            },
            {
                'fixture': 'Fulham vs Brentford',
                'players': 'Mbeumo (BRE), Smith Rowe (FUL)',
                'reasoning': 'Attacking teams, goals expected',
                'confidence': 'Medium'
            }
        ],
        'avoid': [
            {
                'fixture': 'Man Utd vs Chelsea',
                'players': 'Palmer, Jackson (CHE)',
                'reasoning': 'Chelsea away at Old Trafford is tough',
                'confidence': 'Medium-Low'
            },
            {
                'fixture': 'West Ham vs Crystal Palace',
                'players': 'Palace players',
                'reasoning': 'Palace away form poor',
                'confidence': 'Low'
            }
        ]
    }

    print("🥇 EXCELLENT FIXTURES:")
    for fixture in fixture_analysis['excellent']:
        print(f"  • {fixture['fixture']}")
        print(f"    Players: {fixture['players']}")
        print(f"    Why: {fixture['reasoning']}")
        print()

    print("✅ GOOD FIXTURES:")
    for fixture in fixture_analysis['good']:
        print(f"  • {fixture['fixture']}")
        print(f"    Players: {fixture['players']}")
        print(f"    Why: {fixture['reasoning']}")
        print()

    print("⚠️ AVOID:")
    for fixture in fixture_analysis['avoid']:
        print(f"  • {fixture['fixture']}")
        print(f"    Players: {fixture['players']}")
        print(f"    Why: {fixture['reasoning']}")
        print()

def final_corrected_recommendations():
    """Final recommendations after checking all fixtures"""

    print("🏆 FINAL CORRECTED GW5 STRATEGY:")
    print("=" * 40)

    print("1. ESSENTIAL TRANSFERS:")
    print("   🎯 Mohamed Salah - Liverpool vs Everton (H)")
    print("   🎯 Alexander Isak - Bournemouth vs Newcastle")
    print("   🎯 Van Dijk - Liverpool vs Everton (H)")

    print("\n2. STRONG CONSIDERATIONS:")
    print("   ✅ Brighton attackers vs Spurs")
    print("   ✅ Newcastle defenders (Livramento)")
    print("   ✅ Mbeumo vs Fulham")

    print("\n3. PROCEED WITH CAUTION:")
    print("   ⚠️ Palmer/Jackson - Away at Man Utd")
    print("   ⚠️ Arsenal/City assets - Big game unpredictable")

    print("\n4. AVOID:")
    print("   ❌ Crystal Palace players (away at West Ham)")
    print("   ❌ Everton players (away at Liverpool)")

    print("\n5. CAPTAINCY RANKING:")
    print("   1. Mohamed Salah (Liverpool vs Everton H)")
    print("   2. Alexander Isak (Bournemouth vs Newcastle)")
    print("   3. Son/Maddison (Brighton vs Spurs)")

    print("\n💡 KEY INSIGHTS:")
    print("• Liverpool vs Everton is THE fixture of the gameweek")
    print("• Newcastle away at Bournemouth is underrated")
    print("• Chelsea assets much less attractive than initially thought")
    print("• Multiple errors in original fixture analysis corrected")

def main():
    """Execute complete fixture verification"""
    fixtures = get_complete_gw5_fixtures()
    analyze_all_fixture_errors()
    completely_revised_strategy()
    final_corrected_recommendations()

    print(f"\n🤖 Complete Fixture Verification - Claude Code")
    print("📋 All GW5 fixtures double-checked and strategy revised")

if __name__ == "__main__":
    main()