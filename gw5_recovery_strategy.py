#!/usr/bin/env python3
"""
GW5 RECOVERY STRATEGY: MAXIMIZE POINTS AFTER MISSING HAALAND
Strategic recommendations to catch up after missing Haaland's double in GW4
"""

def analyze_gw5_data():
    """Analyze current player form and upcoming fixtures for GW5"""

    print("🚀 GW5 RECOVERY STRATEGY: BOUNCE BACK STRONG")
    print("=" * 80)
    print("🎯 Mission: Strategic moves to recover rank after missing Haaland's haul")
    print()

    # Key insights from GW4 performance
    print("📊 GW4 REALITY CHECK:")
    print("=" * 30)
    gw4_highlights = {
        'haaland_performance': '9 points (2 goals vs Brighton)',
        'top_scorers': [
            'Cole Palmer - 18 points (hat-trick vs Brighton)',
            'Bryan Mbeumo - 14 points (1G, 1A vs Liverpool)',
            'Mohamed Salah - 16 points (2G, 2A vs West Ham)',
            'Virgil van Dijk - 8 points (clean sheet + goal)',
            'Nicolas Jackson - 15 points (2 goals vs Brighton)'
        ],
        'key_lesson': 'Premium players delivered - need exposure to top performers'
    }

    print(f"❌ Haaland: {gw4_highlights['haaland_performance']}")
    print("✅ Top GW4 Performers:")
    for performer in gw4_highlights['top_scorers']:
        print(f"  • {performer}")
    print(f"\n💡 {gw4_highlights['key_lesson']}")

    return gw4_highlights

def gw5_transfer_strategy():
    """Recommend specific transfers for GW5"""

    print("\n🔄 GW5 TRANSFER PRIORITIES:")
    print("=" * 35)

    # Priority transfer targets based on GW5 fixtures and form
    transfers = {
        'premium_forwards': {
            'haaland': {
                'price': '£14.1m',
                'fixture': 'Arsenal (A)',
                'rationale': 'Must-have despite tough fixture - proven scorer',
                'priority': 'HIGH',
                'risk': 'Medium (tough fixture but City bounce back potential)'
            },
            'jackson': {
                'price': '£6.5m',
                'fixture': 'Crystal Palace (A)',
                'rationale': 'In form, scored 2 vs Brighton, favorable fixture',
                'priority': 'MEDIUM',
                'risk': 'Low'
            }
        },
        'premium_midfielders': {
            'palmer': {
                'price': '£10.5m',
                'fixture': 'Crystal Palace (A)',
                'rationale': 'Hat-trick hero, creative force, good fixture',
                'priority': 'VERY HIGH',
                'risk': 'Low'
            },
            'salah': {
                'price': '£14.5m',
                'fixture': 'Nottingham Forest (A)',
                'rationale': 'Most consistent premium, 16 pts in GW4',
                'priority': 'HIGH',
                'risk': 'Very Low'
            },
            'mbeumo': {
                'price': '£8.0m',
                'fixture': 'Manchester City (H)',
                'rationale': 'Excellent value, 14 pts vs Liverpool shows ceiling',
                'priority': 'MEDIUM',
                'risk': 'Medium (tough fixture)'
            }
        },
        'defenders': {
            'van_dijk': {
                'price': '£6.0m',
                'fixture': 'Nottingham Forest (A)',
                'rationale': 'Scored in GW4, Liverpool defensive stability',
                'priority': 'HIGH',
                'risk': 'Low'
            },
            'gabriel': {
                'price': '£6.1m',
                'fixture': 'Manchester City (H)',
                'rationale': 'Arsenal defense, goal threat from set pieces',
                'priority': 'MEDIUM',
                'risk': 'Medium'
            }
        }
    }

    print("🎯 PRIORITY 1: ESSENTIAL MOVES")
    print("-" * 30)
    print(f"1. Cole Palmer ({transfers['premium_midfielders']['palmer']['price']})")
    print(f"   Fixture: {transfers['premium_midfielders']['palmer']['fixture']}")
    print(f"   Why: {transfers['premium_midfielders']['palmer']['rationale']}")
    print(f"   Risk: {transfers['premium_midfielders']['palmer']['risk']}")

    print(f"\n2. Mohamed Salah ({transfers['premium_midfielders']['salah']['price']})")
    print(f"   Fixture: {transfers['premium_midfielders']['salah']['fixture']}")
    print(f"   Why: {transfers['premium_midfielders']['salah']['rationale']}")
    print(f"   Risk: {transfers['premium_midfielders']['salah']['risk']}")

    print("\n🎯 PRIORITY 2: HAALAND DECISION")
    print("-" * 30)
    print(f"Erling Haaland ({transfers['premium_forwards']['haaland']['price']})")
    print(f"Fixture: {transfers['premium_forwards']['haaland']['fixture']}")
    print(f"Why: {transfers['premium_forwards']['haaland']['rationale']}")
    print(f"Risk: {transfers['premium_forwards']['haaland']['risk']}")
    print("💭 Decision: Consider bringing in for GW6 if budget allows - GW5 fixture is tough")

    return transfers

def captaincy_recommendations():
    """Provide captaincy options for GW5"""

    print("\n👑 GW5 CAPTAINCY STRATEGY:")
    print("=" * 35)

    captains = [
        {
            'player': 'Mohamed Salah',
            'fixture': 'Nottingham Forest (A)',
            'ownership': '54.6%',
            'predicted_points': '12-15',
            'rationale': 'Most reliable premium, favorable away fixture',
            'risk_level': 'Low',
            'differential_value': 'Safe template choice'
        },
        {
            'player': 'Cole Palmer',
            'fixture': 'Crystal Palace (A)',
            'ownership': '63.4%',
            'predicted_points': '10-14',
            'rationale': 'Hat-trick form, Palace away is good fixture',
            'risk_level': 'Low-Medium',
            'differential_value': 'Popular but explosive potential'
        },
        {
            'player': 'Nicolas Jackson',
            'fixture': 'Crystal Palace (A)',
            'ownership': '0.1%',
            'predicted_points': '8-12',
            'rationale': 'Massive differential, in form, Palace struggle',
            'risk_level': 'High',
            'differential_value': 'Huge rank gains if successful'
        }
    ]

    print("🥇 RECOMMENDED CAPTAIN: Mohamed Salah")
    print("   Most consistent performer, proven against Forest-level opposition")
    print("   Safe choice to minimize further rank damage")

    print(f"\n🎲 DIFFERENTIAL OPTION: Nicolas Jackson")
    print("   Ultra-low ownership, Palace away, recent form excellent")
    print("   High risk but massive reward potential")

    for i, captain in enumerate(captains, 1):
        print(f"\n{i}. {captain['player']}")
        print(f"   Fixture: {captain['fixture']}")
        print(f"   Ownership: {captain['ownership']}")
        print(f"   Predicted: {captain['predicted_points']} points")
        print(f"   Risk: {captain['risk_level']}")
        print(f"   Rationale: {captain['rationale']}")

    return captains

def budget_strategies():
    """Provide different strategies based on budget constraints"""

    print("\n💰 BUDGET-BASED STRATEGIES:")
    print("=" * 35)

    strategies = {
        'premium_heavy': {
            'budget_required': '£35m+',
            'structure': 'Salah + Palmer + Haaland',
            'pros': 'Maximum ceiling, template protection',
            'cons': 'Expensive, limits squad depth',
            'recommendation': 'If you can afford it, go for it'
        },
        'balanced': {
            'budget_required': '£25-35m',
            'structure': 'Salah + Palmer + Jackson',
            'pros': 'Good balance, solid floor with upside',
            'cons': 'Missing Haaland still a risk',
            'recommendation': 'Solid approach for most managers'
        },
        'value_focused': {
            'budget_required': '£15-25m',
            'structure': 'Palmer + Mbeumo + Van Dijk',
            'pros': 'Great value picks, leaves budget flexibility',
            'cons': 'No premium forward coverage',
            'recommendation': 'Risky but could pay off with differentials'
        }
    }

    for strategy_name, details in strategies.items():
        print(f"\n🎯 {strategy_name.upper().replace('_', ' ')} STRATEGY:")
        print(f"   Budget: {details['budget_required']}")
        print(f"   Core: {details['structure']}")
        print(f"   ✅ Pros: {details['pros']}")
        print(f"   ❌ Cons: {details['cons']}")
        print(f"   💡 Verdict: {details['recommendation']}")

def gw5_fixtures_analysis():
    """Analyze key GW5 fixtures for FPL perspective"""

    print("\n⚽ KEY GW5 FIXTURES ANALYSIS:")
    print("=" * 40)

    key_fixtures = [
        {
            'fixture': 'Arsenal vs Manchester City',
            'fpl_angle': 'Premium midfielders clash - Haaland tough fixture',
            'picks': 'Gabriel (ARS def), avoid City attackers'
        },
        {
            'fixture': 'Crystal Palace vs Chelsea',
            'fpl_angle': 'Palace struggle vs top 6 - Chelsea assets favorable',
            'picks': 'Palmer, Jackson captain options'
        },
        {
            'fixture': 'Nottingham Forest vs Liverpool',
            'fpl_angle': 'Liverpool away form strong, Forest leaky defense',
            'picks': 'Salah captaincy, Van Dijk, avoid Forest'
        },
        {
            'fixture': 'Brentford vs Manchester City',
            'fpl_angle': 'Mbeumo home form vs City rotation risk',
            'picks': 'Mbeumo differential, City assets uncertain'
        }
    ]

    for fixture in key_fixtures:
        print(f"\n🏟️  {fixture['fixture']}")
        print(f"   FPL Angle: {fixture['fpl_angle']}")
        print(f"   Picks: {fixture['picks']}")

def main():
    """Main function to execute GW5 strategy"""

    # Analyze current situation
    gw4_data = analyze_gw5_data()

    # Get transfer recommendations
    transfers = gw5_transfer_strategy()

    # Captaincy advice
    captains = captaincy_recommendations()

    # Budget strategies
    budget_strategies()

    # Fixture analysis
    gw5_fixtures_analysis()

    print("\n🏆 FINAL GW5 RECOMMENDATIONS:")
    print("=" * 40)
    print("1. PRIORITY TRANSFERS:")
    print("   • IN: Mohamed Salah + Cole Palmer")
    print("   • Consider Jackson as Haaland alternative")
    print("\n2. CAPTAINCY:")
    print("   • Safe: Mohamed Salah (vs Forest)")
    print("   • Differential: Nicolas Jackson (vs Palace)")
    print("\n3. STRATEGY:")
    print("   • Focus on form players with good fixtures")
    print("   • Avoid knee-jerk Haaland transfer for tough Arsenal fixture")
    print("   • Target Palace, Forest opponents")
    print("\n4. WILDCARD CONSIDERATION:")
    print("   • If team structure is broken, consider WC in GW6")
    print("   • GW5 is recovery gameweek, don't panic")

    print(f"\n🤖 Generated with Claude Code - GW5 Recovery Strategy")
    print("📈 Target: 65+ points to recover rank position")

if __name__ == "__main__":
    main()