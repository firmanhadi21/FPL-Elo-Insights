#!/usr/bin/env python3
"""
GW4 REVISED TRANSFER STRATEGY
Corrected for team constraints - already have 3 Chelsea players
Current Chelsea players: João Pedro, Cucurella, Caicedo
"""

def main():
    print("🔄 GW4 REVISED TRANSFER STRATEGY")
    print("=" * 50)
    print("⚠️ CONSTRAINT: Already have 3 Chelsea players (João Pedro, Cucurella, Caicedo)")
    print("🎯 Target: 65+ points with proper team limits")
    print()
    
    print("🚫 ORIGINAL ERROR:")
    print("-" * 20)
    print("❌ Semenyo → Enzo Fernández (Chelsea) - IMPOSSIBLE!")
    print("   Reason: Would give you 4 Chelsea players (limit is 3)")
    
    print(f"\n✅ CORRECTED TRANSFER OPTIONS:")
    print("=" * 40)
    
    # Priority 1: Defender transfer (no constraint issues)
    print("🏆 PRIORITY 1 - DEFENDER:")
    print("-" * 25)
    print("OUT: Pedro Porro (Spurs) - 0 points, poor fixture")
    print("IN:  Calafiori (Arsenal) - vs Forest (H), excellent fixture")
    print("Cost: Neutral (similar pricing)")
    print("Predicted: 6-8 points vs Forest")
    
    # Priority 2: Midfielder - revised options
    print(f"\n🎯 PRIORITY 2 - MIDFIELDER:")
    print("-" * 30)
    print("OUT: Semenyo (Bournemouth) - 2 points, poor form")
    
    revised_mid_options = [
        {
            'player': 'Bukayo Saka (Arsenal)',
            'fixture': 'vs Forest (H)',
            'cost': '+3-4m',
            'predicted': '8-10 points',
            'rationale': 'Premium mid, excellent fixture, high ceiling'
        },
        {
            'player': 'Phil Foden (Man City)', 
            'fixture': 'vs Man Utd (H)',
            'cost': '+2-3m',
            'predicted': '7-9 points', 
            'rationale': 'Derby fixture, rotation risk but huge upside'
        },
        {
            'player': 'Martin Ødegaard (Arsenal)',
            'fixture': 'vs Forest (H)',
            'cost': '+1-2m',
            'predicted': '6-8 points',
            'rationale': 'Good value, Arsenal captain, reliable'
        },
        {
            'player': 'Bruno Fernandes (Man United)',
            'fixture': 'vs Man City (A)', 
            'cost': '+2m',
            'predicted': '6-10 points',
            'rationale': 'Penalties, derby goals, medium differential'
        }
    ]
    
    print("REVISED OPTIONS:")
    for i, option in enumerate(revised_mid_options, 1):
        print(f"\n{i}. {option['player']}")
        print(f"   Fixture: {option['fixture']}")
        print(f"   Cost: {option['cost']}")
        print(f"   Predicted: {option['predicted']}")
        print(f"   Rationale: {option['rationale']}")
    
    # Priority 3: Forward options
    print(f"\n⚽ PRIORITY 3 - FORWARD:")
    print("-" * 25)
    print("Current forwards: João Pedro (keep), Richarlison (2 pts), Wood (2 pts)")
    
    forward_options = [
        {
            'out': 'Richarlison (Spurs)',
            'in': 'Gabriel Jesus (Arsenal)',
            'fixture': 'vs Forest (H)',
            'cost': '+0-1m',
            'predicted': '7-9 points',
            'rationale': 'Arsenal striker, excellent fixture'
        },
        {
            'out': 'Wood (Forest)', 
            'in': 'Erling Haaland (Man City)',
            'fixture': 'vs Man Utd (H)',
            'cost': '+4-5m',
            'predicted': '9-12 points',
            'rationale': 'Premium captain option, derby'
        },
        {
            'out': 'Richarlison (Spurs)',
            'in': 'Darwin Núñez (Liverpool)',
            'fixture': 'vs Burnley (A)', 
            'cost': '+2-3m',
            'predicted': '6-8 points',
            'rationale': 'Good fixture, lower ownership'
        }
    ]
    
    for i, option in enumerate(forward_options, 1):
        print(f"\n{i}. {option['out']} → {option['in']}")
        print(f"   Fixture: {option['fixture']}")
        print(f"   Cost: {option['cost']}")
        print(f"   Predicted: {option['predicted']}")
        print(f"   Rationale: {option['rationale']}")
    
    # Recommended transfer combination
    print(f"\n🏆 RECOMMENDED TRANSFER PACKAGE:")
    print("=" * 40)
    
    scenarios = [
        {
            'name': 'Conservative (2 transfers)',
            'moves': [
                'Pedro Porro → Calafiori (Arsenal)',
                'Semenyo → Ødegaard (Arsenal)'
            ],
            'cost': '+1-2m total',
            'arsenal_players': 2,
            'predicted_gain': '+8-12 points',
            'risk': 'Low'
        },
        {
            'name': 'Aggressive (3 transfers)',
            'moves': [
                'Pedro Porro → Calafiori (Arsenal)', 
                'Semenyo → Saka (Arsenal)',
                'Richarlison → Jesus (Arsenal)'
            ],
            'cost': '+4-5m total',
            'arsenal_players': 3,
            'predicted_gain': '+15-20 points',
            'risk': 'High (Arsenal triple-up)'
        },
        {
            'name': 'Balanced (2 transfers)',
            'moves': [
                'Pedro Porro → Calafiori (Arsenal)',
                'Semenyo → Bruno Fernandes (Man Utd)'
            ],
            'cost': '+2-3m total', 
            'arsenal_players': 1,
            'predicted_gain': '+10-15 points',
            'risk': 'Medium'
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. {scenario['name']}:")
        for move in scenario['moves']:
            print(f"   • {move}")
        print(f"   Total Cost: {scenario['cost']}")
        print(f"   Arsenal Players: {scenario['arsenal_players']}/3")
        print(f"   Predicted Gain: {scenario['predicted_gain']}")
        print(f"   Risk Level: {scenario['risk']}")
    
    # Budget analysis
    print(f"\n💰 BUDGET FUNDING:")
    print("=" * 20)
    print("To fund premium transfers:")
    print("• Option 1: Downgrade Wood → 4.5m forward")
    print("• Option 2: Bank transfer value from previous weeks") 
    print("• Option 3: Slight formation change (bench cheaper player)")
    
    # Final recommendation
    print(f"\n🎯 FINAL RECOMMENDATION:")
    print("=" * 30)
    print("✅ PRIORITY MOVES:")
    print("1. Pedro Porro → Calafiori (Arsenal) - ESSENTIAL")
    print("2. Semenyo → Bruno Fernandes (Man Utd) - BALANCED RISK")
    print()
    print("👑 CAPTAINCY:")
    print("• PRIMARY: João Pedro (continue differential success)")
    print("• ALTERNATIVE: Bruno Fernandes (if transferred in)")
    print("• SAFE OPTION: Salah (Liverpool vs Burnley)")
    
    print(f"\n🏆 PREDICTED OUTCOME:")
    print("-" * 25)
    print("With these moves: 60-70 points expected")
    print("• Arsenal defense vs Forest: 6-8 points")
    print("• Bruno vs City: 8-12 points potential")
    print("• João Pedro (C): 8-16 points (doubled)")
    print("• Other players: 30-35 points")
    print("TOTAL TARGET: 65+ points ✅")
    
    print(f"\n⚠️ KEY CONSTRAINTS RESPECTED:")
    print("-" * 35)
    print("✅ Chelsea limit: 3/3 (João Pedro, Cucurella, Caicedo)")
    print("✅ Arsenal limit: 1/3 (Calafiori only)")
    print("✅ Man United limit: 1/3 (Bruno only)")
    print("✅ Budget manageable with minor downgrades")

if __name__ == "__main__":
    main()