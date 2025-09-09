#!/usr/bin/env python3
"""
GW4 FINAL REVISED STRATEGY
Updated for constraints: 3 Chelsea players + Haaland injured + Toney left + Jesus injured
"""

def main():
    print("🔄 GW4 FINAL REVISED STRATEGY")
    print("=" * 50)
    print("⚠️ CONSTRAINTS:")
    print("   • 3 Chelsea players: João Pedro, Cucurella, Caicedo")
    print("   • Haaland injured")
    print("   • Toney left Premier League") 
    print("   • Gabriel Jesus injured")
    print("🎯 Target: 65+ points with available players")
    print()
    
    print("✅ CONFIRMED TRANSFER PRIORITIES:")
    print("=" * 40)
    
    # Priority 1: Defender (unchanged)
    print("🏆 PRIORITY 1 - DEFENDER:")
    print("-" * 25)
    print("OUT: Pedro Porro (Spurs) - 0 points, poor fixture")
    print("IN:  Calafiori (Arsenal) - vs Forest (H), excellent fixture")
    print("Cost: Neutral")
    print("Predicted: 6-8 points")
    print("Status: ✅ ESSENTIAL - No injury concerns")
    
    # Priority 2: Midfielder (unchanged)
    print(f"\n🎯 PRIORITY 2 - MIDFIELDER:")
    print("-" * 30)
    print("OUT: Semenyo (Bournemouth) - 2 points")
    print("IN:  Bruno Fernandes (Man United) - vs City (A)")
    print("Cost: +2m")
    print("Predicted: 6-10 points (penalties, derby)")
    print("Status: ✅ CONFIRMED - Medium differential")
    
    # Priority 3: Forward (REVISED due to Jesus injury)
    print(f"\n⚽ PRIORITY 3 - FORWARD (REVISED):")
    print("-" * 35)
    print("❌ REMOVED OPTIONS:")
    print("   • Gabriel Jesus (Arsenal) - INJURED")
    print("   • Haaland (Man City) - INJURED")
    print("   • Toney - LEFT PREMIER LEAGUE")
    
    print(f"\n✅ AVAILABLE FORWARD OPTIONS:")
    
    forward_options = [
        {
            'player': 'Darwin Núñez (Liverpool)',
            'out': 'Richarlison',
            'fixture': 'vs Burnley (A)',
            'cost': '+2-3m',
            'predicted': '7-9 points',
            'ownership': '~15%',
            'rationale': 'Good fixture, rotation with Gakpo but high ceiling'
        },
        {
            'player': 'Jean-Philippe Mateta (Crystal Palace)',
            'out': 'Wood',
            'fixture': 'vs Sunderland (H)',
            'cost': '+1-2m', 
            'predicted': '6-8 points',
            'ownership': '~8%',
            'rationale': 'Excellent fixture, Palace main striker, differential'
        },
        {
            'player': 'Alexander Isak (Newcastle)',
            'out': 'Richarlison',
            'fixture': 'vs Wolves (H)',
            'cost': '+3-4m',
            'predicted': '7-10 points',
            'ownership': '~20%',
            'rationale': 'Good fixture, Newcastle talisman, consistent'
        },
        {
            'player': 'Kai Havertz (Arsenal)',
            'out': 'Richarlison',
            'fixture': 'vs Forest (H)',
            'cost': '+2-3m',
            'predicted': '6-9 points',
            'ownership': '~12%',
            'rationale': 'Arsenal attack vs Forest, plays as false 9'
        }
    ]
    
    for i, option in enumerate(forward_options, 1):
        print(f"\n{i}. {option['player']}")
        print(f"   OUT: {option['out']}")
        print(f"   Fixture: {option['fixture']}")
        print(f"   Cost: {option['cost']}")
        print(f"   Predicted: {option['predicted']}")
        print(f"   Ownership: {option['ownership']}")
        print(f"   Rationale: {option['rationale']}")
    
    # Recommended strategy scenarios
    print(f"\n🏆 RECOMMENDED STRATEGY SCENARIOS:")
    print("=" * 45)
    
    scenarios = [
        {
            'name': 'Conservative (2 transfers)',
            'moves': [
                'Pedro Porro → Calafiori (Arsenal)',
                'Semenyo → Bruno Fernandes (Man Utd)'
            ],
            'keep_forwards': 'João Pedro + Richarlison + Wood',
            'cost': '+2m total',
            'predicted': '58-65 points',
            'risk': 'Low'
        },
        {
            'name': 'Balanced (3 transfers)', 
            'moves': [
                'Pedro Porro → Calafiori (Arsenal)',
                'Semenyo → Bruno Fernandes (Man Utd)',
                'Richarlison → Darwin Núñez (Liverpool)'
            ],
            'forwards': 'João Pedro + Darwin + Wood',
            'cost': '+4-5m total',
            'predicted': '62-70 points',
            'risk': 'Medium'
        },
        {
            'name': 'Differential (3 transfers)',
            'moves': [
                'Pedro Porro → Calafiori (Arsenal)',
                'Semenyo → Bruno Fernandes (Man Utd)', 
                'Wood → Mateta (Crystal Palace)'
            ],
            'forwards': 'João Pedro + Richarlison + Mateta',
            'cost': '+3-4m total',
            'predicted': '60-68 points',
            'risk': 'Medium-High'
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. {scenario['name']}:")
        print(f"   Transfers:")
        for move in scenario['moves']:
            print(f"     • {move}")
        if 'keep_forwards' in scenario:
            print(f"   Forwards: {scenario['keep_forwards']}")
        else:
            print(f"   Forwards: {scenario['forwards']}")
        print(f"   Total Cost: {scenario['cost']}")
        print(f"   Predicted: {scenario['predicted']}")
        print(f"   Risk: {scenario['risk']}")
    
    print(f"\n💰 BUDGET SOLUTIONS:")
    print("=" * 20)
    print("To fund transfers (+4-5m needed):")
    print("• Downgrade Wood → 4.5m forward (saves ~2m)")
    print("• Use banked transfer value")
    print("• Slight formation adjustment if needed")
    
    print(f"\n👑 CAPTAINCY (UNCHANGED):")
    print("=" * 30)
    
    captaincy_options = [
        {
            'player': 'João Pedro (Chelsea)',
            'predicted': '8-16 points (doubled)',
            'ownership': '~3%',
            'rationale': 'Continue GW3 differential success',
            'risk': 'High reward, proven track record'
        },
        {
            'player': 'Bruno Fernandes (if transferred)',
            'predicted': '12-20 points (doubled)',
            'ownership': '~20%',
            'rationale': 'Derby fixture, penalties, medium differential',
            'risk': 'Balanced risk/reward'
        },
        {
            'player': 'Mohamed Salah (current)',
            'predicted': '10-18 points (doubled)',
            'ownership': '~60%',
            'rationale': 'Safe option vs Burnley away',
            'risk': 'Low risk, template pick'
        }
    ]
    
    print("Captain Options Analysis:")
    for i, option in enumerate(captaincy_options, 1):
        print(f"\n{i}. {option['player']}")
        print(f"   Predicted: {option['predicted']}")
        print(f"   Ownership: {option['ownership']}")
        print(f"   Rationale: {option['rationale']}")
        print(f"   Risk Profile: {option['risk']}")
    
    print(f"\n🎯 FINAL RECOMMENDATION:")
    print("=" * 30)
    print("✅ EXECUTE:")
    print("1. Pedro Porro → Calafiori (Arsenal) - ESSENTIAL")
    print("2. Semenyo → Bruno Fernandes (Man Utd) - STRONG")
    print("3. Keep current forwards OR upgrade Richarlison → Darwin")
    
    print(f"\n👑 CAPTAIN: João Pedro")
    print("   → Continue differential edge from GW3")
    print("   → Low ownership provides massive rank gain potential")
    print("   → Home fixture vs Brentford")
    
    print(f"\n📊 PREDICTED OUTCOME:")
    print("-" * 25)
    print("Conservative approach: 58-65 points")
    print("Balanced approach: 62-70 points")
    print("Target: 15+ points above average ✅")
    
    print(f"\n🔄 INJURY UPDATE IMPACT:")
    print("-" * 30)
    print("✅ Strategy remains strong despite Jesus injury")
    print("✅ Multiple viable forward alternatives available") 
    print("✅ Core differential approach (João Pedro C) intact")
    print("✅ Arsenal coverage via defense (Calafiori) not attack")
    print("⚖️ Balanced risk profile maintained")
    
    print(f"\n🏆 SUCCESS METRICS:")
    print("-" * 20)
    print("• 65+ points = Excellent result")
    print("• João Pedro captain success = Major rank gain")
    print("• Arsenal/United players deliver = Strategy validation")
    print("• Maintain differential edge for future GWs")

if __name__ == "__main__":
    main()