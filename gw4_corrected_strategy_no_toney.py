#!/usr/bin/env python3
"""
GW4 CORRECTED STRATEGY: NO TONEY, NO HAALAND
Updated with accurate player database - Toney already left PL
"""

def main():
    print("🔄 GW4 CORRECTED STRATEGY: DATABASE UPDATED")
    print("=" * 80)
    print("✅ Toney correctly removed from database (moved to Saudi Arabia)")
    print("❌ Haaland injured in international break")
    print("🎯 Revised Target: Still aiming for 65+ points with available players")
    print()
    
    print("⚽ UPDATED FORWARD OPTIONS (Toney Removed):")
    print("=" * 50)
    
    forward_options = [
        {
            'player': 'Jesus (Arsenal)',
            'fixture': 'vs Nott\'m Forest (H)',
            'predicted': '8.5+ pts',
            'pros': ['Excellent fixture vs Forest', 'Arsenal attack', 'Proven PL striker'],
            'cons': ['Rotation risk with Havertz', 'Injury history'],
            'recommendation': '🔥 TOP CHOICE - Premium fixture',
            'ownership': '~15%'
        },
        {
            'player': 'Havertz (Arsenal)',
            'fixture': 'vs Nott\'m Forest (H)', 
            'predicted': '8.0+ pts',
            'pros': ['Same excellent fixture', 'More nailed than Jesus', 'Versatile'],
            'cons': ['Higher price', 'Less explosive than Jesus'],
            'recommendation': '✅ SAFER ARSENAL OPTION',
            'ownership': '~25%'
        },
        {
            'player': 'Darwin/Isak (Liverpool)',
            'fixture': 'vs Burnley (A)',
            'predicted': '8.0+ pts',
            'pros': ['Liverpool attack', 'Good fixture vs Burnley', 'High ceiling'],
            'cons': ['Darwin inconsistent', 'Isak price', 'Away fixture'],
            'recommendation': '⚡ PREMIUM OPTIONS',
            'ownership': '~20-30%'
        },
        {
            'player': 'Mateta (Crystal Palace)',
            'fixture': 'vs Sunderland (H)',
            'predicted': '7.5+ pts',
            'pros': ['Excellent fixture vs Sunderland', 'Palace talisman', 'Good value'],
            'cons': ['Lower ceiling than premiums', 'Palace inconsistency'],
            'recommendation': '💎 VALUE DIFFERENTIAL',
            'ownership': '~8%'
        },
        {
            'player': 'Woltemade (Newcastle)',
            'fixture': 'vs Wolves (H)',
            'predicted': '7.0+ pts',
            'pros': ['Good fixture', 'Newcastle form', 'Lower ownership'],
            'cons': ['Newer PL player', 'Less proven'],
            'recommendation': '🎲 PUNT OPTION',
            'ownership': '~3%'
        },
        {
            'player': 'Keep João Pedro + Wood',
            'fixture': 'Pedro: vs Brentford (A), Wood: vs Arsenal (A)',
            'predicted': '9+4 = 13 pts combined',
            'pros': ['Keep proven João Pedro', 'Save money for midfield', 'Differential Wood'],
            'cons': ['Wood poor fixture', 'Less firepower up front'],
            'recommendation': '💰 BUDGET APPROACH',
            'ownership': 'Pedro ~3%, Wood ~1%'
        }
    ]
    
    for option in forward_options:
        print(f"\n{option['recommendation']} - {option['player']}")
        print(f"   Fixture: {option['fixture']}")
        print(f"   Predicted: {option['predicted']}")
        print(f"   Ownership: {option['ownership']}")
        print(f"   Pros: {', '.join(option['pros'])}")
        print(f"   Cons: {', '.join(option['cons'])}")
    
    print(f"\n🎯 REVISED TRANSFER RECOMMENDATIONS:")
    print("=" * 45)
    
    transfer_scenarios = [
        {
            'scenario': 'PREMIUM ATTACK (Jesus)',
            'transfers': [
                'Pedro Porro → Calafiori (±0m)',
                'Richarlison → Jesus (+1-2m)', 
                'Semenyo → Budget mid (-2m to fund)'
            ],
            'captain': 'João Pedro (differential) or Jesus (fixture)',
            'pros': 'Arsenal double-up vs Forest, premium attack',
            'cons': 'Need to downgrade elsewhere'
        },
        {
            'scenario': 'MIDFIELD FOCUS (Keep forwards)',
            'transfers': [
                'Pedro Porro → Calafiori (±0m)',
                'Semenyo → Enzo (+2.5m)',
                'Keep João Pedro + Wood'
            ],
            'captain': 'João Pedro (differential continuation)',
            'pros': 'Strong midfield, save money, proven captain',
            'cons': 'Weaker forward line, Wood poor fixture'
        },
        {
            'scenario': 'DIFFERENTIAL PUNT (Palace/Newcastle)',
            'transfers': [
                'Pedro Porro → Calafiori (±0m)',
                'Richarlison → Mateta/Woltemade (+0-1m)',
                'Semenyo → Premium mid (+2m)'
            ],
            'captain': 'João Pedro or new forward (ultra differential)',
            'pros': 'Very low ownership forwards, save money',
            'cons': 'Higher risk, less proven options'
        }
    ]
    
    for i, scenario in enumerate(transfer_scenarios, 1):
        print(f"\n{i}. {scenario['scenario']}")
        print(f"   Transfers:")
        for transfer in scenario['transfers']:
            print(f"     • {transfer}")
        print(f"   Captain: {scenario['captain']}")
        print(f"   Pros: {scenario['pros']}")
        print(f"   Cons: {scenario['cons']}")
    
    print(f"\n👑 CAPTAINCY DECISION (No Haaland, No Toney):")
    print("=" * 50)
    
    captaincy_analysis = [
        {
            'player': 'João Pedro',
            'rationale': 'UNCHANGED - Your proven differential weapon',
            'predicted': '8.9 pts (17.8 as captain)',
            'ownership': '~3%',
            'fixture': 'Chelsea vs Brentford (A)',
            'verdict': '🔥 STICK WITH SUCCESS - Low risk for you'
        },
        {
            'player': 'Jesus (if transferred in)',
            'rationale': 'Arsenal vs Forest - premium fixture',
            'predicted': '8.5 pts (17.0 as captain)',
            'ownership': '~15%',
            'fixture': 'Arsenal vs Forest (H)',
            'verdict': '⚡ NEW DIFFERENTIAL - Medium risk'
        },
        {
            'player': 'Calafiori (ultra differential)',
            'rationale': 'EXTREME punt - defender captain',
            'predicted': '11.9 pts (23.8 as captain)',
            'ownership': '~1%',
            'fixture': 'Arsenal vs Forest (H)',
            'verdict': '🎲 ULTIMATE RISK - Massive upside'
        },
        {
            'player': 'Salah (template)',
            'rationale': 'Liverpool vs Burnley - safe option',
            'predicted': '9.0 pts (18.0 as captain)',
            'ownership': '~70%',
            'fixture': 'Liverpool vs Burnley (A)',
            'verdict': '📊 TEMPLATE - No rank gain'
        }
    ]
    
    for option in captaincy_analysis:
        print(f"\n• {option['player']}")
        print(f"  Predicted: {option['predicted']}")
        print(f"  Ownership: {option['ownership']}")
        print(f"  Fixture: {option['fixture']}")
        print(f"  Verdict: {option['verdict']}")
    
    print(f"\n🏆 FINAL RECOMMENDATION:")
    print("=" * 25)
    
    final_strategy = {
        'primary_plan': 'MIDFIELD FOCUS APPROACH',
        'transfers': [
            'Pedro Porro → Calafiori (Arsenal defender)',
            'Semenyo → Enzo Fernández (Chelsea mid)',
            'Keep João Pedro + Wood (save money)'
        ],
        'captain': 'João Pedro (differential continuation)',
        'formation': '3-5-2 (midfield heavy)',
        'rationale': 'Proven differential captain + Arsenal fixture + strong midfield'
    }
    
    print(f"📊 RECOMMENDED APPROACH: {final_strategy['primary_plan']}")
    print(f"\n🔄 Key Transfers:")
    for transfer in final_strategy['transfers']:
        print(f"• {transfer}")
    
    print(f"\n👑 Captain: {final_strategy['captain']}")
    print(f"📐 Formation: {final_strategy['formation']}")
    
    print(f"\n✅ WHY THIS WORKS:")
    rationale_points = [
        "João Pedro proved himself in GW3 (18 pts as captain)",
        "Calafiori gets Arsenal vs Forest premium fixture",
        "Enzo adds Chelsea quality vs Brentford", 
        "Maintains your differential edge (low ownership plays)",
        "Budget efficiency allows stronger overall team",
        "Less reliance on expensive forwards after Haaland/Toney unavailable"
    ]
    
    for point in rationale_points:
        print(f"• {point}")
    
    print(f"\n🎯 UPDATED SUCCESS TARGETS:")
    print("=" * 30)
    success_metrics = [
        "60-65+ points achievable without Haaland/Toney",
        "João Pedro differential captain remains key weapon",
        "Arsenal fixtures provide premium scoring opportunity",
        "Midfield-heavy approach for consistency",
        "Maintain ranking edge through differential strategy"
    ]
    
    for metric in success_metrics:
        print(f"• {metric}")
    
    print(f"\n🚀 FINAL EXECUTION PLAN:")
    print("=" * 30)
    
    execution_steps = [
        "1. Confirm João Pedro fitness (monitor team news)",
        "2. Transfer Pedro Porro → Calafiori (priority)",
        "3. Transfer Semenyo → Enzo Fernández",
        "4. Keep current forwards (João Pedro + Wood)",
        "5. Captain João Pedro (continuation strategy)",
        "6. Formation: 3-5-2 or 4-5-1 (midfield focus)"
    ]
    
    for step in execution_steps:
        print(step)
    
    print(f"\n🏆 CONFIDENCE LEVEL: HIGH")
    print("=" * 25)
    print("✅ Strategy adapts well to Haaland injury + Toney departure")
    print("🎯 João Pedro differential remains your ace card") 
    print("🔥 Arsenal fixtures still provide premium opportunity")
    print("📈 60-65 points very achievable with this approach")
    print("🚀 Your differential philosophy gives you the edge!")

if __name__ == "__main__":
    main()