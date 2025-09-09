#!/usr/bin/env python3
"""
GW4 REVISED STRATEGY: HAALAND INJURED
Adjusted strategy for 65+ points without Haaland
"""

def main():
    print("⚠️ GW4 STRATEGY REVISION: HAALAND INJURED")
    print("=" * 80)
    print("🚨 BREAKING: Haaland injured in international break")
    print("🎯 Revised Target: Still aiming for 65+ points")
    print()
    
    print("🔄 IMMEDIATE STRATEGY ADJUSTMENTS:")
    print("=" * 40)
    
    print("❌ ORIGINAL PLAN CHANGES:")
    print("• Haaland transfer is OFF the table")
    print("• Need new premium forward target")
    print("• Budget allocation shifts")
    print("• Captaincy options change")
    
    print(f"\n🎯 REVISED TRANSFER RECOMMENDATIONS:")
    print("=" * 45)
    
    revised_transfers = [
        {
            'priority': 1,
            'out': 'Pedro Porro',
            'in': 'Calafiori (Arsenal)',
            'cost': '±0m',
            'rationale': '11.9 predicted pts vs Forest - UNCHANGED, still excellent',
            'status': 'CONFIRMED - No change needed'
        },
        {
            'priority': 2,
            'out': 'Semenyo', 
            'in': 'Enzo Fernandez (Chelsea)',
            'cost': '+2.5m',
            'rationale': '9.3 predicted pts - Premium mid becomes MORE important',
            'status': 'UPGRADED PRIORITY - Fund with forward downgrade'
        },
        {
            'priority': 3,
            'out': 'Richarlison',
            'in': 'Jesus (Arsenal) OR Toney (Brentford)',
            'cost': '+1-2m',
            'rationale': 'Arsenal vs Forest OR Brentford vs Chelsea fixtures',
            'status': 'NEW TARGET - Replace Haaland plan'
        }
    ]
    
    for transfer in revised_transfers:
        print(f"\n{transfer['priority']}. {transfer['out']} → {transfer['in']}")
        print(f"   Cost: {transfer['cost']}")  
        print(f"   Rationale: {transfer['rationale']}")
        print(f"   Status: {transfer['status']}")
    
    print(f"\n⚽ NEW FORWARD TARGETS (Haaland Replacements):")
    print("=" * 50)
    
    forward_options = [
        {
            'player': 'Jesus (Arsenal)',
            'fixture': 'vs Nott\'m Forest (H)',
            'predicted': '8.5+ pts',
            'pros': ['Excellent fixture', 'Arsenal attack', 'Good value'],
            'cons': ['Rotation risk', 'Injury concerns'],
            'recommendation': '🔥 TOP CHOICE'
        },
        {
            'player': 'Toney (Brentford)', 
            'fixture': 'vs Chelsea (H)',
            'predicted': '7.5+ pts',
            'pros': ['Penalty taker', 'Big game player', 'Differential'],
            'cons': ['Tough fixture', 'Lower ceiling'],
            'recommendation': '⚡ DIFFERENTIAL'
        },
        {
            'player': 'Isak (Newcastle)',
            'fixture': 'vs Wolves (H)',
            'predicted': '8.0+ pts', 
            'pros': ['Good fixture', 'In form', 'Premium quality'],
            'cons': ['Higher price', 'Budget stretch'],
            'recommendation': '✅ SAFE CHOICE'
        },
        {
            'player': 'Keep Wood',
            'fixture': 'vs Arsenal (A)',
            'predicted': '4.0 pts',
            'pros': ['Save money for midfield upgrades', 'Differential'],
            'cons': ['Poor fixture', 'Low ceiling'],
            'recommendation': '💰 BUDGET OPTION'
        }
    ]
    
    for option in forward_options:
        print(f"\n{option['recommendation']} - {option['player']}")
        print(f"   Fixture: {option['fixture']}")
        print(f"   Predicted: {option['predicted']}")
        print(f"   Pros: {', '.join(option['pros'])}")
        print(f"   Cons: {', '.join(option['cons'])}")
    
    print(f"\n👑 REVISED CAPTAINCY STRATEGY:")
    print("=" * 35)
    
    captaincy_options = [
        {
            'player': 'João Pedro',
            'rationale': 'UNCHANGED - Still the differential king',
            'predicted': '8.9 pts (17.8 as captain)',
            'ownership': '~3%',
            'verdict': '🔥 PRIMARY CHOICE - Strategy unchanged'
        },
        {
            'player': 'Jesus (if transferred in)',
            'rationale': 'Arsenal vs Forest - excellent fixture',
            'predicted': '8.5 pts (17.0 as captain)',
            'ownership': '~15%',
            'verdict': '⚡ NEW DIFFERENTIAL OPTION'
        },
        {
            'player': 'Salah',
            'rationale': 'Liverpool vs Burnley - safe template',
            'predicted': '9.0 pts (18.0 as captain)',
            'ownership': '~70%',
            'verdict': '📊 TEMPLATE - No rank gain'
        },
        {
            'player': 'Calafiori (if brave enough)',
            'rationale': 'ULTRA differential defender captain',
            'predicted': '11.9 pts (23.8 as captain)',
            'ownership': '~1%',
            'verdict': '🎲 ULTIMATE DIFFERENTIAL PUNT'
        }
    ]
    
    for i, option in enumerate(captaincy_options, 1):
        print(f"\n{i}. {option['player']}")
        print(f"   Predicted: {option['predicted']}")
        print(f"   Ownership: {option['ownership']}")
        print(f"   Rationale: {option['rationale']}")
        print(f"   Verdict: {option['verdict']}")
    
    print(f"\n🏆 FINAL CAPTAINCY RECOMMENDATION:")
    print("-" * 35)
    print("🎯 STICK WITH JOÃO PEDRO!")
    print("   → Haaland injury doesn't change this")
    print("   → Still extremely low ownership")
    print("   → Proved himself in GW3")
    print("   → Chelsea vs Brentford still good fixture")
    print("   → Maintains your differential edge")
    
    print(f"\n💰 REVISED BUDGET ALLOCATION:")
    print("=" * 30)
    
    budget_plan = {
        'scenario': 'No Haaland = More Midfield Focus',
        'strategy': 'Invest saved Haaland money into premium midfielders',
        'transfers': [
            'Porro → Calafiori (±0m)',
            'Semenyo → Enzo (+2.5m)', 
            'Keep Wood OR upgrade to Jesus (+1m)',
            'Potentially upgrade another midfielder'
        ],
        'formation': '3-5-2 or 4-5-1 (midfield heavy)'
    }
    
    print(f"📊 Strategy: {budget_plan['strategy']}")
    print(f"Formation: {budget_plan['formation']}")
    print(f"\nTransfer Plan:")
    for i, transfer in enumerate(budget_plan['transfers'], 1):
        print(f"{i}. {transfer}")
    
    print(f"\n🎯 REVISED TEAM STRUCTURE:")
    print("=" * 30)
    
    revised_team = {
        'GKP': 'Vicario (keep)',
        'DEF': 'Calafiori (IN), Cucurella (keep), Budget def',
        'MID': 'Salah (keep), Enzo (IN), Bruno G (keep), Caicedo (keep), Bench mid',
        'FWD': 'João Pedro (C), Jesus/Wood, Bench fwd',
        'Key_Changes': 'More midfield heavy, less reliance on premium forwards'
    }
    
    for position, players in revised_team.items():
        if position != 'Key_Changes':
            print(f"{position}: {players}")
    
    print(f"\nKey Changes: {revised_team['Key_Changes']}")
    
    print(f"\n📈 REVISED SUCCESS TARGETS:")
    print("=" * 30)
    
    success_metrics = [
        "60-65+ points still achievable without Haaland",
        "Calafiori + Enzo upgrades offset Haaland loss", 
        "João Pedro (C) differential still key to strategy",
        "Arsenal players in excellent fixture vs Forest",
        "Midfield-heavy approach for consistency"
    ]
    
    for metric in success_metrics:
        print(f"• {metric}")
    
    print(f"\n✅ WHY THIS STILL WORKS:")
    print("=" * 25)
    
    why_works = [
        "🎯 João Pedro captaincy strategy UNCHANGED",
        "🔥 Arsenal vs Forest still premium fixture", 
        "⚡ More budget for midfield premiums",
        "📊 Less reliance on single premium forward",
        "🎲 Maintains differential approach"
    ]
    
    for reason in why_works:
        print(reason)
    
    print(f"\n🚀 FINAL ACTION PLAN (NO HAALAND):")
    print("=" * 40)
    
    action_plan = [
        "1. Transfer Porro → Calafiori (priority #1)",
        "2. Transfer Semenyo → Enzo Fernandez", 
        "3. Consider Richarlison → Jesus (if budget allows)",
        "4. Captain João Pedro (unchanged strategy)",
        "5. Formation: Midfield-heavy approach"
    ]
    
    for step in action_plan:
        print(step)
    
    print(f"\n🏆 BOTTOM LINE:")
    print("=" * 15)
    print("🎯 Haaland injury is disappointing BUT doesn't kill the strategy!")
    print("⚡ João Pedro differential captaincy remains the key")
    print("🔥 Arsenal players vs Forest still premium targets") 
    print("📈 60-65 points still very achievable")
    print("🚀 Your differential approach still gives you the edge!")

if __name__ == "__main__":
    main()