#!/usr/bin/env python3
"""
GW4 HIGH-SCORING STRATEGY
Target: 65+ points (significantly above average)
Focus: Differential captains, premium fixtures, form players
"""

import pandas as pd
import numpy as np

def analyze_gw4_fixtures():
    """Analyze GW4 fixtures for scoring opportunities"""
    fixtures = pd.read_csv('data/2025-2026/By Gameweek/GW4/fixtures.csv')
    teams = pd.read_csv('data/2025-2026/By Gameweek/GW4/teams.csv')
    
    print("🎯 GW4 FIXTURE ANALYSIS FOR HIGH SCORING")
    print("=" * 60)
    
    # Create team mapping
    team_mapping = {row['code']: row['name'] for _, row in teams.iterrows()}
    
    fixture_analysis = []
    
    for _, fixture in fixtures.iterrows():
        home_code = int(fixture['home_team'])
        away_code = int(fixture['away_team'])
        
        home_team = team_mapping.get(home_code, f'Team_{home_code}')
        away_team = team_mapping.get(away_code, f'Team_{away_code}')
        
        home_elo = fixture['home_team_elo']
        away_elo = fixture['away_team_elo']
        
        # Calculate scoring potential
        elo_diff = home_elo - away_elo
        total_quality = home_elo + away_elo
        
        # High-scoring potential indicators
        attacking_potential = "High" if total_quality > 3600 else "Medium" if total_quality > 3200 else "Low"
        
        fixture_analysis.append({
            'home': home_team,
            'away': away_team,
            'home_elo': home_elo,
            'away_elo': away_elo,
            'elo_diff': elo_diff,
            'total_quality': total_quality,
            'attacking_potential': attacking_potential
        })
    
    # Sort by total quality (highest scoring potential)
    fixture_analysis.sort(key=lambda x: x['total_quality'], reverse=True)
    
    print("🔥 HIGHEST SCORING POTENTIAL FIXTURES:")
    print("-" * 50)
    print("Home Team        vs Away Team         Total Quality  Potential")
    print("-" * 65)
    
    for i, fixture in enumerate(fixture_analysis[:5], 1):
        print(f"{fixture['home']:<15} vs {fixture['away']:<15} "
              f"{fixture['total_quality']:>6.0f}     {fixture['attacking_potential']}")
    
    return fixture_analysis

def analyze_differential_captains():
    """Find high-upside differential captain options"""
    print(f"\n👑 DIFFERENTIAL CAPTAIN TARGETS FOR RANK GAINS:")
    print("=" * 60)
    
    gw3_stats = pd.read_csv('data/2025-2026/By Gameweek/GW3/playerstats.csv')
    
    # Focus on players with high recent scores but likely lower ownership
    differential_targets = []
    
    # Key metrics for captain potential
    high_scorers_gw3 = gw3_stats[gw3_stats['event_points'] >= 8].copy()
    
    print("🎯 HIGH-SCORING GW3 PLAYERS (Potential Captain Targets):")
    print("-" * 55)
    print("Player           Team         Points  Form   Ownership  Potential")
    print("-" * 70)
    
    for _, player in high_scorers_gw3.head(15).iterrows():
        ownership = player['selected_by_percent']
        points = player['event_points']
        form = player['form']
        
        # Differential rating
        if ownership < 15:
            diff_rating = "🔥 High Diff"
        elif ownership < 30:
            diff_rating = "⚡ Medium Diff"
        else:
            diff_rating = "📊 Template"
        
        print(f"{player['web_name']:<15} {player['web_name']:<12} {points:>6} {form:>6.1f} "
              f"{ownership:>9.1f}% {diff_rating}")
        
        differential_targets.append({
            'name': player['web_name'],
            'points_gw3': points,
            'form': form,
            'ownership': ownership,
            'diff_rating': diff_rating
        })
    
    return differential_targets

def identify_premium_fixture_plays():
    """Identify players with the best fixtures for GW4"""
    print(f"\n🎲 PREMIUM FIXTURE PLAYS:")
    print("=" * 40)
    
    # Based on fixture analysis - focus on teams in high-scoring games
    premium_fixtures = {
        'Man City vs Man Utd': {
            'total_quality': 3740,
            'narrative': 'Derby with goals - both teams to score',
            'targets': ['Haaland (C)', 'KDB', 'Foden', 'Bruno Fernandes', 'Rashford']
        },
        'Arsenal vs Nott Forest': {
            'total_quality': 3790,
            'narrative': 'Arsenal at home vs Forest - expect goals',
            'targets': ['Saka', 'Jesus', 'Ødegaard', 'Wood (differential)']
        },
        'Liverpool vs Burnley': {
            'total_quality': 3730,  
            'narrative': 'Liverpool away - high ceiling',
            'targets': ['Salah', 'Nunez', 'Diaz', 'Gakpo']
        },
        'Chelsea vs Brentford': {
            'total_quality': 3712,
            'narrative': 'Chelsea bounce-back game',
            'targets': ['João Pedro', 'Sterling', 'Enzo', 'Toney (differential)']
        }
    }
    
    for fixture, data in premium_fixtures.items():
        print(f"\n⚽ {fixture}")
        print(f"   Quality Score: {data['total_quality']}")
        print(f"   Narrative: {data['narrative']}")
        print(f"   Key Targets: {', '.join(data['targets'])}")
    
    return premium_fixtures

def create_high_scoring_team_template():
    """Create team template for 65+ points"""
    print(f"\n🏆 HIGH-SCORING TEAM TEMPLATE (Target: 65+ pts):")
    print("=" * 60)
    
    template = {
        'formation': '3-5-2',
        'strategy': 'Premium attacking assets + differential captain',
        'target_points': '65-75',
        
        'goalkeepers': {
            'starter': 'Pickford/Raya (good fixture)',
            'bench': '4.0m fodder'
        },
        
        'defenders': {
            'premium': 'Arsenal defender (vs Forest)',
            'value': 'Cucurella (keep - proved value)',
            'differential': 'Brighton/Newcastle defender (good fixtures)',
            'bench': 'Cheap options'
        },
        
        'midfielders': {
            'premium_1': 'Salah (Liverpool vs Burnley)',
            'premium_2': 'KDB/Foden (City vs United)',
            'form_pick': 'Saka (Arsenal vs Forest)', 
            'differential': 'Bruno Fernandes (United vs City)',
            'value': 'Keep Caicedo (consistent)'
        },
        
        'forwards': {
            'premium': 'Haaland (if fit)',
            'differential_captain': 'João Pedro (keep - proved worth)',
            'value': 'Jesus/Toney (good fixtures)'
        }
    }
    
    print(f"🎯 FORMATION: {template['formation']}")
    print(f"📈 STRATEGY: {template['strategy']}")
    print(f"🏆 TARGET: {template['target_points']} points")
    
    print(f"\n🥅 GOALKEEPERS:")
    print(f"  • Starter: {template['goalkeepers']['starter']}")
    print(f"  • Bench: {template['goalkeepers']['bench']}")
    
    print(f"\n🛡️ DEFENDERS:")
    for role, player in template['defenders'].items():
        print(f"  • {role.title()}: {player}")
    
    print(f"\n⚡ MIDFIELDERS:")
    for role, player in template['midfielders'].items():
        print(f"  • {role.replace('_', ' ').title()}: {player}")
    
    print(f"\n⚽ FORWARDS:")
    for role, player in template['forwards'].items():
        print(f"  • {role.replace('_', ' ').title()}: {player}")
    
    return template

def transfer_recommendations():
    """Specific transfer recommendations for rank climbing"""
    print(f"\n🔄 TRANSFER RECOMMENDATIONS FOR RANK GAINS:")
    print("=" * 55)
    
    current_team = {
        'underperformers': ['Pedro Porro (0 pts)', 'Semenyo (2 pts)', 'Richarlison (2 pts)', 'Wood (2 pts)'],
        'keepers': ['João Pedro (18 pts - KEEP!)', 'Cucurella (6 pts)', 'Salah (3 pts)'],
        'budget_available': 'Depends on moves made'
    }
    
    print("❌ PRIORITY TRANSFERS OUT:")
    print("-" * 30)
    for i, player in enumerate(current_team['underperformers'], 1):
        print(f"{i}. {player}")
    
    print(f"\n✅ TRANSFER IN TARGETS:")
    print("-" * 25)
    
    transfer_targets = [
        {'out': 'Pedro Porro', 'in': 'Arsenal defender', 'reason': 'Better fixture vs Forest'},
        {'out': 'Semenyo', 'in': 'Saka/Foden', 'reason': 'Premium mid with high ceiling'},
        {'out': 'Richarlison', 'in': 'Jesus/Toney', 'reason': 'Better fixtures and form'},
        {'out': 'Wood', 'in': 'Haaland/premium FWD', 'reason': 'Captain potential vs United'}
    ]
    
    for i, transfer in enumerate(transfer_targets, 1):
        print(f"{i}. {transfer['out']} → {transfer['in']}")
        print(f"   Reason: {transfer['reason']}")
    
    print(f"\n💰 BUDGET CONSIDERATIONS:")
    print("-" * 25)
    print("• Pedro Porro out → Arsenal def in: Neutral/slight cost")
    print("• Semenyo out → Premium mid in: +2-3m needed")
    print("• Need to downgrade elsewhere to fund premiums")
    print("• Focus on 2-3 key transfers rather than many changes")
    
    return transfer_targets

def captaincy_strategy_for_rank_gains():
    """Advanced captaincy strategy for significant rank improvement"""
    print(f"\n👑 CAPTAINCY STRATEGY FOR RANK CLIMBING:")
    print("=" * 50)
    
    captaincy_options = [
        {
            'player': 'João Pedro',
            'fixture': 'Chelsea vs Brentford (H)',
            'ownership': '~3-5%',
            'risk': 'High',
            'reward': 'Massive if delivers',
            'rationale': 'Proved worth in GW3, low ownership, good fixture'
        },
        {
            'player': 'Haaland',
            'fixture': 'Man City vs Man Utd (H)',
            'ownership': '~60%+',
            'risk': 'Low',
            'reward': 'Safe but no rank gain',
            'rationale': 'Template pick - won\'t help you climb'
        },
        {
            'player': 'Bruno Fernandes',
            'fixture': 'Man Utd vs Man City (A)',
            'ownership': '~15-25%',
            'risk': 'Medium-High',
            'reward': 'High rank gain potential',
            'rationale': 'United talisman, penalties, medium differential'
        },
        {
            'player': 'Jesus',
            'fixture': 'Arsenal vs Nott Forest (H)',
            'ownership': '~10-20%',
            'risk': 'Medium',
            'reward': 'High if starts and scores',
            'rationale': 'Great fixture, lower ownership than Saka'
        }
    ]
    
    print("🎯 CAPTAIN OPTIONS ANALYSIS:")
    print("-" * 40)
    
    for i, option in enumerate(captaincy_options, 1):
        print(f"\n{i}. {option['player']}")
        print(f"   Fixture: {option['fixture']}")
        print(f"   Ownership: {option['ownership']}")
        print(f"   Risk/Reward: {option['risk']} risk, {option['reward']}")
        print(f"   Rationale: {option['rationale']}")
    
    print(f"\n🏆 RECOMMENDATION FOR RANK GAINS:")
    print("-" * 35)
    print("🎯 PRIMARY: João Pedro (if you keep him)")
    print("   → Continuation of GW3 success")
    print("   → Very low ownership = massive rank gains if delivers")
    print("   → Home fixture vs Brentford")
    
    print(f"\n⚡ ALTERNATIVE: Bruno Fernandes")
    print("   → Medium differential")
    print("   → Penalty taker in high-scoring derby")
    print("   → Good balance of safety and differential")
    
    return captaincy_options

def main():
    print("🚀 GW4 STRATEGY: SIGNIFICANTLY ABOVE AVERAGE")
    print("=" * 80)
    print("🎯 Target: 65+ points (15+ above average)")
    print("📈 Goal: Climb rankings with differential strategy")
    print()
    
    # Run all analyses
    fixtures = analyze_gw4_fixtures()
    differentials = analyze_differential_captains()
    premium_plays = identify_premium_fixture_plays()
    team_template = create_high_scoring_team_template()
    transfers = transfer_recommendations()
    captaincy = captaincy_strategy_for_rank_gains()
    
    # Final summary
    print(f"\n🏆 FINAL STRATEGY SUMMARY:")
    print("=" * 40)
    print("🎯 CORE PRINCIPLE: Balanced risk-taking")
    print("👑 CAPTAIN: João Pedro (differential) or Bruno (medium diff)")
    print("🔄 TRANSFERS: 2-3 key moves focusing on fixtures")
    print("💰 BUDGET: Prioritize premium midfielders")
    print("📊 FORMATION: Attack-heavy (3-5-2 or 3-4-3)")
    
    print(f"\n✅ SUCCESS METRICS:")
    print("• 65+ points = Excellent week")
    print("• 60-64 points = Good week above average")
    print("• Successful differential captain = Major rank gain")
    print("• Key is execution of 2-3 perfect moves")
    
    print(f"\n⚠️ RISK MANAGEMENT:")
    print("• Don't take unnecessary hits")
    print("• Keep João Pedro if no clear upgrade")
    print("• Focus on fixtures over form")
    print("• Balance differentials with some safe picks")

if __name__ == "__main__":
    main()