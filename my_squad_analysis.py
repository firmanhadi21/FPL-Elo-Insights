"""
Squad Analysis for GW13 and Beyond
Analyzes current squad, identifies issues, and suggests transfers
"""

import pandas as pd
from pathlib import Path

BASE_PATH = Path("/Users/macbook/Dropbox/GitHub/FPL/FPL-Elo-Insights/data/2025-2026/By Gameweek")

# Team code to name mapping
TEAM_CODES = {
    1: "Man Utd", 2: "Leeds", 3: "Arsenal", 4: "Newcastle", 6: "Spurs",
    7: "Aston Villa", 8: "Chelsea", 11: "Everton", 14: "Liverpool",
    17: "Nott'm Forest", 21: "West Ham", 31: "Crystal Palace", 36: "Brighton",
    39: "Wolves", 43: "Man City", 54: "Fulham", 56: "Sunderland",
    90: "Burnley", 91: "Bournemouth", 94: "Brentford"
}

# Your squad (web_name matches)
MY_SQUAD = [
    "Sels", "Roefs",  # GKs
    "Calafiori", "Virgil", "Rúben", "Pedro Porro", "Alderete",  # DEFs
    "Rice", "Semenyo", "Caicedo", "Reijnders", "Bruno G.",  # MIDs
    "João Pedro", "Mateta", "Haaland"  # FWDs
]

def calculate_fdr(opponent_elo, is_home):
    """Calculate FDR based on Elo"""
    adjusted_elo = opponent_elo - 65 if is_home else opponent_elo + 65
    if adjusted_elo >= 2000:
        return 5
    elif adjusted_elo >= 1900:
        return 4
    elif adjusted_elo >= 1800:
        return 3
    elif adjusted_elo >= 1700:
        return 2
    else:
        return 1

def load_fixtures_multi_gw(gameweeks):
    """Load fixtures for multiple gameweeks"""
    all_fixtures = []
    for gw in gameweeks:
        path = BASE_PATH / f"GW{gw}" / "fixtures.csv"
        if path.exists():
            fixtures = pd.read_csv(path)
            fixtures['gw'] = gw
            all_fixtures.append(fixtures)
    return pd.concat(all_fixtures, ignore_index=True) if all_fixtures else None

def main():
    print("=" * 100)
    print("YOUR SQUAD ANALYSIS - GW13 AND BEYOND")
    print("=" * 100)

    # Load data
    stats = pd.read_csv(BASE_PATH / "GW12" / "playerstats.csv")
    players = pd.read_csv(BASE_PATH / "GW12" / "players.csv")

    # Merge
    stats = stats.merge(
        players[['player_id', 'team_code', 'position']],
        left_on='id', right_on='player_id', how='left'
    )

    # Convert types
    for col in ['form', 'total_points', 'now_cost', 'minutes', 'expected_goal_involvements_per_90', 'selected_by_percent']:
        stats[col] = pd.to_numeric(stats[col], errors='coerce').fillna(0)

    stats['team_name'] = stats['team_code'].map(TEAM_CODES)

    # Get squad data
    squad = stats[stats['web_name'].isin(MY_SQUAD)].copy()

    print("\n" + "=" * 100)
    print("CURRENT SQUAD OVERVIEW")
    print("=" * 100)

    # Sort by position
    pos_order = {'Goalkeeper': 1, 'Defender': 2, 'Midfielder': 3, 'Forward': 4}
    squad['pos_order'] = squad['position'].map(pos_order)
    squad = squad.sort_values('pos_order')

    print(f"\n{'Player':<18} {'Pos':<10} {'Team':<15} {'Form':<6} {'Points':<7} {'Price':<6} {'xGI/90':<7}")
    print("-" * 85)

    total_value = 0
    for _, player in squad.iterrows():
        price = player['now_cost'] / 10
        total_value += price
        print(f"{player['web_name']:<18} {player['position']:<10} {player['team_name']:<15} "
              f"{player['form']:<6.1f} {int(player['total_points']):<7} {price:<6.1f} "
              f"{player['expected_goal_involvements_per_90']:<7.2f}")

    print("-" * 85)
    print(f"{'TOTAL VALUE:':<18} {'':<10} {'':<15} {'':<6} {'':<7} {total_value:<6.1f}")

    # Load fixtures for GW13-16
    fixtures = load_fixtures_multi_gw([13, 14, 15, 16])
    prem_fixtures = fixtures[fixtures['tournament'] == 'prem'].copy()

    # Build fixture schedule for each team
    team_schedule = {}
    for team_code in squad['team_code'].unique():
        team_schedule[team_code] = []
        for gw in [13, 14, 15, 16]:
            gw_fixtures = prem_fixtures[prem_fixtures['gw'] == gw]

            # Find fixture for this team
            home_match = gw_fixtures[gw_fixtures['home_team'] == team_code]
            away_match = gw_fixtures[gw_fixtures['away_team'] == team_code]

            if len(home_match) > 0:
                match = home_match.iloc[0]
                opp = int(match['away_team'])
                opp_elo = float(match['away_team_elo']) if pd.notna(match['away_team_elo']) else 1750
                fdr = calculate_fdr(opp_elo, is_home=True)
                team_schedule[team_code].append({
                    'gw': gw, 'opponent': opp, 'opp_name': TEAM_CODES.get(opp, '?'),
                    'venue': 'H', 'fdr': fdr
                })
            elif len(away_match) > 0:
                match = away_match.iloc[0]
                opp = int(match['home_team'])
                opp_elo = float(match['home_team_elo']) if pd.notna(match['home_team_elo']) else 1750
                fdr = calculate_fdr(opp_elo, is_home=False)
                team_schedule[team_code].append({
                    'gw': gw, 'opponent': opp, 'opp_name': TEAM_CODES.get(opp, '?'),
                    'venue': 'A', 'fdr': fdr
                })
            else:
                team_schedule[team_code].append({'gw': gw, 'opponent': None, 'opp_name': 'BLANK', 'venue': '-', 'fdr': 0})

    print("\n" + "=" * 100)
    print("FIXTURE DIFFICULTY RATING (GW13-16)")
    print("=" * 100)
    print("\nFDR Scale: 1=Very Easy, 2=Easy, 3=Medium, 4=Difficult, 5=Very Difficult")

    print(f"\n{'Player':<18} {'Team':<15} {'GW13':<12} {'GW14':<12} {'GW15':<12} {'GW16':<12} {'Avg FDR':<8}")
    print("-" * 95)

    squad_issues = []

    for _, player in squad.iterrows():
        team_code = player['team_code']
        schedule = team_schedule.get(team_code, [])

        gw_strs = []
        fdrs = []
        for fix in schedule:
            if fix['opponent'] is not None:
                gw_str = f"{fix['opp_name'][:3]}({fix['venue']}){fix['fdr']}"
                gw_strs.append(gw_str)
                fdrs.append(fix['fdr'])
            else:
                gw_strs.append("BLANK")
                fdrs.append(5)  # Blank = bad

        avg_fdr = sum(fdrs) / len(fdrs) if fdrs else 3

        # Pad to 4 gameweeks
        while len(gw_strs) < 4:
            gw_strs.append("-")

        print(f"{player['web_name']:<18} {player['team_name']:<15} "
              f"{gw_strs[0]:<12} {gw_strs[1]:<12} {gw_strs[2]:<12} {gw_strs[3]:<12} {avg_fdr:<8.1f}")

        # Track issues
        if avg_fdr >= 3.5:
            squad_issues.append((player['web_name'], player['position'], 'Tough fixtures', avg_fdr))
        if player['form'] <= 2 and player['minutes'] >= 270:
            squad_issues.append((player['web_name'], player['position'], 'Poor form', player['form']))

    # GW13 Starting XI Recommendation
    print("\n" + "=" * 100)
    print("GW13 STARTING XI RECOMMENDATION")
    print("=" * 100)

    # Add GW13 fixture info to squad
    squad_gw13 = squad.copy()
    squad_gw13['gw13_info'] = None
    squad_gw13['gw13_fdr'] = 5
    squad_gw13['gw13_venue'] = '-'
    squad_gw13['gw13_opp'] = ''

    for idx, player in squad_gw13.iterrows():
        team_code = player['team_code']
        schedule = team_schedule.get(team_code, [])
        for fix in schedule:
            if fix['gw'] == 13 and fix['opponent'] is not None:
                squad_gw13.at[idx, 'gw13_fdr'] = fix['fdr']
                squad_gw13.at[idx, 'gw13_venue'] = fix['venue']
                squad_gw13.at[idx, 'gw13_opp'] = fix['opp_name']
                break

    # Calculate GW13 score for each player
    squad_gw13['gw13_score'] = (
        (squad_gw13['form'] / 10) * 0.35 +
        ((6 - squad_gw13['gw13_fdr']) / 5) * 0.35 +
        (squad_gw13['gw13_venue'].apply(lambda x: 1 if x == 'H' else 0.6)) * 0.15 +
        (squad_gw13['expected_goal_involvements_per_90'].clip(0, 1.5) / 1.5) * 0.15
    )

    # Select best XI (must be valid formation)
    gks = squad_gw13[squad_gw13['position'] == 'Goalkeeper'].nlargest(1, 'gw13_score')
    defs = squad_gw13[squad_gw13['position'] == 'Defender'].nlargest(5, 'gw13_score')
    mids = squad_gw13[squad_gw13['position'] == 'Midfielder'].nlargest(5, 'gw13_score')
    fwds = squad_gw13[squad_gw13['position'] == 'Forward'].nlargest(3, 'gw13_score')

    # Choose best 11 with valid formation (at least 3 DEF, 2 MID, 1 FWD)
    starting_xi = []
    bench = []

    # GK (1)
    starting_xi.extend(gks['web_name'].tolist())
    bench_gk = squad_gw13[(squad_gw13['position'] == 'Goalkeeper') & (~squad_gw13['web_name'].isin(starting_xi))]

    # We need to pick 10 outfield players from 5 DEF + 5 MID + 3 FWD = 13 outfield
    # Let's score them all and pick top 10 with formation constraints
    outfield = squad_gw13[squad_gw13['position'] != 'Goalkeeper'].copy()
    outfield = outfield.sort_values('gw13_score', ascending=False)

    selected_defs = []
    selected_mids = []
    selected_fwds = []

    # Must have at least 3 DEF, 2 MID, 1 FWD
    for _, p in outfield.iterrows():
        if p['position'] == 'Defender' and len(selected_defs) < 3:
            selected_defs.append(p)
        elif p['position'] == 'Midfielder' and len(selected_mids) < 2:
            selected_mids.append(p)
        elif p['position'] == 'Forward' and len(selected_fwds) < 1:
            selected_fwds.append(p)

    # Now fill remaining 4 spots with highest scorers
    already_selected = [p['web_name'] for p in selected_defs + selected_mids + selected_fwds]
    remaining = outfield[~outfield['web_name'].isin(already_selected)].nlargest(4, 'gw13_score')

    for _, p in remaining.iterrows():
        if p['position'] == 'Defender' and len(selected_defs) < 5:
            selected_defs.append(p)
        elif p['position'] == 'Midfielder' and len(selected_mids) < 5:
            selected_mids.append(p)
        elif p['position'] == 'Forward' and len(selected_fwds) < 3:
            selected_fwds.append(p)

    print(f"\nFormation: {len(selected_defs)}-{len(selected_mids)}-{len(selected_fwds)}")
    print(f"\n{'Player':<18} {'Pos':<10} {'vs':<15} {'H/A':<5} {'FDR':<5} {'Form':<6} {'Score':<6}")
    print("-" * 70)

    # Print GK
    for _, p in gks.iterrows():
        print(f"{p['web_name']:<18} {'GK':<10} {p['gw13_opp']:<15} {p['gw13_venue']:<5} "
              f"{int(p['gw13_fdr']):<5} {p['form']:<6.1f} {p['gw13_score']:<6.3f}")

    # Print DEFs
    for p in sorted(selected_defs, key=lambda x: x['gw13_score'], reverse=True):
        print(f"{p['web_name']:<18} {'DEF':<10} {p['gw13_opp']:<15} {p['gw13_venue']:<5} "
              f"{int(p['gw13_fdr']):<5} {p['form']:<6.1f} {p['gw13_score']:<6.3f}")

    # Print MIDs
    for p in sorted(selected_mids, key=lambda x: x['gw13_score'], reverse=True):
        print(f"{p['web_name']:<18} {'MID':<10} {p['gw13_opp']:<15} {p['gw13_venue']:<5} "
              f"{int(p['gw13_fdr']):<5} {p['form']:<6.1f} {p['gw13_score']:<6.3f}")

    # Print FWDs
    for p in sorted(selected_fwds, key=lambda x: x['gw13_score'], reverse=True):
        print(f"{p['web_name']:<18} {'FWD':<10} {p['gw13_opp']:<15} {p['gw13_venue']:<5} "
              f"{int(p['gw13_fdr']):<5} {p['form']:<6.1f} {p['gw13_score']:<6.3f}")

    # Bench
    all_starting = [p['web_name'] for p in [gks.iloc[0]] + selected_defs + selected_mids + selected_fwds]
    bench_players = squad_gw13[~squad_gw13['web_name'].isin(all_starting)].sort_values('gw13_score', ascending=False)

    print("\n--- BENCH ---")
    for _, p in bench_players.iterrows():
        print(f"{p['web_name']:<18} {p['position']:<10} {p['gw13_opp']:<15} {p['gw13_venue']:<5} "
              f"{int(p['gw13_fdr']):<5} {p['form']:<6.1f} {p['gw13_score']:<6.3f}")

    # Captain recommendation
    print("\n" + "=" * 100)
    print("CAPTAIN RECOMMENDATION")
    print("=" * 100)

    captain_candidates = squad_gw13[
        squad_gw13['position'].isin(['Midfielder', 'Forward'])
    ].nlargest(3, 'gw13_score')

    print(f"\n{'Rank':<5} {'Player':<18} {'vs':<15} {'H/A':<5} {'FDR':<5} {'Form':<6} {'xGI/90':<7} {'Score':<6}")
    print("-" * 75)

    for rank, (_, p) in enumerate(captain_candidates.iterrows(), 1):
        print(f"{rank:<5} {p['web_name']:<18} {p['gw13_opp']:<15} {p['gw13_venue']:<5} "
              f"{int(p['gw13_fdr']):<5} {p['form']:<6.1f} {p['expected_goal_involvements_per_90']:<7.2f} "
              f"{p['gw13_score']:<6.3f}")

    # Issues and Transfer Suggestions
    print("\n" + "=" * 100)
    print("SQUAD ISSUES & TRANSFER SUGGESTIONS")
    print("=" * 100)

    # Identify weak spots
    print("\n🔴 CONCERNS:")

    # Check each player
    concern_players = []
    for _, p in squad.iterrows():
        issues = []
        team_code = p['team_code']
        schedule = team_schedule.get(team_code, [])

        # Calculate avg FDR
        fdrs = [f['fdr'] for f in schedule if f['opponent'] is not None]
        avg_fdr = sum(fdrs) / len(fdrs) if fdrs else 3

        if avg_fdr >= 3.5:
            issues.append(f"Tough run (avg FDR {avg_fdr:.1f})")
        if p['form'] <= 2 and p['minutes'] >= 270:
            issues.append(f"Poor form ({p['form']:.1f})")
        if p['minutes'] < 180:
            issues.append("Limited minutes")

        if issues:
            concern_players.append((p['web_name'], p['position'], p['team_name'], issues))

    for name, pos, team, issues in concern_players:
        print(f"   • {name} ({pos}, {team}): {', '.join(issues)}")

    # Transfer suggestions
    print("\n💡 TRANSFER SUGGESTIONS:")

    # Find good replacements for weak spots
    # Get players not in squad with good fixtures
    non_squad = stats[~stats['web_name'].isin(MY_SQUAD)].copy()
    non_squad = non_squad[non_squad['minutes'] >= 270]

    # Add fixture info
    for idx, p in non_squad.iterrows():
        team_code = p['team_code']
        schedule = team_schedule.get(team_code, [])
        fdrs = [f['fdr'] for f in schedule if f.get('opponent') is not None]
        non_squad.at[idx, 'avg_fdr'] = sum(fdrs) / len(fdrs) if fdrs else 3

        # GW13 FDR
        gw13_fix = [f for f in schedule if f['gw'] == 13]
        if gw13_fix and gw13_fix[0]['opponent']:
            non_squad.at[idx, 'gw13_fdr'] = gw13_fix[0]['fdr']
            non_squad.at[idx, 'gw13_venue'] = gw13_fix[0]['venue']
            non_squad.at[idx, 'gw13_opp'] = gw13_fix[0]['opp_name']
        else:
            non_squad.at[idx, 'gw13_fdr'] = 5
            non_squad.at[idx, 'gw13_venue'] = '-'
            non_squad.at[idx, 'gw13_opp'] = 'BLANK'

    # Score potential transfers
    non_squad['transfer_score'] = (
        (non_squad['form'] / 10) * 0.3 +
        ((6 - non_squad['avg_fdr']) / 5) * 0.3 +
        ((6 - non_squad['gw13_fdr']) / 5) * 0.2 +
        (non_squad['expected_goal_involvements_per_90'].clip(0, 1.5) / 1.5) * 0.2
    )

    # Suggest by position
    for pos in ['Defender', 'Midfielder', 'Forward']:
        print(f"\n   Best {pos} options:")
        pos_options = non_squad[non_squad['position'] == pos].nlargest(3, 'transfer_score')
        for _, p in pos_options.iterrows():
            price = p['now_cost'] / 10
            print(f"      → {p['web_name']} ({p['team_name']}) - £{price:.1f}m | Form {p['form']:.1f} | "
                  f"Avg FDR {p['avg_fdr']:.1f} | GW13: {p['gw13_opp']}({p['gw13_venue']})")

    # Specific recommendations based on squad issues
    print("\n" + "=" * 100)
    print("PRIORITY TRANSFERS")
    print("=" * 100)

    # Check Spurs players (tough fixtures)
    spurs_players = [p for p in MY_SQUAD if squad[squad['web_name'] == p]['team_name'].values[0] == 'Spurs' if len(squad[squad['web_name'] == p]) > 0]

    print("\n1️⃣ IMMEDIATE (GW13):")

    # Check for players with FDR 4-5 in GW13
    for _, p in squad_gw13.iterrows():
        if p['gw13_fdr'] >= 4:
            # Find replacement
            pos = p['position']
            price = p['now_cost'] / 10
            replacements = non_squad[
                (non_squad['position'] == pos) &
                (non_squad['now_cost'] <= p['now_cost'] + 10) &
                (non_squad['gw13_fdr'] <= 2)
            ].nlargest(2, 'transfer_score')

            if len(replacements) > 0:
                print(f"\n   {p['web_name']} ({p['team_name']}) has tough GW13 fixture (FDR {int(p['gw13_fdr'])})")
                print(f"   Consider:")
                for _, r in replacements.iterrows():
                    r_price = r['now_cost'] / 10
                    print(f"      → {r['web_name']} ({r['team_name']}) - £{r_price:.1f}m | "
                          f"GW13: {r['gw13_opp']}({r['gw13_venue']}) FDR {int(r['gw13_fdr'])}")

    print("\n2️⃣ MEDIUM TERM (Next 4 GWs):")

    # Find players with consistently tough fixtures
    for _, p in squad.iterrows():
        team_code = p['team_code']
        schedule = team_schedule.get(team_code, [])
        fdrs = [f['fdr'] for f in schedule if f.get('opponent') is not None]
        avg_fdr = sum(fdrs) / len(fdrs) if fdrs else 3

        if avg_fdr >= 3.5:
            pos = p['position']
            replacements = non_squad[
                (non_squad['position'] == pos) &
                (non_squad['avg_fdr'] <= 2.5) &
                (non_squad['form'] >= 4)
            ].nlargest(2, 'transfer_score')

            if len(replacements) > 0:
                print(f"\n   {p['web_name']} ({p['team_name']}) has tough run (avg FDR {avg_fdr:.1f})")
                print(f"   Consider:")
                for _, r in replacements.iterrows():
                    r_price = r['now_cost'] / 10
                    print(f"      → {r['web_name']} ({r['team_name']}) - £{r_price:.1f}m | "
                          f"Avg FDR {r['avg_fdr']:.1f} | Form {r['form']:.1f}")

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    # Find best captain
    best_captain = captain_candidates.iloc[0]
    print(f"\n🎯 CAPTAIN: {best_captain['web_name']} vs {best_captain['gw13_opp']} ({best_captain['gw13_venue']})")

    # Best players to play
    print(f"\n✅ MUST START: Haaland (FDR 1, Home vs Leeds)")

    # Players to bench
    print(f"\n🪑 CONSIDER BENCHING: Players with FDR 4+ in GW13")

if __name__ == "__main__":
    main()
