# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**FPL-Elo-Insights** combines official FPL API data with detailed match statistics and ClubElo team ratings for Fantasy Premier League analysis. Data is sourced from a Supabase database and exported to organized CSV files.

## Key Commands

```bash
# Data export (requires SUPABASE_URL and SUPABASE_KEY env vars)
python scripts/export_data.py

# Install dependencies (requires Python 3.10+)
pip install pandas numpy supabase python-dotenv
```

## Data Structure

Season data lives in `data/{season}/` (current: `2025-2026`):

- **Master files** (root): Latest aggregated `players.csv`, `teams.csv`, `playerstats.csv`, `gameweek_summaries.csv`
- **By Gameweek** (`By Gameweek/GW{x}/`): Point-in-time snapshots with all data files
- **By Tournament** (`By Tournament/{name}/GW{x}/`): Competition-specific data (Premier League, Champions League, EFL Cup, etc.)

## Critical Data Concepts

### Stat Types (in playerstats)

- **Cumulative columns**: `total_points`, `minutes`, `goals_scored`, `assists`, `clean_sheets`, `bonus`, `bps`, `expected_goals`, `expected_assists`, etc. - these accumulate across the season
- **Snapshot columns**: `now_cost`, `form`, `selected_by_percent`, `event_points`, `ep_next` - point-in-time values at each gameweek
- **`player_gameweek_stats.csv`**: Auto-calculated discrete weekly performance (cumulative diffs between gameweeks)

### Key ID Relationships

- `playerstats.id` → `players.player_id`
- `playermatchstats.player_id` → `players.player_id`
- `playermatchstats.match_id` → `matches.match_id`
- `matches.home_team` / `away_team` → `teams.id`

### Tournament Slugs

The `tournament` field in matches uses these slugs:

- `prem` or `premier-league` → Premier League
- `champions-league`, `europa-league`, `conference-league` → European competitions
- `efl-cup` → EFL Cup

## Automation

GitHub Actions runs `scripts/export_data.py` 3x daily (8:15, 17:00, 22:00 UTC). The export:

1. Fetches all tables from Supabase
2. Filters out friendlies and GW0 (pre-season)
3. Organizes data by gameweek and tournament
4. Calculates discrete weekly stats from cumulative data

## Analysis Scripts

Root-level `gw*.py` files are gameweek-specific analysis scripts. Common patterns:

```python
# Load from By Gameweek folder
BASE_PATH = Path("data/2025-2026/By Gameweek")
stats = pd.read_csv(BASE_PATH / f"GW{gw}" / "playerstats.csv")

# Merge player info with stats
merged = stats.merge(players[['player_id', 'team_code', 'position']],
                     left_on='id', right_on='player_id')

# Filter Premier League fixtures
prem_fixtures = fixtures[fixtures['tournament'] == 'prem']
```

## Environment Variables

- `SUPABASE_URL`: Supabase project URL (required for export)
- `SUPABASE_KEY`: Supabase API key (required for export)
