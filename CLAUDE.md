# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is **FPL-Elo-Insights**, a comprehensive Fantasy Premier League (FPL) dataset that combines official FPL API data with detailed match statistics and ClubElo team ratings. The project provides automated data collection, processing, and analysis for FPL research.

## Key Commands

### Data Export and Processing
```bash
# Main data export script (requires SUPABASE_URL and SUPABASE_KEY environment variables)
python scripts/export_data.py

# Split historical data by gameweek (for 2024-25 season)
python scripts/split_by_gameweek.py

# Fix CSV formatting issues
python scripts/fixcsv.py

# Split CSV data into structured format
python scripts/split_csv_data.py
```

### Running Analysis Scripts
```bash
# Run any of the analysis scripts for gameweek predictions
python gw4_enhanced_predictions.py
python gw4_final_strategy.py
python gw3_team_performance_analysis.py
# (and other gw*.py files for specific gameweek analysis)
```

### Dependencies
Install required Python packages:
```bash
pip install pandas numpy supabase python-dotenv
```

## Architecture

### Data Sources
1. **Supabase Database**: Primary data source containing live FPL and match data
2. **CSV Exports**: Local data files organized by season, gameweek, and tournament
3. **ClubElo.com**: Team strength ratings integrated into match data

### Data Structure
The project follows a hierarchical data organization under `data/{season}/`:

- **Master Files** (`data/2025-2026/`): Current season aggregated data
  - `players.csv`, `teams.csv`, `playerstats.csv`, `gameweek_summaries.csv`

- **By Gameweek** (`data/{season}/By Gameweek/GW{x}/`): Point-in-time snapshots
  - Complete data state for each gameweek
  - Includes `player_gameweek_stats.csv` (discrete weekly performance)

- **By Tournament** (`data/{season}/By Tournament/{tournament}/GW{x}/`): Competition-specific data
  - Premier League, EFL Cup, Champions League, etc.
  - Same file structure as gameweek folders

### Key Data Files
- `matches.csv`: Comprehensive match data with team stats and Elo ratings
- `playermatchstats.csv`: Individual player performance per match (includes CBIT metrics: Clearances, Blocks, Interceptions, Tackles)
- `playerstats.csv`: Cumulative FPL player statistics
- `player_gameweek_stats.csv`: Discrete weekly player performance (auto-calculated from cumulative data)
- `fixtures.csv`: Upcoming matches (same structure as matches.csv)

### Automation
- **GitHub Actions**: Automated data updates via `.github/workflows/`
  - `update_data.yml`: Runs data export 3x daily (8:15 UTC, 17:00 UTC, 22:00 UTC)
  - `splitdata.yml`: Manual workflow for splitting historical CSV data
- **Data Processing**: `scripts/export_data.py` handles the complete pipeline from Supabase to organized CSV files

### Analysis Scripts
Root-level Python files (gw*.py) contain gameweek-specific analysis and prediction models:
- Use historical performance data
- Implement form calculations and player valuations
- Generate transfer recommendations and team selections
- Focus on high-scoring strategies (65+ points)

## Important Notes

- **Environment Variables**: SUPABASE_URL and SUPABASE_KEY must be set for data export
- **Data Filtering**: Export script automatically excludes friendlies and GW0 (pre-season) matches
- **Tournament Mapping**: Match IDs contain tournament slugs mapped to readable names in `TOURNAMENT_NAME_MAP`
- **Discrete Stats**: `player_gameweek_stats.csv` provides week-over-week deltas, not cumulative totals
- **Multi-Competition**: Data includes Premier League, cups, and European competitions linked to FPL player IDs