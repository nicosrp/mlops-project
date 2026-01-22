# Feature Analysis: Current vs Proposed Model

## 🔴 CRITICAL ISSUES WITH CURRENT MODEL

### Current Feature Structure (190 total features)
The model uses **10 historical matches per team** with the following features per match:
- ✅ Goals scored/conceded (makes sense)
- ✅ Team ratings (makes sense)  
- ✅ Opponent ratings (makes sense)
- ❌ **Coach ID per historical match** (DOESN'T MAKE SENSE)
- ❌ **League ID per historical match** (DOESN'T MAKE SENSE)
- ✅ Is home/away (makes sense)
- ✅ Is cup match (makes sense)
- ✅ Days since match (makes sense)

### Why Coach & League Per Historical Match Don't Make Sense

#### ❌ Problem 1: Coach IDs (20 features: 10 home + 10 away)
**Current approach:** Model receives coach ID for each of the 10 historical matches per team
```
home_team_history_coach_1, home_team_history_coach_2, ..., home_team_history_coach_10
away_team_history_coach_1, away_team_history_coach_2, ..., away_team_history_coach_10
```

**The issue:**
- We're predicting a FUTURE match between Team A (home) vs Team B (away)
- This match has exactly **2 coaches**: one for Team A, one for Team B
- Historical coach IDs from 10 past matches are **not relevant** to the current match prediction
- Example: If Team A played 6 months ago with a different coach, that coach ID doesn't help predict today's match

**What happens:**
- Model learns spurious patterns like "coach ID 5000 always loses away games"
- Random coach IDs in manual predictions override all other features
- Explains why our extreme test cases all predicted the same outcome

#### ❌ Problem 2: League IDs per Historical Match (20 features: 10 home + 10 away)
**Current approach:** Model receives league ID for each of the 10 historical matches per team
```
home_team_history_league_id_1, home_team_history_league_id_2, ..., home_team_history_league_id_10
away_team_history_league_id_1, away_team_history_league_id_2, ..., away_team_history_league_id_10
```

**The issue:**
- The match we're predicting happens in **ONE specific league** (e.g., Premier League)
- Historical league IDs are confusing: they represent which league those past matches were played in
- Not clear if this helps: does knowing "Team A played in Championship 6 months ago" help predict today's Premier League match?
- More logical: **Current match league ID** as a single feature (not 20)

**What makes sense:**
- ✅ A **single** `league_id` feature: "This match is in the Premier League (league_id=636)"
- ❌ NOT 20 league IDs from historical matches

---

## ✅ PROPOSED IMPROVED FEATURE SET

### Core Philosophy
**Only include features that:**
1. Directly represent the teams' recent performance (goals, ratings, form)
2. Provide context about the current match (league, home/away)
3. Can be reasonably known/estimated before the match

### Proposed Features (Per Team, 10 Historical Matches)

#### ✅ Performance Metrics (80 features: 8 per match × 10 matches × 2 teams)
**Home team (40 features):**
- `home_team_history_goal_1..10` - Goals scored in last 10 matches
- `home_team_history_opponent_goal_1..10` - Goals conceded in last 10 matches
- `home_team_history_rating_1..10` - Team rating in last 10 matches
- `home_team_history_opponent_rating_1..10` - Opponent strength in last 10 matches

**Away team (40 features):**
- `away_team_history_goal_1..10` - Goals scored in last 10 matches
- `away_team_history_opponent_goal_1..10` - Goals conceded in last 10 matches
- `away_team_history_rating_1..10` - Team rating in last 10 matches
- `away_team_history_opponent_rating_1..10` - Opponent strength in last 10 matches

#### ✅ Context Features (40 features: 2 per match × 10 matches × 2 teams)
**Home team (20 features):**
- `home_team_history_is_play_home_1..10` - Was this historical match at home?
- `home_team_history_is_cup_1..10` - Was this historical match a cup game?

**Away team (20 features):**
- `away_team_history_is_play_home_1..10` - Was this historical match at home?
- `away_team_history_is_cup_1..10` - Was this historical match a cup game?

#### ✅ Time Features (20 features: 1 per match × 10 matches × 2 teams)
**Home team (10 features):**
- `home_team_history_match_days_since_1..10` - Days ago this match happened

**Away team (10 features):**
- `away_team_history_match_days_since_1..10` - Days ago this match happened

#### ✅ Match-Level Features (2 features)
- `league_id` - Single value: which league is THIS match in? (encoded 0-872)
- `is_cup` - Single value: is THIS match a cup game? (0 or 1)

#### ❓ Optional: Coach Features (2 features) - Requires careful consideration
- `home_team_coach_id` - Current home team coach (single value)
- `away_team_coach_id` - Current away team coach (single value)

**Note:** If we include coaches, they should be:
- **Current coaches only** (not historical)
- Validated that they actually help model performance
- Handled carefully in manual predictions (use meaningful defaults or allow selection)

---

## 📊 FEATURE COMPARISON

| Feature Type | Current Model | Proposed Model | Change |
|--------------|---------------|----------------|---------|
| Goals/Conceded | ✅ 40 (10×2×2) | ✅ 40 (10×2×2) | Keep |
| Ratings | ✅ 40 (10×2×2) | ✅ 40 (10×2×2) | Keep |
| Home/Away flags | ✅ 20 (10×2) | ✅ 20 (10×2) | Keep |
| Cup flags (historical) | ✅ 20 (10×2) | ✅ 20 (10×2) | Keep |
| Days since | ✅ 20 (10×2) | ✅ 20 (10×2) | Keep |
| **Coach IDs (historical)** | ❌ 20 (10×2) | ❌ **REMOVE** | -20 |
| **League IDs (historical)** | ❌ 20 (10×2) | ❌ **REMOVE** | -20 |
| **Match league ID** | ❌ None | ✅ **ADD** 1 | +1 |
| **Match is_cup flag** | ❌ None | ✅ **ADD** 1 | +1 |
| **Current coaches** | ❓ Present but not used | ❓ Optional 2 | TBD |
| **TOTAL** | **160** training features | **142** (or 144 with coaches) | **-18 to -16** |

---

## 🎯 EXPECTED IMPROVEMENTS

### 1. Manual Predictions Will Work
- No more random coach IDs causing constant "AWAY" predictions
- Extreme cases (5 goals vs 0 goals) will predict correctly
- Model will actually respond to goal/rating differences

### 2. Model Will Learn Meaningful Patterns
- Focus on performance metrics (goals, ratings)
- Learn home advantage properly
- Understand recent form (via days_since and sequence)

### 3. Simpler Manual Input
- Only need: 10 historical matches' stats + match league + is_cup
- No need to guess/fake coach IDs
- League is a single dropdown, not 20 values

### 4. Interpretability
- Clear what model learns: "Team with more goals in recent matches tends to win"
- Not confused by: "Coach ID 5000 in league 766 always loses"

---

## 🚀 IMPLEMENTATION PLAN

### Step 1: Create New Processed Data
**Script:** `src/mlops_project/data.py` (modify preprocessing)
- Remove coach/league columns from historical features
- Add single `league_id` and `is_cup` as match-level features
- Validate shape: should be **142 features** (or 144 if keeping current coaches)

### Step 2: Update Model Architecture
**Script:** `src/mlops_project/model.py`
- Change `input_size=22` → `input_size=14` (or 15 with match-level features)
  - Current: 22 features × 10 matches = 220 → reshaped to (10, 22)
  - Proposed: 14 features × 10 matches = 140 → reshaped to (10, 14)
  - Match-level features (league, is_cup) either:
    - Added as separate inputs after LSTM
    - Repeated for all 10 timesteps
    - Combined with LSTM output before final layers

### Step 3: Train on GCP
**Script:** `configs/train_gcp.yaml`
- Update data path to new processed data
- Retrain model with new feature set
- Compare validation accuracy to current 64%

### Step 4: Update API & Frontend
**Scripts:** `src/mlops_project/api.py`, `frontend.py`
- Remove coach/league inputs for historical matches
- Add single league dropdown
- Add is_cup checkbox
- Simplify feature creation logic

### Step 5: Validate with Extreme Cases
**Script:** `test_extreme_cases.py`
- Re-run with new model
- Expect: extreme home dominance → HOME prediction
- Expect: balanced → even probabilities
- Expect: extreme away dominance → AWAY prediction

---

## ❓ QUESTIONS TO CONSIDER

### Should we keep current coach features?
**Option A: Remove coaches entirely**
- ✅ Simplest approach
- ✅ No spurious correlations
- ❌ Might lose some signal (good coaches do win more)

**Option B: Keep as match-level features (2 features total)**
- `home_team_coach_id` - Current home coach
- `away_team_coach_id` - Current away coach
- ✅ Might capture coach quality
- ❌ Requires coach selection in Streamlit
- ❌ Needs careful handling of unknown coaches

**Recommendation:** Start without coaches (Option A), add later if needed

### Should we add derived features?
**Potential additions:**
- Recent form: win rate in last 5 matches
- Goal difference: avg goals scored - conceded
- Momentum: rating trend (increasing/decreasing)

**Recommendation:** Start with raw features, add derived features in iteration 2

---

## 📝 SUMMARY

**Current Issue:**
- Model has 20 coach IDs + 20 league IDs that don't make logical sense
- Causes manual predictions to be constant (~33% each)
- Model learns coach/league combinations instead of performance patterns

**Proposed Solution:**
- Remove historical coach/league features (40 features removed)
- Add single match-level league_id + is_cup (2 features added)
- Keep all performance metrics (goals, ratings, home/away, days_since)
- Optionally keep current coach IDs as match-level features

**Expected Result:**
- Model learns from performance, not coach IDs
- Manual predictions work correctly for extreme cases
- Simpler, more interpretable model
- Potentially better validation accuracy

**Next Steps:**
1. ✅ Review this analysis - does it make sense?
2. Create new data processing script
3. Train new model on GCP
4. Compare performance to current model
5. Update API & frontend if new model performs better
