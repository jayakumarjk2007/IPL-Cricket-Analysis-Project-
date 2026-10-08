# IPL Cricket Data Analysis & Winner Prediction

[![Python](https://img.shields.io/badge/Python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458.svg)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243.svg)](https://numpy.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75.svg)](https://plotly.com/)
[![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Plots-4C72B0.svg)](https://seaborn.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org/)

An extensive data science and sports analytics project evaluating the **Indian Premier League (IPL)** across eight seasons (2016–2023). This project conducts deep statistical exploratory data analysis, ball-by-ball match dynamics breakdown, player performance benchmarking (Orange & Purple Caps), interactive visual telemetry with Plotly, and historical probability-based winner prediction built purely with **Pandas** and **NumPy**.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Dataset Specifications](#dataset-specifications)
- [Analysis Pipeline & Modules](#analysis-pipeline--modules)
- [Winner Prediction Methodology](#winner-prediction-methodology)
- [Technologies & Libraries](#technologies--libraries)
- [Setup & Installation](#setup--installation)
- [Usage Instructions](#usage-instructions)
- [Key Insights & Statistical Findings](#key-insights--statistical-findings)
- [Author & Acknowledgments](#author--acknowledgments)

---

## Project Overview

The Indian Premier League is among the most competitive and dynamic T20 cricket tournaments in the world. With thousands of balls bowled and unpredictable match scenarios, data science offers powerful insights into player consistency, venue behaviors, toss impact, and match outcome probabilities.

This repository provides an end-to-end analytical study:
- **Comprehensive Match Records:** Analyzing 476 matches spanning the 2016 to 2023 seasons.
- **Ball-by-Ball Granularity:** Evaluating thousands of individual deliveries to assess player strike rates, economy rates, and wicket dismissal types.
- **Statistical Modeling:** Developing a transparent historical probability heuristic to predict match winners without opaque black-box machine learning dependencies.
- **Interactive Visualizations:** Leveraging Plotly for dynamic, responsive chart exploration alongside publication-ready Matplotlib and Seaborn visuals.

---

## Repository Structure

```text
IPL-Cricket-Analysis-Project-/
├── deliveries.csv        # Ball-by-ball delivery level dataset (innings, batsman, bowler, runs, dismissals)
├── matches.csv           # Match level metadata (476 matches, seasons 2016-2023, scores, toss, winners)
├── ipl_analysis.ipynb    # Main analytical notebook covering all 21 modular stages
├── requirements.txt      # List of dependencies required to run the notebook
└── README.md             # Project documentation and guide
```

---

## Dataset Specifications

The project operates on two interconnected datasets:

### 1. `matches.csv` (Match Metadata)
- **Dimensions:** 476 matches, 17 columns
- **Coverage:** IPL seasons from 2016 to 2023
- **Attributes:**
  - `id`: Unique match identifier
  - `season`: IPL tournament year (2016–2023)
  - `city`, `venue`: Geographic match location and stadium
  - `team1`, `team2`: Competing franchises
  - `toss_winner`, `toss_decision`: Toss outcome and decision (`bat` or `field`)
  - `winner`, `result`, `result_margin_type`, `result_margin`: Match winner and margin (runs/wickets)
  - `team1_score`, `team2_score`: Total innings totals
  - `eliminator`: Binary flag for matches decided via Super Over
  - `umpire1`, `umpire2`: On-field officials

### 2. `deliveries.csv` (Ball-by-Ball Telemetry)
- **Attributes:**
  - `match_id`: Foreign key referencing `matches.csv`
  - `inning`, `over`, `ball`: Progression tracking
  - `batting_team`, `bowling_team`: Active teams
  - `batsman`, `bowler`: On-strike batsman and bowler
  - `batsman_runs`, `extra_runs`, `total_runs`: Scoring breakdown
  - `is_wicket`, `dismissal_kind`, `player_dismissed`: Fall-of-wicket information

---

## Analysis Pipeline & Modules

The `ipl_analysis.ipynb` notebook is organized into 21 sequential, modular sections:

1. **Data Loading & Inspection:** Verification of data integrity, types, and dimensions.
2. **Data Cleaning:** Normalization of franchise names (accounting for historical renames such as Delhi Daredevils to Delhi Capitals, Rising Pune Supergiant variations), handling missing venue/city records.
3. **Basic Match Statistics:** Total matches, win proportions, distribution of margin types.
4. **Data Visualization:** Exploratory distributions of winning margins, venues, and seasonal game counts.
5. **Team & Toss Analysis:** Correlation between winning the toss and winning the match; breakdown of batting first vs. chasing win rates.
6. **Season-wise Winner Analysis:** Trophy count, dominant teams, and final match histories.
7. **Super Over Analysis:** Detailed breakdown of tied fixtures decided by eliminator overs.
8. **High Scoring Matches (200+ Runs):** Frequency and trends of scores exceeding 200 runs across seasons and venues.
9. **Prepare Match + Ball Data:** Relational join of `matches.csv` and `deliveries.csv` for fine-grained player analytics.
10. **Batting Overview:** Aggregated run totals, boundaries (4s and 6s), balls faced, and strike rates.
11. **Bowling Overview:** Total deliveries bowled, runs conceded, dot ball percentages, and bowling economy rates.
12. **Orange Cap Winners by Season:** Identifying the highest run-scorer for each tournament season.
13. **Purple Cap Winners by Season:** Tracking leading wicket-takers year-over-year.
14. **Player Comparison:** Side-by-side head-to-head metrics comparing elite batsmen and bowlers.
15. **Team Head-to-Head Analysis:** Historical win/loss records between specific franchise rivalries (e.g., MI vs CSK, RCB vs KKR).
16. **Winner Prediction using Pandas & NumPy:** Formulating an algorithmic prediction engine based on historical win ratios, venue advantages, and head-to-head records.
17. **Prediction Accuracy Assessment:** Calculating model precision, confusion matrices, and overall prediction hit rate.
18. **Prediction Accuracy by Year:** Seasonal backtesting to examine consistency across changing team rosters.
19. **Interactive Plotly Visualization:** Dynamic hover-enabled charts for interactive exploration.
20. **Key Insights:** Actionable sports analytics conclusions and strategic takeaways.

---

## Winner Prediction Methodology

Rather than relying on black-box classification libraries, the prediction engine is implemented transparently using **NumPy** and **Pandas**:

$$\text{Win Probability} = w_1 \cdot P_{\text{H2H}} + w_2 \cdot P_{\text{Venue}} + w_3 \cdot P_{\text{Recent Form}} + w_4 \cdot P_{\text{Toss Advantage}}$$

- **Head-to-Head Index ($P_{\text{H2H}}$):** Historical winning percentage between the two competing franchises.
- **Venue Index ($P_{\text{Venue}}$):** Winning track record of each franchise at the specific stadium.
- **Toss Factor ($P_{\text{Toss Advantage}}$):** Historical impact of toss decisions (batting first vs chasing) at that specific ground.
- **Backtesting Evaluation:** Models are backtested season-by-season against actual historical match outcomes to verify predictive power.

---

## Technologies & Libraries

| Technology | Purpose |
|---|---|
| **Python 3.8+** | Core programming language |
| **Pandas** | Data frame manipulation, grouping, aggregations, and feature engineering |
| **NumPy** | High-performance vector operations and probabilistic calculations |
| **Matplotlib** | Baseline statistical charts and distribution plots |
| **Seaborn** | Heatmaps, violin plots, and styling |
| **Plotly** | Interactive dashboards and dynamic visual inspection |
| **Jupyter Notebook** | Reproducible execution and interactive narrative presentation |

---

## Setup & Installation

### Prerequisites
- Python 3.8 or higher installed on your system
- Git installed

### 1. Clone the Repository
```bash
git clone https://github.com/jayakumarjk2007/IPL-Cricket-Analysis-Project-.git
cd IPL-Cricket-Analysis-Project-
```

### 2. Create and Activate a Virtual Environment
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Usage Instructions

### Run the Analysis Notebook
Launch Jupyter Notebook or JupyterLab:
```bash
jupyter notebook
```
In your browser:
1. Open `ipl_analysis.ipynb`.
2. Run cells sequentially (`Shift + Enter`) or select **Cell > Run All** from the top menu.
3. Interact with the Plotly figures directly inside the notebook.

### Perform Custom Head-to-Head Queries
Inside the notebook, you can query any two teams or any player:
```python
# Example: Query head-to-head record
team_a = "Mumbai Indians"
team_b = "Chennai Super Kings"

h2h = matches[
    ((matches["team1"] == team_a) & (matches["team2"] == team_b)) |
    ((matches["team1"] == team_b) & (matches["team2"] == team_a))
]
print(h2h["winner"].value_counts())
```

---

## Key Insights & Statistical Findings

- **Toss Advantage:** Teams electing to field first won over **54%** of matches across 2016–2023, reflecting the growing chasing advantage under evening dew conditions.
- **High-Scoring Evolution:** The frequency of 200+ team totals increased substantially in recent seasons (2022–2023), driven by higher batting strike rates in powerplays and death overs.
- **Dominant Franchises:** Mumbai Indians and Chennai Super Kings maintained the highest historical win percentages and championship conversions across the 2016–2023 timeframe.
- **Eliminator Games:** Super Over tied encounters accounted for ~2.5% of total matches, highlighting the razor-thin margins in elite T20 competition.

---

## Author & Acknowledgments

- **Author:** Jayakumar P
- **GitHub:** [@jayakumarjk2007](https://github.com/jayakumarjk2007)
- **Data Source:** Official IPL historical match scorecards and delivery telemetry.
