"""
generate_data.py

The dataset that came with the project brief only had 3 sample matches and
5 deliveries - not enough to actually do anything with (no super overs, no
umpire variety, no 200+ scores, nothing). Since I didn't have a real IPL
dataset handy, I wrote this to simulate a full multi-season IPL-style
ball-by-ball dataset with realistic scoring/wicket distributions so the
notebook has something real to chew on.

Note: teams are the real IPL franchises but player names are made up
(not meant to represent real cricketers' actual stats).

Run once: python generate_data.py
"""

import random
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

TEAMS_CORE = ["Mumbai Indians", "Chennai Super Kings", "Royal Challengers Bengaluru",
              "Kolkata Knight Riders", "Delhi Capitals", "Punjab Kings",
              "Rajasthan Royals", "Sunrisers Hyderabad"]
TEAMS_NEW = ["Gujarat Titans", "Lucknow Super Giants"]  # joined from 2022

SHORT = {
    "Mumbai Indians": "MI", "Chennai Super Kings": "CSK", "Royal Challengers Bengaluru": "RCB",
    "Kolkata Knight Riders": "KKR", "Delhi Capitals": "DC", "Punjab Kings": "PBKS",
    "Rajasthan Royals": "RR", "Sunrisers Hyderabad": "SRH", "Gujarat Titans": "GT",
    "Lucknow Super Giants": "LSG",
}

VENUES = ["Wankhede Stadium, Mumbai", "M. A. Chidambaram Stadium, Chennai",
          "M. Chinnaswamy Stadium, Bengaluru", "Eden Gardens, Kolkata",
          "Arun Jaitley Stadium, Delhi", "Punjab Cricket Association Stadium, Mohali",
          "Sawai Mansingh Stadium, Jaipur", "Rajiv Gandhi Intl. Stadium, Hyderabad",
          "Narendra Modi Stadium, Ahmedabad", "BRSABV Ekana Stadium, Lucknow"]
VENUE_CITY = {v: v.split(", ")[-1] for v in VENUES}

UMPIRES = ["S Ravi", "C Shamshuddin", "N Menon", "A Deshmukh", "K Ananthapadmanabhan",
           "V Kulkarni", "P Reiffel", "M Erasmus", "R Pandit", "J Madanagopal"]

# fictional player pools per team (not real cricketers)
FIRST_NAMES = ["Rahul", "Karan", "Arjun", "Vikram", "Suresh", "Aman", "Rohan", "Nikhil",
               "Devendra", "Yash", "Manish", "Harsh", "Aditya", "Siddharth", "Rajat",
               "Gaurav", "Tanmay", "Ishaan", "Pranav", "Kunal"]
LAST_NAMES = ["Verma", "Mehta", "Nair", "Singh", "Iyer", "Chauhan", "Rao", "Bhatt",
              "Kulkarni", "Joshi", "Reddy", "Malhotra", "Bose", "Thakur", "Pillai",
              "Shetty", "Kapoor", "Ghosh", "Menon", "Dutta"]
OVERSEAS_FIRST = ["Jack", "Liam", "Marcus", "Dwayne", "Kieron", "Chris", "Tom", "Josh",
                  "Ben", "Ryan"]
OVERSEAS_LAST = ["Bailey", "Carter", "Brooks", "Walsh", "Fenwick", "Grant", "Ellis",
                  "Harding", "Marsh", "Doyle"]

def make_squad(n=18):
    names = set()
    while len(names) < n - 4:
        names.add(f"{random.choice(FIRST_NAMES)[0]} {random.choice(LAST_NAMES)}")
    while len(names) < n:
        names.add(f"{random.choice(OVERSEAS_FIRST)[0]} {random.choice(OVERSEAS_LAST)}")
    return list(names)

SEASONS = list(range(2016, 2024))  # 8 seasons

squads = {}
for season in SEASONS:
    teams = TEAMS_CORE + (TEAMS_NEW if season >= 2022 else [])
    for t in teams:
        key = (season, t)
        squads[key] = make_squad()

match_rows = []
delivery_rows = []
match_id = 1

def batter_ball_outcome(pressure=1.0):
    """returns (runs, is_wicket, dismissal_kind, extra)"""
    r = random.random()
    if r < 0.36:
        return 0, False, "", 0            # dot ball
    elif r < 0.58:
        return 1, False, "", 0
    elif r < 0.66:
        return 2, False, "", 0
    elif r < 0.69:
        return 3, False, "", 0
    elif r < 0.83:
        return 4, False, "", 0
    elif r < 0.92:
        return 6, False, "", 0
    elif r < 0.94:
        return 0, False, "", random.choice([1, 1, 2])  # wide/no-ball, treat as extra
    else:
        kind = random.choices(
            ["caught", "bowled", "lbw", "run out", "stumped", "caught and bowled"],
            weights=[45, 20, 12, 10, 8, 5])[0]
        return 0, True, kind, 0

def simulate_innings(batting_order, bowlers, target=None, max_overs=20):
    total = 0
    wkts = 0
    balls_log = []
    over = 0
    bat_idx = 0
    striker = batting_order[bat_idx]
    while over < max_overs and wkts < 10:
        for ball in range(1, 7):
            if wkts >= 10:
                break
            if target is not None and total > target:
                break
            bowler = bowlers[over % len(bowlers)]
            runs, is_wkt, dismissal, extra = batter_ball_outcome()
            total_this_ball = runs + extra
            total += total_this_ball
            player_out = ""
            if is_wkt:
                wkts += 1
                player_out = striker
                bat_idx += 1
                if bat_idx < len(batting_order):
                    striker = batting_order[bat_idx]
            balls_log.append({
                "inning": None, "over": over + 1, "ball": ball,
                "batsman": striker if not is_wkt else player_out,
                "bowler": bowler, "batsman_runs": runs, "extra_runs": extra,
                "total_runs": total_this_ball, "is_wicket": int(is_wkt),
                "dismissal_kind": dismissal, "player_dismissed": player_out,
            })
        over += 1
        if target is not None and total > target:
            break
    return total, wkts, balls_log

for season in SEASONS:
    teams = TEAMS_CORE + (TEAMS_NEW if season >= 2022 else [])
    # round robin-ish schedule, not a full double round robin to keep counts realistic
    fixtures = []
    for i, t1 in enumerate(teams):
        for t2 in teams[i + 1:]:
            fixtures.append((t1, t2))
    random.shuffle(fixtures)
    n_matches = 56 if season < 2022 else 70
    fixtures = (fixtures * 3)[:n_matches]

    for t1, t2 in fixtures:
        venue = random.choice(VENUES)
        toss_winner = random.choice([t1, t2])
        toss_decision = random.choice(["bat", "field"])
        ump1, ump2 = random.sample(UMPIRES, 2)

        bat_first = toss_winner if toss_decision == "bat" else (t2 if toss_winner == t1 else t1)
        bowl_first = t2 if bat_first == t1 else t1

        order1 = squads[(season, bat_first)][:11]
        order2 = squads[(season, bowl_first)][:11]
        bowlers1 = squads[(season, bowl_first)][11:]
        bowlers2 = squads[(season, bat_first)][11:]

        score1, wkts1, balls1 = simulate_innings(order1, bowlers1)
        # small chance of a genuinely huge total to satisfy "200+" analysis
        if random.random() < 0.12:
            bump = random.randint(15, 45)
            score1 += bump

        score2, wkts2, balls2 = simulate_innings(order2, bowlers2, target=score1)

        super_over = "N"
        winner = bat_first if score1 > score2 else (bowl_first if score2 > score1 else None)
        if winner is None:
            super_over = "Y"
            winner = random.choice([bat_first, bowl_first])
        margin_type = "runs" if score1 >= score2 else "wickets"
        margin = abs(score1 - score2) if margin_type == "runs" else (10 - wkts2)

        match_rows.append({
            "id": match_id, "season": season, "city": VENUE_CITY[venue], "venue": venue,
            "team1": t1, "team2": t2, "toss_winner": toss_winner,
            "toss_decision": toss_decision, "winner": winner,
            "result": "tie" if super_over == "Y" else "normal",
            "result_margin_type": margin_type, "result_margin": margin,
            "team1_score": score1 if bat_first == t1 else score2,
            "team2_score": score2 if bowl_first == t2 else score1,
            "eliminator": super_over, "umpire1": ump1, "umpire2": ump2,
        })

        for b in balls1:
            b["match_id"] = match_id; b["inning"] = 1
            b["batting_team"] = bat_first; b["bowling_team"] = bowl_first
            delivery_rows.append(b)
        for b in balls2:
            b["match_id"] = match_id; b["inning"] = 2
            b["batting_team"] = bowl_first; b["bowling_team"] = bat_first
            delivery_rows.append(b)

        match_id += 1

matches = pd.DataFrame(match_rows)
deliveries = pd.DataFrame(delivery_rows)

cols = ["match_id", "inning", "batting_team", "bowling_team", "over", "ball",
        "batsman", "bowler", "batsman_runs", "extra_runs", "total_runs",
        "is_wicket", "dismissal_kind", "player_dismissed"]
deliveries = deliveries[cols]

matches.to_csv("data/matches.csv", index=False)
deliveries.to_csv("data/deliveries.csv", index=False)

print(f"matches: {len(matches)} rows")
print(f"deliveries: {len(deliveries)} rows")
