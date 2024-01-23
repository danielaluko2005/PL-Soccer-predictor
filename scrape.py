import requests
from bs4 import BeautifulSoup
import pandas as pd 
import time

base_url = "https://fbref.com/en/comps/9/Premier-League-Stats"

all_data = []

seasons = list(range(2023, 2015, -1))

for season in seasons:
    data = requests.get(base_url)
    soup = BeautifulSoup(data.text, "html.parser")
    standings_table = soup.select('table.stats_table')[0]
    prev_season = soup.select("a.prev")[0].get("href")
    base_url = f"https://fbref.com/{prev_season}"

    links = standings_table.find_all("a")
    links = [l.get("href") for l in links]

    links = [l for l in links if '/squads/' in l]

    team_urls = [f"https://fbref.com{l}" for l in links]

    for team_url in team_urls:
        team_name = team_url.split("/")[-1].replace("-Stats", "").replace("-", " ")

        data = requests.get(team_url)

        matches = pd.read_html(data.text, match="Scores & Fixtures")

        soup = BeautifulSoup(data.text, "html.parser")

        links = soup.find_all("a")

        links = [l.get("href") for l in links]
        
        links = [l for l in links if l and 'all_comps/shooting/' in l]

        data = requests.get(f"https://fbref.com{links[0]}")

        shooting = pd.read_html(data.text, match="Shooting")[0]

        shooting.columns = shooting.columns.droplevel()

        try:
            team_data = matches[0].merge(shooting[["Date", "Sh", "SoT", "Dist", "FK", "PK", "PKatt"]], on="Date")
        except ValueError:
            continue

        team_data = team_data[team_data["Comp"] == "Premier League"]
        team_data["Season"] = season
        team_data["Team"] = team_name
        all_data.append(team_data)
        time.sleep(1)

match_df = pd.concat(all_data)
match_df.columns = [c.lower() for c in match_df.columns]

match_df.to_csv("matches.csv")
