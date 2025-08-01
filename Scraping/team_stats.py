import requests
from bs4 import BeautifulSoup
import json
import os

def get_team_stats_by_year(year):
    url = f"https://www.basketball-reference.com/leagues/NBA_{year}_standings.html"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch page: {response.status_code}")

    soup = BeautifulSoup(response.text, 'html.parser')
    east_table = soup.find('table', {'id': 'confs_standings_E'})
    west_table = soup.find('table', {'id': 'confs_standings_W'})

    if not east_table or not west_table:
        raise Exception("Could not locate standings tables")

    def extract_table_data(table):
        data = []
        for row in table.tbody.find_all('tr'):
            team = row.find('th').text.strip()
            tds = row.find_all('td')
            abbrev = row.find('a')['href'].split('/')[2].split('.')[0].upper()
            wins = int(tds[0].text)
            losses = int(tds[1].text)
            win_pct = float(tds[2].text)
            data.append({
                'Team Name': team,
                'Team Abbreviation': abbrev,
                'Wins': wins,
                'Losses': losses,
                'win_percentage': win_pct
            })
        return data

    all_teams = extract_table_data(east_table) + extract_table_data(west_table)
    sorted_teams = sorted(all_teams, key=lambda x: x['Wins'], reverse=True)

    for i, team in enumerate(sorted_teams):
        team['ranking'] = i + 1

    return sorted_teams
