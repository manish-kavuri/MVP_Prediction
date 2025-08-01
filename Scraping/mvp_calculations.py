import requests
from bs4 import BeautifulSoup


def get_mvps(given_year):
    if given_year == 2025:
        # TEMPORARY: manually return the MVP(s) if known, or empty list
        return ['Shai Gilgeous-Alexander']  # or [] if you want to skip labeling for now

    url = f"https://www.basketball-reference.com/awards/awards_{given_year}.html"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    nba_winners = soup.find('table', {'id': 'nba'})

    if nba_winners is None:
        raise ValueError("Couldn't find the MVP table on the page. Structure may have changed.")

    rows = nba_winners.find_all('tr')
    mvps = []

    for row in rows:
        th = row.find('th', {'scope': 'row'})
        if th and 'MVP' in th.text:
            player_cell = row.find('td')
            if player_cell:
                player_name = player_cell.text.strip().replace('*', '')
                mvps.append(player_name)

    return mvps

