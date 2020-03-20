# from warnings import warn
from sportsreference.ncaab.roster import Player

from player_ids import id_2000

for player_id in id_2000:
    print(f"trying player id: {player_id}")
    player = Player(player_id)
