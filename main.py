from warnings import warn
from sportsreference.ncaab.roster import Player

from player_ids import id_2000

for player_id in id_2000:
    player = Player(player_id)
    if player is None:
        warn(f"Player {player_id} is NONE")
    else:
        print(player.__dict__)
        # write attributes to csv
