# from warnings import warn
import numpy as np
from data_utils import *

<<<<<<< HEAD
data = gather_and_clean_data()
=======
from player_ids import *

years = [id_2000, id_2001, id_2002, id_2003, id_2004, id_2005, id_2006, id_2007, id_2008, id_2009, id_2010, id_2011]
for year in years:
    for [player_id, is_all_star] in year:
        print(player_id)
        player = Player(player_id)


# conference, position, height, weight, offensive win shares, defensive win shares, minutes/games, fg, fga, fgp, threes, 3pa, 3pp, twos, 2pa, 2pp, ft, fta, tfp, ORB, DRB, rb, asst, steal, block, tos, fouls, points, true shooting percent, effect. field goal percent, attempt rates,
>>>>>>> 9d50bfce305d10a80147b8fa08cf9b30cf40bf52
