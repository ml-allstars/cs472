from sportsreference.ncaab.roster import Player
import numpy as np
from player_ids import *


def gather_and_clean_data():
    data = gather_data()
    return clean_data(data)


def gather_data():
    years = [id_2000, id_2001, id_2002, id_2003, id_2004, id_2005]
    instances = []
    for year in years:
        for [player_id, is_all_star] in year:
            player: Player = Player(player_id)
            instance = player_to_instance(player, is_all_star)
            instances.append(instance)
    return np.array(instances)


def clean_data(data):
    data[data == None] = -1
    return data


def position_to_int(position: str):
    positions = ['Center', 'Guard', 'Forward']
    if position in positions:
        return positions.index(position)
    else:
        return None


def height_to_int(height: str):
    [feet, inches] = height.split('-')
    return 12 * int(feet) + int(inches)


def player_to_instance(player: Player, is_all_star):
    return np.array([
        position_to_int(player.position),
        height_to_int(player.height),
        player.weight,
        player.field_goals,
        player.field_goal_attempts,
        player.field_goal_percentage,
        player.three_pointers,
        player.three_point_attempts,
        player.three_point_percentage,
        player.three_pointers,
        player.three_point_attempts,
        player.three_point_percentage,
        player.effective_field_goal_percentage,
        player.free_throws,
        player.free_throw_attempts,
        player.free_throw_percentage,
        player.offensive_rebounds,
        player.defensive_rebounds,
        player.total_rebounds,
        player.assists,
        player.steals,
        player.blocks,
        player.turnovers,
        player.personal_fouls,
        player.points,
        player.true_shooting_percentage,
        player.three_point_attempt_rate,
        player.free_throw_attempt_rate,
        player.offensive_rebound_percentage,
        player.defensive_rebound_percentage,
        player.total_rebound_percentage,
        player.assist_percentage,
        player.steal_percentage,
        player.block_percentage,
        player.turnover_percentage,
        player.usage_percentage,
        int(is_all_star),
    ])
