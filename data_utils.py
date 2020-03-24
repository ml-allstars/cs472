from sportsreference.ncaab.roster import Player
import numpy as np
from player_ids import years


def gather_and_clean_data():
    data = gather_data()
    return clean_data(data)


def gather_data():
    instances = []
    for year in years:
        for [player_id, is_all_star] in year:
            player: Player = Player(player_id)
            instance = player_to_instance(player, is_all_star)
            instances.append(instance)
    return np.array(instances, dtype=float)


def clean_data(data):
    data[data is None] = -1
    return data


def position_to_num(position: str):
    positions = ['Center', 'Guard', 'Forward']
    if position in positions:
        return positions.index(position)
    else:
        return None


def height_to_num(height: str):
    [feet, inches] = height.split('-')
    return 12 * int(feet) + int(inches)


def to_f(val):
    if val is None:
        val = -1
    return float(val)


def player_to_instance(player: Player, is_all_star):
    return [
        to_f(position_to_num(player.position)),
        to_f(height_to_num(player.height)),
        to_f(player.weight),
        to_f(player.field_goals),
        to_f(player.field_goal_attempts),
        to_f(player.field_goal_percentage),
        to_f(player.three_pointers),
        to_f(player.three_point_attempts),
        to_f(player.three_point_percentage),
        to_f(player.three_pointers),
        to_f(player.three_point_attempts),
        to_f(player.three_point_percentage),
        to_f(player.effective_field_goal_percentage),
        to_f(player.free_throws),
        to_f(player.free_throw_attempts),
        to_f(player.free_throw_percentage),
        to_f(player.offensive_rebounds),
        to_f(player.defensive_rebounds),
        to_f(player.total_rebounds),
        to_f(player.assists),
        to_f(player.steals),
        to_f(player.blocks),
        to_f(player.turnovers),
        to_f(player.personal_fouls),
        to_f(player.points),
        to_f(player.true_shooting_percentage),
        to_f(player.three_point_attempt_rate),
        to_f(player.free_throw_attempt_rate),
        to_f(player.offensive_rebound_percentage),
        to_f(player.defensive_rebound_percentage),
        to_f(player.total_rebound_percentage),
        to_f(player.assist_percentage),
        to_f(player.steal_percentage),
        to_f(player.block_percentage),
        to_f(player.turnover_percentage),
        to_f(player.usage_percentage),
        to_f(is_all_star),
    ]
