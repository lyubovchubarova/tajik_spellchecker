from typing import Dict, Tuple

def distance_between_symbols(symbol1: str,
                             symbol2: str,
                             qwerty_positions: Dict[str, Tuple]) -> int:
    """Returns the relative distance between two symbols on qwerty keybord"""
    symbol1_x, symbol1_y = qwerty_positions[symbol1][0], qwerty_positions[symbol1][1]
    symbol2_x, symbol2_y = qwerty_positions[symbol2][0], qwerty_positions[symbol2][1]
    distance = abs(symbol1_x - symbol2_x) + abs(symbol1_y - symbol2_y)

    return distance

def create_distances_dict(qwerty_positions: Dict[str, Tuple[int, int]]) -> Dict[str, Dict[str, int]]:
    """Returns the dict of symbols where each symbols maps with the other symbols and its relative distance.
    The distance does not exceed 3"""
    int_distances = {}

    for symbol1 in qwerty_positions:
        int_nested_distances = {}
        for symbol2 in qwerty_positions:
            int_nested_distances[symbol2] = distance_between_symbols(symbol1, symbol2, qwerty_positions)
        int_distances[symbol1] = int_nested_distances

    return int_distances
