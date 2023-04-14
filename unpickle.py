import json
import pickle

player_json = 'players.json'
pickled_players = 'players.p'

with open(player_json, 'r') as f:
    data = json.load(f)

with open(pickled_players, 'wb') as f:
    pickle.dump(data, f)


