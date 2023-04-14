import argparse
import json
import pickle


def decode_pickle_to_json(pickle_file_path, json_file_path):
    with open(pickle_file_path, 'rb') as f:
        data = pickle.load(f)
    with open(json_file_path, 'w') as f:
        json.dump(data, f)


def decode_json_to_pickle(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    with open(output_file, 'wb') as f:
        # Dump the JSON data into the pickle file
        pickle.dump(data, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Decode a JSON file to pickle format.')
    parser.add_argument('input_file', type=str, help='path to the input pickle file')
    parser.add_argument('output_file', type=str, help='path to the output JSON file')
    args = parser.parse_args()
    decode_json_to_pickle(args.input_file, args.output_file)
    
    """
    parser = argparse.ArgumentParser(description='Decode a pickle file to JSON format.')
    parser.add_argument('input_file', type=str, help='path to the input pickle file')
    parser.add_argument('output_file', type=str, help='path to the output JSON file')
    args = parser.parse_args()

    decode_pickle_to_json(args.input_file, args.output_file)
    """
