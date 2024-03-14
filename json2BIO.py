import json
import argparse

# 仅仅实现了JSON-MINI TO BIO的功能
def convert_to_bio(json_data):
    bio_data = []

    for entry in json_data:
        text = entry['text']
        labels = entry['label']
        bio_tags = ['O'] * len(text)
        for label in labels:
            start = label['start']
            end = label['end']
            entity_type = label['labels'][0]

            bio_tags[start] = 'B-' + entity_type
            for i in range(start + 1, end):
                bio_tags[i] = 'I-' + entity_type

        for char, tag in zip(text, bio_tags):
            bio_data.append(f"{char} {tag}")

        bio_data.append("")

    return bio_data


def main():
    parser = argparse.ArgumentParser(description="Convert Label Studio JSON to BIO format.")
    parser.add_argument('input_file', type=str, help="Input JSON file path")
    parser.add_argument('output_file', type=str, help="Output BIO file path")
    args = parser.parse_args()

    with open(args.input_file, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
    bio_data = convert_to_bio(json_data)

    with open(args.output_file, 'w', encoding='utf-8') as file:
        for line in bio_data:
            file.write(line + "\n")


if __name__ == "__main__":
    main()
