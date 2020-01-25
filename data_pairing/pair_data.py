"""
Script for recreating the FactCC dataset from CNN/DM Story files.

CNN/DM Story files can be downloaded from https://cs.nyu.edu/~kcho/DMQA/

Unpaired FactCC data should be downloaded and unpacked.
CNN/DM data to be stored in a `cnndm` directory with `cnn` and `dm` sub-directories.
"""
import argparse
import json
import os

from tqdm import tqdm


def parse_story_file(content):
    """
    Remove article highlights and unnecessary white characters.
    """
    content_raw = content.split("@highlight")[0]
    content = " ".join(filter(None, [x.strip() for x in content_raw.split("\n")]))
    return content


def main(args):
    """
    Walk data sub-directories and recreate examples
    """
    for path, _, filenames in os.walk(args.unpaired_data):
        for filename in filenames:
            if not ".jsonl" in filename:
                continue

            unpaired_path = os.path.join(path, filename)
            print("Processing file:", unpaired_path)

            with open(unpaired_path) as fd:
                dataset = [json.loads(line) for line in fd]

            for example in tqdm(dataset):
                story_path = os.path.join(args.story_files, example["filepath"])

                with open(story_path) as fd:
                    story_content = fd.read()
                    example["text"] = parse_story_file(story_content)

            paired_path = unpaired_path.replace("unpaired_", "")
            os.makedirs(os.path.dirname(paired_path), exist_ok=True)
            with open(paired_path, "w") as fd:
                for example in dataset:
                    fd.write(json.dumps(example, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("unpaired_data", type=str, help="Path to directory holding unpaired data")
    PARSER.add_argument("story_files", type=str, help="Path to directory holding CNNDM story files")
    ARGS = PARSER.parse_args()
    main(ARGS)
