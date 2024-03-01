from generator import Generator
from config import Config

import argparse


def main(args):
    config = Config.from_file(args.config)
    generator = Generator(config)
    generator.run()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.json')
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
