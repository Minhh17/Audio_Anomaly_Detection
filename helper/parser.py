from argparse import ArgumentParser

def arg_parser(description=None):
    parser = ArgumentParser(description=description)
    parser.add_argument('-cfg', '--config', help='specify the default .yaml file', required=True)
    parser.add_argument('-f', '--filePath', help='specify the default data file', required=True)
    args = parser.parse_args()
    return args.config