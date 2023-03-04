import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--directory', required=True)
parser.add_argument('-st', '--sim_type', required=True)
args = parser.parse_args()

print(type(args.directory), args.sim_type)