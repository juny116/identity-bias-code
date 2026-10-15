"""Compute Pearson correlation between two score columns in a CSV file."""
import argparse
import csv

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--x", required=True)
    parser.add_argument("--y", required=True)
    args = parser.parse_args()
    from scipy.stats import pearsonr
    with open(args.input) as stream:
        rows = list(csv.DictReader(stream))
    print(pearsonr([float(row[args.x]) for row in rows], [float(row[args.y]) for row in rows]))

if __name__ == "__main__":
    main()
