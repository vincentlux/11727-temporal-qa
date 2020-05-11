import os
import json
import csv
import argparse

from time_normalize import TimeNormalize

def read_lines(input_file):
    with open(input_file, "r", encoding="utf-8") as fin:
        lines = fin.readlines()
        return lines


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file",
                        required=True,
                        help="path to the csv file to convert.")
    parser.add_argument("--out_file",
                        required=True,
                        help="path to the converted csv file")

    args = parser.parse_args()
    lines = read_lines(args.in_file)
    norm = TimeNormalize()

    
    assert args.out_file.endswith('.tsv'), 'out_file should be ended with .tsv'
    seconds = []
    with open(args.out_file, 'w') as wf:
        writer = csv.writer(wf, delimiter='\t')
        for line in lines:
            group = line.strip().split("\t")
            prev = group[2]
            success, second = norm.convert_time_to_seconds(group[2])
            if success:
                seconds.append(second)
                group[2] = second
                # print(str(prev), str(second))
            else:
                group[2] = -1
                print(str(prev), str(-1))
            writer.writerow(group)
    
    print(f'finish writing to {args.out_file} with {len(seconds)} converted')



                



    