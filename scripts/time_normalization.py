import os
import json
import csv
import argparse

from time_normalize import TimeNormalize



def read_lines(input_file):
    with open(input_file, "r", encoding="utf-8") as fin:
        lines = fin.readlines()
        return lines


'''
Example usage:
python scripts/time_normalization.py --in_file mctaco-data/test_9442.tsv --out_file mctaco-data/normalized_test_9442.tsv
'''
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
    with open(args.out_file, 'w') as wf:
        writer = csv.writer(wf, delimiter='\t')
        count = 0
        for line in lines:
            group = line.strip().split("\t")
            prev = group[2]
            group[2] = norm.get_normalized_field(group[2], convert_surface_words_to_floats=False, number_included=True, post_process_century=True, round_method='closest')
            group[2] = norm.post_convert_num_to_word(group[2], only_convert_first=True, not_convert_time=False)
            if prev != group[2]:
                print(str(prev), '\t',str(group[2]))
                count += 1
            writer.writerow(group)
    print(f'finish writing to {args.out_file} with {count} converted')







