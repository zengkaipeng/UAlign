import pandas
import os
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--path', required=True)
args = parser.parse_args()


base_char = set(
    [chr(ord('0') + x) for x in range(10)] +
    [chr(ord('a') + x) for x in range(26)] +
    [chr(ord('A') + x) for x in range(26)]
)


set_all = set()


for x in os.listdir(args.path):
    if x.endswith('.csv'):
        df_data = pandas.read_csv(os.path.join(args.path, x))
        for resu in df_data['reactants>reagents>production']:
            rea, prd = resu.strip().split('>>')
        char_rea = set(rea)
        char_prd = set(prd)
        set_all |= char_rea
        set_all |= char_prd


set_all |= base_char
set_all.update(['@', '/', '\\', '#', '$'])
set_all |= {'=', ':', '(', ']', '+', '-', ')', '.', '['}

with open('All_token.json', 'w') as Fout:
    json.dump(list(set_all), Fout, indent=4)
