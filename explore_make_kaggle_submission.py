import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--template-csv', required=False, help='Path to Kaggle test csv', default='preds_kaggle.csv')
parser.add_argument('--preds', required=True, help='Path to preds csv')
parser.add_argument('--output', required=True, help='Output csv path')

args = parser.parse_args()

df_template = pd.read_csv(args.template_csv)

df_preds = pd.read_csv(args.preds)['0']
#df_preds = pd.read_csv(args.preds)['predictions']
df_template['prediction'] = df_preds
df_template.to_csv(args.output, index=False)

import ipdb;ipdb.set_trace()
