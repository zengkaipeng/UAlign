from model import OverallModel
from data_utils import avg_edge_logs
import argparse
import pandas

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parser')
    parser.add_argument(
        '--model_config', required=True, type=str,
        help='the path of model config for well trained model'
    )
    parser.add_argument(
        '--file', type=str, required=True,
        help='the path of file containing the test set'
    )

    args = parser.parse_args()

    meta_file = pandas.read_csv(args.file)

    
