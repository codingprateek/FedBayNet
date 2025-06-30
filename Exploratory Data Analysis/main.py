import argparse
from data_preprocessing import DataPreprocessor

def main():
    parser = argparse.ArgumentParser(description="Preprocess CSV data with encoding.")
    parser.add_argument('--input_csv', required=True, help="Path to input CSV file")
    parser.add_argument('--output_csv', help="Path to output CSV file")
    args = parser.parse_args()

    preprocessor = DataPreprocessor(args.input_csv, args.output_csv)
    preprocessor.preprocess()
    preprocessor.save_data()

if __name__ == '__main__':
    main()
