import csv
import os

def merge_csvs_in_current_folder(output_filename='trajectories_50.csv'):
    current_folder = os.path.dirname(os.path.abspath(__file__))
    csv_files = [f for f in os.listdir(current_folder) if f.endswith('.csv') and f != output_filename]

    if not csv_files:
        print("No CSV files found in the current folder.")
        return

    header_written = False
    with open(os.path.join(current_folder, output_filename), 'w', newline='', encoding='utf-8') as fout:
        writer = None

        for filename in csv_files:
            file_path = os.path.join(current_folder, filename)
            with open(file_path, 'r', newline='', encoding='utf-8') as fin:
                reader = csv.reader(fin)
                try:
                    header = next(reader)
                except StopIteration:
                    continue  # skip empty file

                if not header_written:
                    writer = csv.writer(fout)
                    writer.writerow(header)
                    header_written = True

                for row in reader:
                    writer.writerow(row)

    print(f"Merged {len(csv_files)} files into {output_filename}")

if __name__ == '__main__':
    merge_csvs_in_current_folder()
