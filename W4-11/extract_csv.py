import os
import pandas as pd

def read_data_from_files():
    data = []
    for filename in os.listdir('.'):  # Assumes the script is running in the directory containing the .txt files.
        if filename.endswith('.txt'):
            with open(filename, 'r') as file:
                content = file.read()
                molecule_name = filename[:-4]  # Remove the .txt extension to get the molecule name

                # Split content based on "Condition" lines
                sections = content.split('Condition')

                # Initialize the scores dictionary for the molecule
                scores = {'Molecule Name': molecule_name}

                # Loop through each section to extract scores
                for section in sections[1:]:
                    lines = section.strip().split('\n')
                    condition = int(lines[0].split()[0])
                    score_key = f'Score {condition}'
                    score_value = float(lines[1].split(':')[-1].strip())
                    scores[score_key] = score_value

                # Fill missing scores with 0.0
                for condition in [4, 7, 10, 13, 17, 15]:
                    scores.setdefault(f'Score {condition}', 0.0)

                data.append(scores)

    return data

def print_table(data):
    df = pd.DataFrame(data)
    print(df)
    return df

def save_as_csv(df):
    df.to_csv('scores_table.csv', index=False)

if __name__ == "__main__":
    data = read_data_from_files()
    table_df = print_table(data)
    save_as_csv(table_df)
