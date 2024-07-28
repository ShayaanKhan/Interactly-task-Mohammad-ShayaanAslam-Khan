import pandas as pd

def load_data(candidate_data_path='data\RecruterPilot candidate sample input dataset - Sheet1.csv', training_data_path='data/resumes.csv'):
    candidate_df = pd.read_csv(candidate_data_path)
    training_df = pd.read_csv(training_data_path)
    return candidate_df, training_df

if __name__ == "__main__":
    candidate_df, training_df = load_data()
    print(candidate_df.head())
    print(training_df.head())
