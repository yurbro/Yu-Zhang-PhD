import pandas as pd
import random
from icecream.icecream import ic

def create_train_test_sets(file_name):
    # Load the spreadsheet
    # file_path = r'C:\Users\yz02380\OneDrive - University of Surrey\Science Research\Codes\Early_Decision\GPR_BO_Decision\Dataset\Dataset_EachFormula-Loocv.xlsx'
    file_path = file_name
    sheet_name = 'All-oa'

    # Read the data from the specified sheet
    data = pd.read_excel(file_path, sheet_name=sheet_name)

    # Extract formulas and their values
    formulas = data.iloc[:, 0].tolist()
    values = data.iloc[:, 1].tolist()

    # Create a DataFrame to facilitate operations
    df = pd.DataFrame({'Formula': formulas, 'Value': values})

    def create_train_test_sets(df):
        while True:
            # Randomly shuffle the data without a fixed random seed
            df_shuffled = df.sample(frac=1).reset_index(drop=True)

            # Split the shuffled data into training and testing sets
            train_set = df_shuffled.iloc[:12]
            test_set = df_shuffled.iloc[12:]

            # Ensure that the test set has four values below and four values above the maximum value in the training set
            train_max = train_set['Value'].max()

            # Separate the test set into below and above the max value of the training set
            below_max = test_set[test_set['Value'] < train_max]
            above_max = test_set[test_set['Value'] >= train_max]

            # Ensure we have exactly four in each category
            if len(below_max) >= 4 and len(above_max) >= 4:
                below_max = below_max.sample(4)
                above_max = above_max.sample(4)
                final_test_set = pd.concat([below_max, above_max]).reset_index(drop=True)
                break

        return train_set, final_test_set

    # Create the train and test sets
    train_set, final_test_set = create_train_test_sets(df)

    # Find the maximum value and corresponding formula in the training set
    max_formula = train_set.loc[train_set['Value'].idxmax()]

    # Extract the names of the formulas in the test set
    test_names = final_test_set['Formula'].tolist()

    # Extract the names of the formulas in the training set
    train_names = train_set['Formula'].tolist()

    # Load each sheet corresponding to the training set formulas and combine them into a single DataFrame
    combined_train_df = pd.concat([pd.read_excel(file_path, sheet_name=name) for name in train_names])


    # Output the results
    print("Training Set (12 formulas):")
    print(train_set)
    print("\nTesting Set (8 formulas):")
    print(final_test_set)
    print("\nMaximum in Training Set:")
    print(f"Formula: {max_formula['Formula']}, Value: {max_formula['Value']}")
    print("\nTest Set Names:")
    print(f"test_names = {test_names}")

    # Output the combined DataFrame
    # print("Combined Training Set DataFrame:")
    # print(combined_train_df.head())
    # ic(combined_train_df.shape, combined_train_df)

    return test_names, combined_train_df