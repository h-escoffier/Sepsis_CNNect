import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


def p_reader(folder, file):
    """
    Transform a p______.psv to a dataframe
    :param folder:
    :param file:
    :return:
    """
    with open(folder + '/' + file, 'r') as p_file:
        content = [x.strip('\n').split('|') for x in p_file.readlines()]
    df_p = (pd.DataFrame(content[1:], columns=content[0]))
    df_p = df_p.replace('NaN', np.nan)
    for columns in df_p:
        df_p[columns] = df_p[columns].astype(float)
    sum_na = df_p.isna().sum()
    return df_p, sum_na


def collect_median(median_file):
    with open(median_file, 'r') as med_file:
        med_list = [x.strip('\n') for x in med_file.readlines()]
    med_list = list(map(float, med_list))
    return med_list


def missing_values(df_p, med_list):
    # Handle NAN values
    nb_column = 0
    for columns in df_p:
        df_p[columns].fillna(value=round(df_p[columns].median(), 2), inplace=True)  # Round 2
        if np.isnan(df_p[columns][1]):
            df_p[columns] = med_list[nb_column]
        nb_column += 1
    df_p = df_p.drop('EtCO2', axis=1)  # This column never has a value and has therefore been deleted.
    return df_p


def normalization(df_p):
    columns_to_normalize = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'BaseExcess', 'HCO3', 'FiO2', 'pH',
                            'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine',
                            'Bilirubin_direct', 'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium', 'Bilirubin_total',
                            'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets', 'Age', 'HospAdmTime',
                            'ICULOS']
    df_subset = df_p[columns_to_normalize]
    df_normalized = df_subset.apply(lambda x: (x - x.mean()) / x.std() if x.std() != 0 else x - x.mean())
    df_p[columns_to_normalize] = df_normalized
    return df_p


def adjust_df_size(df_p, m):
    # Get the current number of rows in the dataframe
    n = len(df_p)
    # If the dataframe has less than m rows
    while n < m:
        random_row = df_p.sample()
        # Get the index of the random row
        random_row_index = random_row.index[0]
        # Select the row with index random_row_index
        row = df_p.loc[random_row_index, :]
        # Append the row to the dataframe
        df_p = df_p.append(row, ignore_index=True)
        # Sort the dataframe by its indices
        df_p = df_p.sort_index()
        n = len(df_p)
    # If the dataframe has more than m rows
    while n > m:
        random_row = df_p.sample()
        # Get the index of the random row
        random_row_index = random_row.index[0]
        # Drop the random row from the dataframe
        df_p = df_p.drop(random_row_index)
        n = len(df_p)
    return df_p


def main(training_folder, new_training_folder, nb_lines_mean, median_file):
    total_na = 0
    try:
        os.mkdir(new_training_folder)
    except OSError:
        pass
    for p_file in tqdm(iterable=os.listdir(training_folder), desc=training_folder):
        p_name = p_file.split('.')[0]
        df_p, sum_na = p_reader(training_folder, p_file)
        # total_na += sum_na
        list_med = collect_median(median_file)
        df_p = missing_values(df_p, list_med)
        df_p = normalization(df_p)
        df_p = adjust_df_size(df_p, nb_lines_mean)
        df_p.to_csv(new_training_folder + '/' + p_name + '.csv', index=False)
    print(total_na)


main('Training_SetA', 'PP_T_SetA', 40, 'medians/Median_Training_SetB.txt')
main('Training_SetB', 'PP_T_SetB', 40, 'medians/Median_Training_SetB.txt')
