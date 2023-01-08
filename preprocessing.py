import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import random

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
    return df_p


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
    df_normalized = df_subset.apply(lambda x: (x - x.mean()) / x.std())
    df_p[columns_to_normalize] = df_normalized
    return df_p


def adjust_df_size(df_p, m):
    # Get the current number of rows in the dataframe
    n = len(df_p)
    # If the dataframe has less than m rows
    if n < m:
        # Calculate how many rows we need to add
        num_rows_to_add = m - n
        # Get a list of the current row indices
        row_indices = list(df_p.index)
        # Choose num_rows_to_add random row indices to duplicate
        # rows_to_duplicate = random.sample(row_indices, num_rows_to_add)
        rows_to_duplicate = random.choices(row_indices, k=num_rows_to_add)
        # Duplicate the rows and add them to the dataframe
        for i in rows_to_duplicate:
            df_p = df_p.append(df_p.loc[i, :], ignore_index=False)
            df_p = df_p.sort_index()
    # If the dataframe has more than m rows
    if n > m:
        # Calculate how many rows we need to remove
        num_rows_to_remove = n - m
        # Choose num_rows_to_remove random row indices to remove
        rows_to_remove = random.sample(range(n), num_rows_to_remove)
        # Drop the rows from the dataframe
        df_p = df_p.drop(rows_to_remove)

    return df_p


def main(training_folder, new_training_folder, nb_lines_mean):
    for p_file in tqdm(iterable=os.listdir(training_folder), desc=training_folder):
        p_name = p_file.split('.')[0]
        df_p = p_reader(training_folder, p_file)
        list_med = collect_median('Median_Training_SetA.txt')
        df_p = missing_values(df_p, list_med)
        df_p = normalization(df_p)
        df_p = adjust_df_size(df_p, nb_lines_mean)
        try:
            os.mkdir(new_training_folder)
        except OSError:
            pass
        df_p.to_csv(new_training_folder + '/' + p_name + '.csv', index=False)


main('Training_SetA', 'Pp_Trainig_SetA', 40)
