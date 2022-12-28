import pandas as pd
import numpy as np
import os


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


def preprocess(df_p, med_list,):
    # Handle NAN values
    nb_column = 0
    for columns in df_p:
        df_p[columns].fillna(value=round(df_p[columns].median(), 2), inplace=True)  # Round 2
        if np.isnan(df_p[columns][1]):
            df_p[columns] = med_list[nb_column]
        nb_column += 1
    df_p = df_p.drop('EtCO2', axis=1)  # This column never has a value and has therefore been deleted.
    return df_p


def main(training_folder, new_training_folder):
    for p_file in os.listdir(training_folder):
        p_name = p_file.split('.')[0]
        df_p = p_reader(training_folder, p_file)
        list_med = collect_median('Median_Training_SetA.txt')
        df_p = preprocess(df_p, list_med)
        try:
            os.mkdir(new_training_folder)
        except OSError:
            pass
        df_p.to_csv(new_training_folder + '/' + p_name + '.csv', index=False)


main('afac', 'new_afac')
