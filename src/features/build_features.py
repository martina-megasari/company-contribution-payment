import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

import copy
from setup import CONFIG, ROOT_DIR
import os

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import logging

pd.options.mode.chained_assignment = None
logging.basicConfig(level=logging.INFO)


def calc_vif(x: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate variance inflation factor
    :param x: dataframe for which the variance inflation factor is to be calculated
    :return: a dataframe with 2 columns [variables, VIF]
    """
    vif = pd.DataFrame()
    vif["variables"] = x.columns
    vif["VIF"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
    return vif


def select_features(x_train: pd.DataFrame, method: str = 'vif'):
    """
    Perform feature selection
    :param x_train: training dataset for which feature selection is to be performed
    :param method: feature selection method
    :return:
    """

    if method == 'vif':
        vif_test = copy.deepcopy(x_train)
        df_vif = calc_vif(vif_test)

        top_vif = df_vif.sort_values(by='VIF', ascending=False).iloc[0]
        cols_to_remove = [top_vif['variables']]
        while top_vif['VIF'] > 10:
            print(top_vif)
            new_columns = list(df_vif.variables.values)
            cols_to_remove.append(top_vif['variables'])

            new_columns.remove(top_vif['variables'])
            df_vif = calc_vif(vif_test[new_columns])
            print()
            print(df_vif)
            top_vif = df_vif.sort_values(by='VIF', ascending=False).iloc[0]

        selected_features = df_vif['variables'].values
    elif method is None:
        selected_features = x_train.columns
    else:
        logging.info("Feature selection method has not been implemented, returning all columns")
        selected_features = x_train.columns
    logging.info("Feature selection done with {}. Selected features are {}".format(method, selected_features))
    return selected_features


def load_data() -> pd.DataFrame:
    """
    Load the dataset based on the filepath set in the config
    :return: pd.DataFrame of the dataset
    """
    df = pd.read_csv(os.path.join(ROOT_DIR, CONFIG['dataset_filepath']))
    df.drop(['company_id'], axis=1, inplace=True)
    df.rename({'median_contribution': 'contribution'}, axis=1, inplace=True)
    df['total_percentage'] = df['employee_percentage'] + df['company_percentage']

    del df['regaddress_posttown']
    del df['state']

    df = df.loc[(df['contribution'] > 0) & (df['contribution'] < 1 * 1e7)]
    df['log_contribution'] = np.log(df['contribution'])
    return df


def impute_missing_values(df: pd.DataFrame) -> None:
    """
    Impute missing values in the dataset, this function modifies the input dataframe directly
    :param df: the dataframe to be imputed
    :return: None
    """
    # impute missing values
    for col in ['company_size_band', 'reg_address_district', 'reg_address_region',
                'sic2007_category', 'legal_structure',
                'freq_period_type']:
        df.loc[df[col].isna(), col] = 'unknown'

    for col in ['company_age']:
        df.loc[df[col].isna(), col] = df[col].median()


def encode_categorical_features(x_train: pd.DataFrame, x_test: pd.DataFrame, y_train, target:str) -> None:
    """
    Encode categorical features into numerical columns, this function modifies the input dataframe directly
    :param x_train: the dataframe containing the training features
    :param x_test: the dataframe containing the testing features
    :param y_train: the target variable of the training dataset
    :return: None
    """
    cols_one_hot_encoding = []
    cols_label_encoding = ['company_size_band']
    cols_target_encoding = ['reg_address_district', 'reg_address_region', 'sic2007_category',
                            'legal_structure', 'freq_period_type']
    cat_var_encoder = {
        'company_size_band': {
            'Zero employees': 0,
            '1) 1 - 5 employees': 3,
            '2) 6 - 10 employees': 8,
            '3) 11 - 49 employees': 30,
            '4) 50 - 99 employees': 75,
            '5) 100 - 249 employees': 175,
            '6) 250 - 499 employees': 375,
            '7) 500 - 999 employees': 750,
            '8) 1000+ employees': 1500
        }
    }
    for col in cols_one_hot_encoding:
        enc = OneHotEncoder(handle_unknown='error', drop='first')
        f = enc.fit_transform(x_train[[col]])
        cat_var_encoder[col] = enc

        x_train.loc[:, ['{}_{}'.format(col, c) for c in enc.categories_[0][1:]]] = f.toarray()
        x_train.drop([col], axis=1, inplace=True)

        x_test.loc[:, ['{}_{}'.format(col, c) for c in enc.categories_[0][1:]]] = f.toarray()
        x_test.drop([col], axis=1, inplace=True)

    x_train.loc[:, target] = y_train
    for col in cols_target_encoding:
        cat_var_encoder[col] = x_train.groupby(col)[target].median()
        x_train.loc[:, col] = x_train[col].map(cat_var_encoder[col])
        x_test.loc[:, col] = x_test[col].map(cat_var_encoder[col])
    del x_train[target]

    for col in cols_label_encoding:
        x_train.loc[:, col] = x_train[col].map(cat_var_encoder[col])
        x_test.loc[:, col] = x_test[col].map(cat_var_encoder[col])


def scale_dataset(x_train: pd.DataFrame, x_test: pd.DataFrame):
    """
    Scale dataset with RobustScaler
    :param x_train: training dataset
    :param x_test: testing dataset
    :param selected_features
    :return:
    """
    scaler = RobustScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test


def build_features():
    is_extended = True
    target = CONFIG['prediction_target']

    df = load_data()
    impute_missing_values(df)

    # build data set
    if is_extended:
        X = df[['company_size_band', 'reg_address_district', 'reg_address_region',
                'sic2007_category', 'legal_structure', 'company_age',
                'freq_period_type', 'company_percentage', 'employee_percentage',
                'median_age', 'median_pensionable_salary', 'total_percentage']].copy()
    else:
        X = df[['company_size_band', 'reg_address_district', 'reg_address_region',
                'sic2007_category', 'legal_structure', 'company_age',
                'freq_period_type']].copy()
    y = df[target].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

    encode_categorical_features(X_train, X_test, y_train, target)
    selected_features = select_features(X_train, CONFIG["feature_selection_method"])
    X_train, X_test = scale_dataset(X_train[selected_features], X_test[selected_features])
    X_train = pd.DataFrame(X_train, columns=selected_features)
    X_test = pd.DataFrame(X_test, columns=selected_features)
    return X_train, X_test, y_train, y_test

