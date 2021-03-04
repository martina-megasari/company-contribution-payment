import logging
import pandas as pd
from setup import ROOT_DIR, CONFIG
import os


def main():
    """
    Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # load companies csv
    df_companies = pd.read_csv(os.path.join(ROOT_DIR, 'data/raw/companies.csv'))
    df_companies.columns = df_companies.columns.str.lower()

    # load contribution csv
    df_emp_cont = pd.read_csv(os.path.join(ROOT_DIR, 'data/raw/employees_contributions.csv'))
    df_emp_cont.drop(df_emp_cont.loc[df_emp_cont['COMPANY_ID'].isna()].index, inplace=True)
    df_emp_cont['COMPANY_ID'] = df_emp_cont['COMPANY_ID'].astype(int)
    df_emp_cont.columns = df_emp_cont.columns.str.lower()
    df_emp_cont['pensionable_salary'] = df_emp_cont['gross_qualifying_earnings_pennies'] \
        .combine_first(df_emp_cont['pensionable_earnings_pennies'])

    df_emp_cont = df_emp_cont.groupby(['y', 'm', 'company_id']).agg({'company_percentage': 'median',
                                                                     'employee_percentage': 'median',
                                                                     'total_amount_pennies': 'sum',
                                                                     'age': 'median',
                                                                     'pensionable_salary': 'sum'}).reset_index()

    df_emp_cont = df_emp_cont.groupby('company_id').agg({'company_percentage': 'median',
                                                         'employee_percentage': 'median',
                                                         'total_amount_pennies': 'median',
                                                         'age': 'median',
                                                         'pensionable_salary': 'median'})

    # assign the contribution median to df companies
    df_companies = df_companies.set_index('company_id')

    df_companies[['company_percentage', 'employee_percentage',
                  'median_contribution', 'median_age', 'median_pensionable_salary']] = df_emp_cont

    # remove the companies without any contribution history
    df_companies.drop(df_companies.loc[df_companies['median_contribution'].isna()].index, inplace=True)

    # save to csv
    df_companies.to_csv(os.path.join(ROOT_DIR, CONFIG['dataset_filepath']))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()




