import pandas as pd

# load companies csv
df_companies = pd.read_csv("../data/companies.csv")
df_companies.columns = df_companies.columns.str.lower()

# load contribution csv
df_com_cont = pd.read_csv("../data/contributions.csv")
df_com_cont.drop(df_com_cont.loc[df_com_cont['COMPANY_ID'].isna()].index, inplace=True)
df_com_cont['COMPANY_ID'] = df_com_cont['COMPANY_ID'].astype(int)
df_com_cont.columns = df_com_cont.columns.str.lower()

# calculate the contribution median
company_contrib_median = df_com_cont.groupby(['company_id'])['company_amount_pennies'].median()

# assign the contribution median to df companies
df_companies = df_companies.set_index('company_id')
df_companies['median_contribution'] = company_contrib_median

# remove the companies without any contribution history
df_companies.drop(df_companies.loc[df_companies['median_contribution'].isna()] .index, inplace=True)

# save to csv
df_companies.to_csv("../data/companies_median_monthly_contribution.csv")


