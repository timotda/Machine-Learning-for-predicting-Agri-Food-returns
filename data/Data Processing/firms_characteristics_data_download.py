# Import libraries
import wrds

# WRDS connection
db = wrds.Connection(wrds_username="candicebus")

# Create SQL query
sql_query = f"""
            SELECT datadate, cusip, conm, tic, oibdpq, capxy, invtq, actq, ancq, ltq, lctq, niq, cogsq, revtq --sic, permno, permco
            FROM compa.fundq
            WHERE fic='USA' AND datadate>='2000-01-01' --AND loc='USA'
            """

# datadate
# cusip
# conm: Company Name
# tic: Ticker
# oibdpq: Operating Income Before Depreciation - Quarterly
# capxy: Capital Expenditures
# invtq: Inventories - Total
# actq: Current Assets - Total
# ancq: Non-Current Assets - Total
# ltq: Liabilities - Total
# lctq: Current Liabilities - Total
# niq: Net Income (Loss)
# cogsq: Cost of Goods Sold
# revtq: Revenue - Total

# Fetch data
data_financial_predictors = db.raw_sql(sql_query)

# Save data to a csv file
data_financial_predictors.to_csv('firms_characteristics_data.csv', index=False)
