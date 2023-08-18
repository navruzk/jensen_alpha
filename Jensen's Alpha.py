# Databricks notebook source
# MAGIC %md
# MAGIC # S&P500 Stocks' Jensen's Alpha 

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Introduction
# MAGIC ##### Stock market is where the stocks are bought and sold, and stock prices regularly fluctuate. Diving deep into the stock price data and modeling the stock returns shows that some of stocks usually show better performance than the others. This notebook answers a very usual question of stock market enthusiasts and practitioners, what type of stocks beat the market?
# MAGIC
# MAGIC Notebook is organized in a research paper style. It presents data set and its source, and the usual model for estimation of the Jensen's Alpha. Next, the main results are presented and the notebook is concluded.
# MAGIC
# MAGIC ##### But first what is the Jensen's Alpha?
# MAGIC
# MAGIC Stock prices are volatilem, however, not a single stock can indiciate how the market as a whole is doing. For that reason we look into all of the stocks together for example the index (S&P500, Nikkei225).
# MAGIC So basically stock index shows the whole market itself. If index price is going up that means all the market is doing good.
# MAGIC
# MAGIC Historically, individual stocks are tested if they follow the market or not. Usually some of the stocks go up with the market or go down when the market goes up. For that reason several mathematical financial models are developed and the most practical one is called Capital Asset Pricing Model (CAPM).
# MAGIC
# MAGIC $$
# MAGIC R_{S} - R_{F} = \beta_{rm} (R_{M} - R_{F})
# MAGIC $$
# MAGIC
# MAGIC Here, RS is a stock return, RF is the risk free rate, and RM is the market return. So expectation is that the stock return will relate to the market return by the parameter beta.
# MAGIC
# MAGIC However, some of stocks perform better than the market and the CAPM model captures that factor by alpha. It is called Jensen's Alpha (Jensen, 1967., https://papers.ssrn.com/sol3/papers.cfm?abstract_id=244153).
# MAGIC
# MAGIC $$
# MAGIC R_{S} - R_{F} = \alpha + \beta_{rm} (R_{M} - R_{F})
# MAGIC $$
# MAGIC
# MAGIC Now, stocks with the larger alpha is assumed to perform better than the market and portfolio engineers always look for that alpha.
# MAGIC
# MAGIC Thus, in this notebook we will look into S&P500 stocks and estimate the alpha. We also look into which industry is showing better alpha and better market and alpha relation. Since greater the alpha greater the chance of stock performing better than the market.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data
# MAGIC
# MAGIC Data is coming from two sources. 
# MAGIC FED's open source is used to get risk free rates and Stooq.com is used to get stock prices.
# MAGIC
# MAGIC But before, all required packages should be installed.

# COMMAND ----------

!pip install datapackage --quiet 

# COMMAND ----------

!pip install pandas_datareader --quiet 

# COMMAND ----------

### import packages 

import datapackage
import pandas as pd
import pandas_datareader as pdr
import statsmodels.api as sm
import numpy as np
np.random.seed(9876789)

### here S&P500 stock names are imported

data_url = "https://datahub.io/core/s-and-p-500-companies/datapackage.json"

package = datapackage.Package(data_url)

resources = package.resources
for resource in resources:
    if resource.tabular:
        data = pd.read_csv(resource.descriptor['path'])


# COMMAND ----------

### prepare risk free rate data

rf_data = pdr.get_data_fred('DGS1MO', start = "2010-01-01",end = "2024-01-01")
rf_data["rf"] = rf_data["DGS1MO"] / 100

### lets take a look at the description of the data
### it looks ok since it is already processed and offered through fed source
### min is 0% as expected and max is 6%
### seems like rf data has std of 1.2%

rf_data.describe()

# COMMAND ----------

### so rf data is from 2010 until 2023
rf_data

# COMMAND ----------

#### uncomment this section if it is the first time running the notebook
#### here all stock data is downloaded from stooq

# stock_data_all = pd.DataFrame(index = pd.date_range("2010-01-01","2024-01-01"))

# for ticker in ticker_list[:]:
    
#     stock_data = pdr.get_data_stooq(ticker)
    
#     if stock_data.shape[0] == 0:
#         continue

#     stock_data[ticker + "_close"] = stock_data["Close"].copy()
    
#     stock_data_all = pd.merge(stock_data_all, stock_data[ticker + "_close"], left_index=True, right_index=True, how = "left")

# COMMAND ----------

#### if it is the second time running the notebook simply use already downloaded data

# file location and type
file_location = "/FileStore/tables/stck_data-2.csv"
file_type = "csv"

# schema info
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# read data
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

# some cleaning and make it pandas df
stock_data_all = df.to_pandas_on_spark()
stock_data_all.set_index("_c0", inplace = True)
stock_data_all.dropna(inplace = True)

stock_data_all = stock_data_all.to_pandas()

# COMMAND ----------

### lets take a look at the description of the data
### we have all the stock prices for most of stocks in S&P500

stock_data_all.describe()

# COMMAND ----------

### our stock price data is in a daily frequency and it is from 2020 April until 2023 August
stock_data_all

# COMMAND ----------

### data preparation
### prepare S&P500 and risk free data

market_data = pdr.get_data_fred('SP500')
market_data["market_return"] = market_data["SP500"].diff() / market_data["SP500"]

rf_data = pdr.get_data_fred('DGS1MO', start = "2010-01-01",end = "2024-01-01")
rf_data["rf"] = rf_data["DGS1MO"] / 100
rf_data["yyyymm"] = rf_data.index.year.astype("str") + rf_data.index.month.astype("str").str.zfill(2)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Modeling

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC First we start by fitting a linear regression. By using Ordinary Least Squares method we estimate parameters. They are estimated parameters so they will have a cap on it.
# MAGIC
# MAGIC $$
# MAGIC R_{i,t} - R_{F} = \hat{\alpha}_{i} + \hat{\beta}_{i} (R_{M}  - R_{F} )+ \epsilon_{i,t}
# MAGIC $$
# MAGIC
# MAGIC Here, i is for stock ticker, and t is for time.
# MAGIC
# MAGIC So now we have market index price data, risk free data and stock prices. We can start modeling from now on.
# MAGIC
# MAGIC But before that lets choose one stock and explain what is the model result and what is the estimated alpha.
# MAGIC

# COMMAND ----------

## lets choose apple stocks
ticker = "AAPL"

## make temporary data
stock_data = stock_data_all[[ticker + "_close"]].copy()

## estimate returns
stock_data["stock_return"] = stock_data[ticker + "_close"].diff() / stock_data[ticker + "_close"]

## merge with market data
data_mr = pd.merge(market_data["market_return"], 
                   stock_data["stock_return"], 
                   left_index=True, 
                   right_index=True)

data_mr["yyyymm"] = data_mr.index.year.astype("str") + data_mr.index.month.astype("str").str.zfill(2)
data_mr["yyyymmdd"] = data_mr.index.copy()

## merge with rf data
data_mr = pd.merge(data_mr[["yyyymmdd","stock_return","market_return","yyyymm"]], 
                   rf_data[["rf","yyyymm"]], 
                   left_on="yyyymm", 
                   right_on="yyyymm")

data_mr["stock_return"] = data_mr["stock_return"] - data_mr["rf"] 
data_mr["market_return"] = data_mr["market_return"] - data_mr["rf"] 

## clean data
data_mr.dropna(inplace=True)

## add constant into the data so model will estimate alpha
data_mr = sm.add_constant(data_mr)

## model fitting
model = sm.OLS(data_mr.stock_return, data_mr[["const","market_return"]])
results = model.fit()


# COMMAND ----------

print(results.summary())

### here R-squared shows that mraket can explain almost 80% (0.789) of the fluctuations in apple stocks
### looking into coef for const (that is our alpha) we see P value is almost zero
### p value shows if the coef is statistically significant
### so in short, apple stocks showing 0.0015 better performance than the market

# COMMAND ----------

### here we repeat the same process for all other stocks 
### estimate the alpha and save them for further use

# COMMAND ----------

ticker_list = data["Symbol"].tolist()

market_pv_list = list()
alpha_pv_list = list()

market_coef_list = list()
alpha_coef_list = list()

symbol_list = list()
    
for ticker in ticker_list[:]:
    
    if ticker + "_close" not in stock_data_all.columns.tolist():
        continue

    stock_data = stock_data_all[[ticker + "_close"]].copy()

    if stock_data.shape[0] == 0:
        continue

    stock_data["stock_return"] = stock_data[ticker + "_close"].diff() / stock_data[ticker + "_close"]

    ## merge with market data
    data_mr = pd.merge(market_data["market_return"], 
                       stock_data["stock_return"], 
                       left_index=True, 
                       right_index=True)

    data_mr["yyyymm"] = data_mr.index.year.astype("str") + data_mr.index.month.astype("str").str.zfill(2)
    data_mr["yyyymmdd"] = data_mr.index.copy()

    ## merge with rf data
    data_mr = pd.merge(data_mr[["yyyymmdd","stock_return","market_return","yyyymm"]], 
                       rf_data[["rf","yyyymm"]], 
                       left_on="yyyymm", 
                       right_on="yyyymm")


    data_mr["stock_return"] = data_mr["stock_return"] - data_mr["rf"] 
    data_mr["market_return"] = data_mr["market_return"] - data_mr["rf"] 

    ## clean data
    data_mr.dropna(inplace=True)

    ## add constant into the data so model will estimate alpha
    data_mr = sm.add_constant(data_mr)

    ## model fitting
    model = sm.OLS(data_mr.stock_return, data_mr[["const","market_return"]])
    results = model.fit()

    ## save parameters
    market_pv_list.append(results.pvalues[0]) ##market pvalue
    alpha_pv_list.append(results.pvalues[1]) ##alpha pvalue

    ## save pvalues
    market_coef_list.append(results.params[0]) ##market pvalue
    alpha_coef_list.append(results.params[1]) ##alpha pvalue

    ## save tickers
    symbol_list.append(ticker)


# COMMAND ----------

### here save all data into dataframe
final_data = pd.DataFrame()
final_data["symbol"] = symbol_list
final_data["alpha_coef"] = alpha_coef_list
final_data["alpha_pv"] = alpha_pv_list
final_data["market_coef"] = market_coef_list
final_data["marke_pv"] = market_pv_list

### lets focus on only significant data
final_data_significant = final_data[final_data["alpha_pv"] <= 0.1]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results

# COMMAND ----------

# MAGIC %md
# MAGIC ### Estimated Jensen's alpha

# COMMAND ----------

## lets check alphas
## it seems it goes up to 1.35 and goes down to 1

final_data.describe()

# COMMAND ----------

### by looking into all alpha in below graph we see interesting result

### just for reminder the meaning of alpha is, higher the alpha from 1 better the performance and alphas lower than 1 are the cases when stock is doing poorly than the market

### thus we can conclude that stocks forming two groups
### first, over 1 alpha. they do better than the market return
### second is the lower than 1 alpha. they usually doing worse than the market

### next we check market coef

# COMMAND ----------

display(final_data_significant)

# COMMAND ----------

### stocks's market relation is captured by the beta factor
### interestingly some of stocks are showing strong relation to the overall market while other stocks are showing less then 0 relation

### but most imporantly as we expected stocks vary depending on how they relate to the market
### we can see clearly two groups are formed, one positive, higher than 0 market coef, and the other group of stocks have negative market coef.
### stocks with positive coef will go up when the market goes up and stocks with negative coef will go down when market goes up


# COMMAND ----------

display(final_data_significant)

# COMMAND ----------

### thus so far our main results
### alphas form two groups
### market relation also forms two groups

### thus we need to check by industry wise averages
### different type of industry should have different alpha


# COMMAND ----------

### lets look into data by industry wise
### we look into average alphas for each industry

data_final_mr = pd.merge(final_data, data, left_on="symbol", right_on="Symbol")
data_final_mr_gr = data_final_mr.groupby("Sector").mean()
data_final_mr_gr.reset_index(inplace=True)

# COMMAND ----------

### so on average energy, information technology, financials and consumer discretionary stocks are showing better alpha
display(data_final_mr_gr)

# COMMAND ----------

### lets check the alpha and market relation in one graph
### this combination should find us the best performing sector
### diving deep into that shows that the energy sector shows the best combinations
### meaining that it has stronger relation with the market itself and also higher alpha

display(data_final_mr_gr)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Industry return
# MAGIC
# MAGIC Since we found based on alpha and market relation the high performing industry, now it is time to actually dive deeper into industry returns.
# MAGIC As we saw earlier, energy sector shows the best combination of alpha and beta relation. So we assume energy sector return should be the best.

# COMMAND ----------

### construct portfolio of those high performing energy sectors

# COMMAND ----------

stock_data_all_v1 = pd.DataFrame((stock_data_all.loc[stock_data_all.index[-1]] - stock_data_all.loc[stock_data_all.index[0]]) / stock_data_all.loc[stock_data_all.index[0]], columns=["return"])
stock_data_all_v1.index = [a.split("_")[0] for a in stock_data_all_v1.index.tolist()]
stock_data_all_v1.reset_index(inplace = True)
stock_data_all_v1.rename(columns={"index":"symbol"}, inplace=True)

# COMMAND ----------

data_mr_v1 = pd.merge(data, stock_data_all_v1, left_on="Symbol", right_on="symbol")
data_mr_v1 = data_mr_v1.groupby("Sector").mean()
data_mr_v1.reset_index(inplace=True)

# COMMAND ----------

data_mr_v1.sort_values("return")

# COMMAND ----------

### based on results we see that energy sector indeed is showing the best return
### also materials and consumer discretionary also have high returns

# COMMAND ----------

display(data_mr_v1.sort_values("return"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC
# MAGIC In this notebook we have seen how to estimate Jensen's alpha, a simple indicator that shows which stocks are doing better than the market.
# MAGIC
# MAGIC By looking into almost 500 stocks we estimated the alphas and checked the best stock sector based on the alpha and beta relation.
# MAGIC
# MAGIC Looking into that relation we found that indeed best sector that performed well for the past years is the energy sector stocks.
# MAGIC
