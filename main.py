from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import ssl
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt



finviz_url = "https://finviz.com/quote.ashx?t="
tickers = ["AMZN", "GOOG", "META", "COIN"]
news_tables = {}


for ticker in tickers:
    url = finviz_url + ticker
    # downloads html data from finviz
    req = Request(url=url, headers={"user-agent": "SentimentAnalysis"})
    gcontext = ssl.SSLContext()
    res = urlopen(req, context=gcontext)

    # passing data to BeautifulSoup for parsing
    html = BeautifulSoup(res, "html")

    # we only need the news table
    table = html.find(id="news-table")
    # adding it to the news tables dictionary
    news_tables[ticker] = table


parsed_data = []
for ticker, news_tables in news_tables.items():

    # iterating through all the articles in each news table.
    for row in news_tables.findAll("tr"):
        # we want to collect the title, date and time
        title = row.a.get_text()
        date_info = row.td.get_text()
        date_data = date_info.split()
        # some articles only have a time stamp and others have both date and time
        if len(date_data) == 1:
            time = date_data[0]
        else:
            time = date_data[1]
            date = date_data[0]
        # adding the title and timestamp to a larger array.
        parsed_data.append([ticker, time, date, title])


# turning the nested arrays into a pandas dataframe
df = pd.DataFrame(parsed_data, columns=["ticker", 'time', 'date', 'title'])
# creating a sentiment intensity analyzer
vader = SentimentIntensityAnalyzer()

# adding the polarity score to the dataframe
df['compound_score'] = df['title'].apply(lambda title: vader.polarity_scores((title))['compound'])
df['date'] = pd.to_datetime(df.date).dt.date

# plotting the scores using matplotlib.
# plt.figure(figsize=(10,8))

mean_df = df.groupby(['ticker', 'date']).mean()
mean_df = mean_df.unstack()
mean_df = mean_df.xs('compound_score', axis='columns').transpose()
mean_df.plot(kind='bar')
plt.show()
