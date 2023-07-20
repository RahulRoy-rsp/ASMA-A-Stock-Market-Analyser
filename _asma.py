from email import header
from random import random
import requests
import urllib.request
from datetime import date, timedelta
import datetime
from bs4 import BeautifulSoup
from prettytable import PrettyTable
import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

from matplotlib import style
import yfinance as yf
style.use('ggplot')
import plotly.express as px
import plotly.graph_objects as go
import plotly.graph_objs as go
import altair as alt
import snscrape
import snscrape.modules.twitter as sntwitter
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from cleantext import clean
import seaborn as sns
from textblob import TextBlob
from nltk.tokenize import word_tokenize
import nltk
from fbprophet import Prophet
import fbprophet
import tweepy
from keys import access_token, access_token_secret, api_key, api_key_secret

# nltk.download('stopwords')
# nltk.download('stopwords')
# nltk.download('punkt')

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from wordcloud import WordCloud
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import mean_squared_error, mean_absolute_error
# download the vader lexicon
# nltk.download('vader_lexicon')



st.set_page_config(page_title="ASMA: A Stock Market Analyser", page_icon="star2")

#   Local URL: http://localhost:8501
#   Network URL: http://192.168.0.103:8501



current_time = datetime.datetime.now().time()
greet = ""
if current_time.hour < 12:
    greet = "Good morning"
elif current_time.hour < 18:
    greet = "Good afternoon"
else:
    greet = "Good evening"



st.markdown(
    f"""
    <h3 style='color: #300030; font-family: Georgia;'>
        ASMA
    </h3>
    """
    , unsafe_allow_html=True
)

st.write(
    f"""
    <h5 style='color: #028f76; font-family: Georgia;'>
    A Stock Market Analyser
    </h5>
    """
    , unsafe_allow_html=True
)

   
st.write(
    f"""
    <h1 style='color: #005bc5; font-family: Georgia;'>
        {greet} Investor!
    </h1>
    """
    , unsafe_allow_html=True
)


df = pd.read_csv("EQUITY_L.csv")

    # reading company symbol
sym = df.loc[:, "SYMBOL"]

    # reading company name
company = df.loc[:, "NAME OF COMPANY"]

    # combining symbol with company name as dictionary
sym_company_dict = {sym[i]: company[i] for i in range(len(sym))}

    # extracting symbol and company as one in form of list
sym_company_list = []

for key, value in sym_company_dict.items():
    # print(key+'- '+value)
    sym_company_list.append(key + ' ' + '(' + value + ')')



st.sidebar.write(
    f"""
    <h3 style='color: white; font-family: Georgia;'>
        ASMA: A Stock Market Analyser
    </h3>
    """
    , unsafe_allow_html=True
)

st.sidebar.write(
    f"""
    <div style='color: #242B2E; font-family: Roboto;'>
        Select a Stock
    </div>
    """
    , unsafe_allow_html=True
)

sel = st.sidebar.selectbox("", sym_company_list, index=1651)

    # splitting symbol and company name
sel_split = sel.split(' ', 1)
sel_sym = sel_split[0]
sel_company = sel_split[1]
    # removing brackets from company name
sel_company = str(sel_company).replace('(', '')
sel_company = sel_company.replace(')', '')

st.sidebar.write(
    f"""
    <h4 style='color: #242B2E; font-family: Roboto;'>
        Select the number of days for prediction
    </h4>
    """
    , unsafe_allow_html=True
)

noOfDays = st.sidebar.slider(" ", 20, 1300, 111)
    # getting data button
proceed_button = st.sidebar.button("Search", key="proceed")

st.write(f"You have currently selected {sel}. \nPlease select a stock from the sidebar and enter Search to view the analysis")

st.markdown(f"""
    ### You are currently looking for {sel_company}
    """)


st.sidebar.markdown(
    f"""
    <h3 style='color: yellow; font-family: sans serif;'>
        Stock Symbol: {sel_sym}
    </h3>
    <h3 style='color: white; font-family: sans serif;'>
        Predicting for: {noOfDays} days
    </h3>    
    """
    , unsafe_allow_html=True
)

if proceed_button:
    table_ = PrettyTable(["Company Name",
                              "Stock Ticker"
                              ])

    table_.add_row([sel_company, sel_sym])
    # print(table_)
    st.write(table_)

    linkForReports = "https://www.screener.in/"

    companyForReports = sel_sym.lower()
    new_linkForReports = linkForReports + f"company/{companyForReports}/consolidated/"

    page = requests.get(new_linkForReports)
    soup = BeautifulSoup(page.content, 'html.parser')

    # print("Company Name: ")
    for span in soup.find('div', class_='flex-row').findAll('h1'):
        # print(span.text)
        st.subheader(span.text)
    

    # print("About: ")
    st.subheader("About: ")
    for span in soup.find('div', class_='about').findAll('p'):
        # print(span.text)
        # st.write(span.text)
        about_ = span.text
        about_ = re.sub(r"\[\d+\]", "", about_)
        st.write(
                f"""
                <p style='color: #242B2E; font-family: Roboto Slab;'>           
                    {about_}
                </p>
                """
                , unsafe_allow_html=True
                )
    

    linkForSales = "https://www.screener.in/"
    new_linkForSales = linkForSales + f"company/{sel_sym.lower()}/consolidated/"

    response = requests.get(new_linkForSales)

    # Use BeautifulSoup to parse the HTML content
    soupforTables = BeautifulSoup(response.content, 'html.parser')

    # Find the table(s) you want to extract
    tables_ = soupforTables.find_all('table')[0]

    # Print out the contents of the table(s)
    # for table in tables_:
    try:
        quarterlyDF = pd.read_html(str(tables_))[0]
        quarterlyDF = quarterlyDF.rename(columns={'Unnamed: 0': 'RSP'})
        quarterlyDF['RSP'] = quarterlyDF['RSP'].str.replace('+', '')
        quarterlyDF = quarterlyDF.iloc[:-1, :]
        st.subheader("Quarterly Results")
        st.write('Figures mentioned are in Crores(Rs)')
        st.dataframe(quarterlyDF)
        # st.table(quarterlyDF) 
        st.subheader("Quarterly Sales")
        datesSales = quarterlyDF.columns[1:]
        sales_ = quarterlyDF.iloc[0, 1:]
        datesVsSales = pd.DataFrame(columns=['Dates', 'Sales'])
        datesVsSales['Sales'] = sales_
        datesVsSales['Dates'] = datesSales
        
        chartTab_1, chartTab_2, chartTab_3 = st.tabs(['Bar Chart', 'Line Chart', 'Area Chart'])
        with chartTab_1:
            st.write(f"Quarterly sales in form of Bar Chart for {sel_company}")
            salesFig_1 = px.bar(datesVsSales, x='Dates', y='Sales')
            salesFig_1.update_layout(plot_bgcolor='#cad1c3')
            st.plotly_chart(salesFig_1)
        
        with chartTab_2:
            st.write(f"Quarterly sales in form of Line Chart for {sel_company}")
            salesFig_2 = px.line(datesVsSales, x='Dates', y='Sales', line_shape='spline', color_discrete_sequence=['red'])
            salesFig_2.update_layout(plot_bgcolor='#abe4ff')
            st.plotly_chart(salesFig_2)

        with chartTab_3:
            st.write(f"Quarterly sales in form of Area Chart for {sel_company}")
            salesFig_3 = px.area(datesVsSales, x='Dates', y='Sales', color_discrete_sequence=['#f0a818', '#fe9600'])
            salesFig_3.update_traces(mode='lines', hovertemplate=None)
            salesFig_3.update_xaxes(showgrid=False, zeroline=True)
            salesFig_3.update_yaxes(showgrid=False, zeroline=True)
            st.plotly_chart(salesFig_3)    


    except KeyError:
        st.error("Could not find the quarterly results for this stock", icon="🚨")      

    try:
        st.subheader("Profit & Loss")
        tables_PL = soupforTables.find_all('table')[1]
        PLDF = pd.read_html(str(tables_PL))[0]
        PLDF = PLDF.rename(columns={'Unnamed: 0': 'RSP'})
        PLDF['RSP'] = PLDF['RSP'].str.replace('+', '')
        st.table(PLDF) 

        st.subheader("Operating Profits/Losses:")
        yearlyDates = PLDF.columns[1:]
        pl_ = PLDF.iloc[2, 1:].values
        datesVsPL = pd.DataFrame(columns=['Dates', 'Profit'])
        datesVsPL['Dates'] = yearlyDates
        datesVsPL['Profit'] = pl_
        int_list = [float(x) for x in datesVsPL['Profit'].tolist()]
        datesVsPL['Values'] = int_list

        chartTab_1, chartTab_2, chartTab_3 = st.tabs(['Bar Chart', 'Line Chart', 'Area Chart'])
        with chartTab_1:
 
            fig_1 = px.bar(datesVsPL, x='Dates', y='Values', color=datesVsPL['Values'] > 0,
                color_discrete_map={False: '#f44336', True: '#64dd17'})

            fig_1.update_layout(title=f'{sel_company}: Profits/Loss', xaxis_title='Date', yaxis_title='Profit', plot_bgcolor='#f1f2f6')
            fig_1.for_each_trace(lambda trace: trace.update(name='Profit' if trace.name == 'True' else 'Loss'))
            st.plotly_chart(fig_1)

        
        with chartTab_2:
            # st.write("Line Chart")
            fig_2 = px.line(datesVsPL, x='Dates', y='Values', color=datesVsPL['Values'] > 0,
                        color_discrete_map={False: '#f44336', True: '#64dd17'})

            fig_2.update_layout(title=f'{sel_company}: Profits/Loss', xaxis_title='Date', yaxis_title='Profit', plot_bgcolor='white')
            fig_2.for_each_trace(lambda trace: trace.update(name='Profit' if trace.name == 'True' else 'Loss'))

            st.plotly_chart(fig_2)


        with chartTab_3:
            # st.write("Area Chart")
            fig_3 = px.area(datesVsPL, x='Dates', y='Values', color=datesVsPL['Values'] > 0,
                        color_discrete_map={False: '#f44336', True: '#64dd17'})

            fig_3.update_layout(title=f'{sel_company}: Profits/Loss', xaxis_title='Date', yaxis_title='Profit', plot_bgcolor='white')
            fig_3.for_each_trace(lambda trace: trace.update(name='Profit' if trace.name == 'True' else 'Loss'))
            st.plotly_chart(fig_3)
   



    except KeyError:
        st.error("Could not find the Profit/Loss for this stock", icon="🚨")      


    try:
        tables_BS = soupforTables.find_all('table')[6]
        BSDF = pd.read_html(str(tables_BS))[0]
        BSDF = BSDF.rename(columns={'Unnamed: 0': 'RSP'})
        BSDF['RSP'] = BSDF['RSP'].str.replace('+', '')
        st.subheader("Balance Sheet")
        st.table(BSDF)

    except KeyError:
        st.error("Could not find the Balance Sheet for this stock", icon="🚨")      

    try:
        tables_SH = soupforTables.find_all('table')[9]
        SHDF = pd.read_html(str(tables_SH))[0]
        SHDF = SHDF.rename(columns={'Unnamed: 0': 'RSP'})
        SHDF['RSP'] = SHDF['RSP'].str.replace('+', '')
        st.subheader("Shareholders")
        st.table(SHDF)

    except KeyError:
        st.error("Could not find the Shareholders for this stock.", icon="🚨")      
    except IndexError:
        st.error("Shareholding pattern is currently not available for this company.", icon="🚨")


    sales_growth, profit_growth = st.columns(2)
    with sales_growth:
        st.write(
                f"""
                <h4 style='color: #242B2E; font-family: Merriweather;'>
                    Compound Sales Growth
                </h4>
                """
                , unsafe_allow_html=True
            )

        tables_SG = soupforTables.find_all('table')[2]
        SGDF = pd.read_html(str(tables_SG))[0]
        SGDF_ = SGDF.to_string(index=False, header=False)
        st.text(SGDF_)

    with profit_growth:
        st.write(
                f"""
                <h4 style='color: #242B2E; font-family: Merriweather;'>
                    Compound Profit Growth
                </h4>
                """
                , unsafe_allow_html=True
            )
        tables_PG = soupforTables.find_all('table')[3]
        PGDF = pd.read_html(str(tables_PG))[0]
        PGDF_ = PGDF.to_string(index=False, header=False)
        st.text(PGDF_)

    cagr, roe = st.columns(2)
    with cagr:
            st.write(
                    f"""
                    <h4 style='color: #242B2E; font-family: Merriweather;'>
                        Compound Annual Growth Rate
                    </h4>
                    """
                    , unsafe_allow_html=True
                )

            tables_cagr = soupforTables.find_all('table')[3]
            CAGRDF = pd.read_html(str(tables_cagr))[0]
            CAGRDF_ = CAGRDF.to_string(index=False, header=False)
            st.text(CAGRDF_)

    with roe:
        st.write(
                f"""
                <h4 style='color: #242B2E; font-family: Merriweather;'>
                    Return on Equity
                </h4>
                """
                , unsafe_allow_html=True
            )

        tables_roe = soupforTables.find_all('table')[4]
        ROEDF = pd.read_html(str(tables_roe))[0]
        ROEDF_ = ROEDF.to_string(index=False, header=False)
        st.text(ROEDF_)


    st.write("Check below the defintion about the above terms")
    tabInfo_1, tabInfo_2, tabInfo_3, tabInfo_4 = st.tabs(['Compound Sales Growth',
                                                        'Compound Profit Growth',
                                                        "CAGR",
                                                        "Return on Equity"]) 

    with tabInfo_1:
        st.write("A compound annual growth rate is a metric that smooths annual gains in revenue, returns, customers, and so on over a specified number of years as if the growth had happened steadily each year over that period.")
    with tabInfo_2:
        st.write("""Profit Growth means the compound annual percentage growth in profit of the Company or a business unit of the Company, as the case may be, for the Performance Period.""")
    with tabInfo_3:
        st.write("Compound annual growth rate, or CAGR, is the mean annual growth rate of an investment over a specified period of time longer than one year. It represents one of the most accurate ways to calculate and determine returns for individual assets, investment portfolios, and anything that can rise or fall in value over time. CAGR is a term used when investment advisors tout their market savvy and funds promote their returns.")
    with tabInfo_4:
        st.write(" Return on equity (ROE) is a measure of financial performance calculated by dividing net income by shareholders' equity. Because shareholders' equity is equal to a company’s assets minus its debt, ROE is considered the return on net assets. ROE is considered a gauge of a corporation's profitability and how efficient it is in generating profits. The higher the ROE, the more efficient a company's management is at generating income and growth from its equity financing.")

    st.header("Announcements, Quarterly Results and Reports: ")
    with st.expander('Click to expand!'):
        tab1, tab2, tab3 = st.tabs(['Announcements', 'Quarterly Results', 'Reports'])

        with tab1:
            st.markdown("From latest to oldest :arrow_double_down:")
            ann_links = []
            for link in soup.find_all('a',
                                        attrs={'href': re.compile("^https://www.bseindia.com/xml-data/corpfiling/.*\.pdf$")}):
                # print(link.get('href'))
                ann_links.append(link.get('href'))

            for link in soup.find_all('a',
                                        attrs={'href': re.compile("^https://archives.nseindia.com/corporate/.*\.pdf$")}):
                ann_links.append(link.get('href'))

            for i in range(len(ann_links)):
                st.write(ann_links[i])

        with tab2:
            updated_links = []
            _links = []
            for link in soup.find_all('a',
                                      attrs={'href': re.compile("^/company/source")}):
                _links.append(link.get('href'))
 
            updated_links = ["https://www.screener.in" + x for x in _links]
            updated_links = updated_links[::-1]

            if len(updated_links) == 0:
                st.error("Quarterly Results not available.", icon="🚨")      
            else:
                st.markdown("From latest to oldest :arrow_double_down:")
                for i in range(len(updated_links)):
                    st.write(updated_links[i])

        with tab3:
            st.markdown("From latest to oldest :arrow_double_down:")
            reportLinks = []
            for link in soup.find_all('a',
                                      attrs={'href': re.compile(
                                          "^https://www.bseindia.com/bseplus/AnnualReport/.*\.pdf$")}):
                reportLinks.append(link.get('href'))
            for link in soup.find_all('a',
                                      attrs={'href': re.compile(
                                          "^https://archives.nseindia.com/annual_reports/.*\.zip$")}):
                reportLinks.append(link.get('href'))

            for i in range(len(reportLinks)):
                st.write(reportLinks[i])

    st.header("Sentimental Analysis ")


consumer_key = api_key
    consumer_secret = api_key_secret
    access_token = access_token
    access_token_secret = access_token_secret

    # Authenticate with the Twitter API
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)

    # Define search query
    favs = 5
    com_ = sel_company.replace(" ", "")
    one_ = f"#{sel_sym.upper()}"
    two_ = f"#{sel_sym.lower()}"
    three_ = com_.upper()
    four_ = com_.lower()
    five_ = f"{sel_sym.upper()}"
    six_ = f"{sel_sym.lower()}"


    # search_query = f"{one_} OR {two_} OR {three_} OR {four_} OR {five_} OR {six_} min_faves:{favs} lang:en"
    # search_query = f"{one_} OR {two_} OR {three_} OR {four_} min_faves:{favs} lang:en"
    search_query = f"{one_} OR {two_} OR {five_} OR {six_} min_faves:{favs} lang:en"
    

    # Create empty list to store tweets
    tweetsList = []

    # Iterate through search results and append to list
    for tweet in tweepy.Cursor(api.search_tweets, q=search_query, lang='en', tweet_mode='extended').items(500):
        if tweet.favorite_count >= 5:
            tweetsList.append([tweet.created_at, tweet.id_str, tweet.user.screen_name, tweet.full_text, tweet.favorite_count, tweet.retweet_count])

    # Convert list to DataFrame
    tweetsDF = pd.DataFrame(tweetsList, columns=['Datetime', 'Tweet_Id', 'UserName', 'Tweets', 'Likes', 'Retweets'])

    # st.write(tweetsDF.shape)
    # print(tweetsDF.shape)
    
    if tweetsDF.shape[0]< 5:
        st.error(f"Tweets are not available for {sel_company}")
    else:
        tweetsDF['Date'] = pd.to_datetime(tweetsDF['Datetime']).dt.date
        tweetsDF['Time'] = pd.to_datetime(tweetsDF['Datetime']).dt.time
        tweetsDF = tweetsDF.reindex(columns=['Date', 'Time', 'Tweets', 'UserName', 'Likes', 'Retweets', 'Tweet_Id', 'Datetime'])
        # st.write(tweetsDF)

        senti_tab1, senti_tab2 = st.tabs(["Charts", "More"])

        with senti_tab1:
            st.write(
                    f"""
                    <h4 style='color: #512da8; font-family: Georgia;'>
                    Sentimental Analysis of {sel_company} using Vader-Sia Model
                    </h4>
                    """
                    , unsafe_allow_html=True
                )
            stop_words = set(stopwords.words('english'))
            # initialize the sentiment analyzer
            sia = SentimentIntensityAnalyzer()
            
            def preprocess_tweet_text(tweet):
            # Remove URLs, mentions, and hashtags
                tweet = re.sub(r"http\S+|www\S+|https\S+|\@\w+|\#\w+", "", tweet)
            
                # Convert text to lowercase
                tweet = tweet.lower()
            
                # Remove stop words and punctuation
                tweet_tokens = word_tokenize(tweet)
                filtered_words = [word for word in tweet_tokens if word not in stop_words and word.isalpha()]
                tweet = ' '.join(filtered_words)
            
                return tweet
            
        
            tweetsDF['Method3_CT'] = tweetsDF['Tweets'].apply(preprocess_tweet_text)

            def get_sentiment_scores(text):
                scores = sia.polarity_scores(text)
                return pd.Series(scores)
            
            tweetsDF[['neg_4', 'neu_4', 'pos_4', 'compound_4']] = tweetsDF['Method3_CT'].apply(get_sentiment_scores)
            
            bar_4, pie_4 = st.columns(2)
            
            labels = ['Negative', 'Neutral', 'Positive']
            sizes = [tweetsDF['neg_4'].mean(), tweetsDF['neu_4'].mean(), tweetsDF['pos_4'].mean()]
            colors = ['red', 'yellow', 'green']
            explode = (0.1, 0.1, 0.1)

            fig, ax = plt.subplots()
            sns.barplot(x=labels, y=sizes, palette=colors)
            ax.set_title('Sentiment Analysis Results')
            ax.set_xlabel('Sentiment')
            ax.set_ylabel('Average Score')
            # plt.show()
            bar_4.pyplot(plt)

            fig, ax = plt.subplots()
            wp = {'linewidth': 2, 'edgecolor': "black"}
            ax.pie(sizes, labels=labels, shadow=True, colors=colors, autopct='%1.1f%%', startangle=90, wedgeprops=wp, explode=explode)
            ax.axis('equal')
            plt.title('Sentiment Analysis Results')
            # plt.show()
            pie_4.pyplot(plt)
        

        with senti_tab2:
            st.write(
                    f"""
                    <h5 style='color: black; font-family: Georgia;'>
                    Most Liked Tweet
                    </h5>
                    """
                    , unsafe_allow_html=True
                )
            
            mostLikedDF = tweetsDF[tweetsDF['Likes'] == tweetsDF['Likes'].max()]
            # st.write(mostLikedDF)
            mostLikedTweet = mostLikedDF['Tweets'].values[0]
            tweetedBy = mostLikedDF['UserName'].values[0]
            at1 = mostLikedDF['Date'].values[0]
            on1 = mostLikedDF['Time'].values[0]

            mostLiked = mostLikedDF['Likes'].values[0]
            # st.write(mostLikedTweet)
            st.write(
                    f"""
                    <h6 style='color: white; font-family: Georgia; background-color: black;'>
                    {mostLikedTweet}
                    </h6>
                    <p> Tweeted by {tweetedBy} on {at1}, {on1}.
                    """
                    , unsafe_allow_html=True
                )
            st.write(
                    f"""
                    <p style='color: black; font-family: Georgia;'>
                    Like Count: {mostLiked}
                    </p>
                    """
                    , unsafe_allow_html=True
                )

            st.write(
                    f"""
                    <h5 style='color: black; font-family: Georgia;'>
                    Most Retweeted Tweet
                    </h5>
                    """
                    , unsafe_allow_html=True
                )
            
            mostRetweetedDF = tweetsDF[tweetsDF['Retweets'] == tweetsDF['Retweets'].max()]
            mostRetweetedTweet = mostRetweetedDF['Tweets'].values[0]
            mostRetweeted = mostRetweetedDF['Retweets'].values[0]
            tweetedBy = mostRetweetedDF['UserName'].values[0]
            at2 = mostRetweetedDF['Date'].values[0]
            on2 = mostRetweetedDF['Time'].values[0]
            st.write(
                    f"""
                    <h6 style='color: white; font-family: Georgia; background-color: black;'>
                    {mostRetweetedTweet}
                    </h6>
                    <p> Tweeted by {tweetedBy} on {at2}, {on2}.
                    """
                    , unsafe_allow_html=True
                )
            st.write(
                    f"""
                    <p style='color: black; font-family: Georgia;'>
                    Retweet Count: {mostRetweeted}
                    </p>
                    """
                    , unsafe_allow_html=True
                )

            st.write(
                    f"""
                    <h6 style='color: white; font-family: Georgia; background-color: black;'>
                    Word Cloud of the tweets
                    </h6>
                    """
                    , unsafe_allow_html=True
                )
            tweets_text = " ".join(tweet for tweet in tweetsDF.Method3_CT)
            wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='tab10').generate(tweets_text)

            # Display the generated image using matplotlib
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            # plt.show()
            st.pyplot(plt)
   
  

    st.header("Prediction Analysis ")
    tabPred_1, tabPred_2, tabPred_3 = st.tabs(['Model', 'Chart', 'More'])

    symbol = sel_sym+'.ns'
    start_date = date.today() - timedelta(days=365*13)
  
    from datetime import datetime

    try:
        end_date = datetime.today().strftime("%Y-%m-%d")
    except AttributeError:
        end_date = (datetime.today() - timedelta(days=3)).strftime("%Y-%m-%d")



    df = yf.download(symbol, start=start_date, end=end_date)

    # Prepare the data for Prophet
    df = df.reset_index()
    df = df.rename(columns={'Date': 'ds', 'Close': 'y'})


    with tabPred_1:
        st.write(
                    f"""
                    <h4 style='color: #512da8; font-family: Georgia;'>
                    Predictive Analysis of {sel_company} using Prophet Model
                    </h4>
                    <p> Predicting for {noOfDays} days </p>
                    """
                    , unsafe_allow_html=True
                ) 


        model = Prophet(daily_seasonality=True)
        model.fit(df)
        future_dates = model.make_future_dataframe(periods=noOfDays)
        forecast_ = model.predict(future_dates)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], name='Historical Prices', line=dict(color='blue', width=2)))
        fig.add_trace(go.Scatter(x=forecast_['ds'], y=forecast_['yhat'], name='Predicted Prices', line=dict(color='red', width=2)))
        fig.update_layout(title='Historical and Predicted Prices for {}'.format(sel_company), xaxis_title='Date', yaxis_title='Price')
        
        st.plotly_chart(fig)

    with tabPred_2:
        from datetime import datetime
        # st.write('Comparison')
        
        # st.write(forecast_.tail(100))

        last_row = forecast_.tail(1)
        last_ds = last_row['ds'].values[0]
        last_ds = np.datetime_as_string(last_ds, unit='D')
        last_yhat = round(last_row['yhat'].values[0], 2)

        # st.write(df.tail(5))
        # st.write(end_date)

        try:
            # end_date = end_date 
            match = df[df['ds'] == end_date + " 00:00:00"]
            y_value = match['y'].values[0]
        except IndexError:
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
            end_date = end_date - timedelta(days=3)
            end_date = end_date.strftime('%Y-%m-%d')
            # end_date = end_date 
            match = df[df['ds'] == end_date + " 00:00:00"]
            y_value = round(match['y'].values[0], 2)
    
  


        recent_value = y_value
        predicted_value = last_yhat
        barDF_2 = pd.DataFrame({'price': [recent_value, predicted_value], 'type': ['Last Closing', 'Predicted']})

        chart = alt.Chart(barDF_2).mark_bar().encode(
            x='type',
            y='price',
            color=alt.Color('type', scale=alt.Scale(range=['#1f77b4', '#ff7f0e']))
        ).properties(
            width=400,
            height=400
        )

        st.altair_chart(chart)

        bool = "increase"
        percentage2 = round((predicted_value - recent_value) / recent_value * 100, 2)
        if percentage2 < 0:
            bool = "decrease"
        st.markdown(
                f"""
                <h5 style='color: #303f9f; font-family: Georgia;'>
                    According to our Model the stock price for {sel_company} will {bool} by {percentage2}%
                     which means {round(predicted_value - recent_value, 2)}Rs after {noOfDays} days
                </h5>
                """
                , unsafe_allow_html=True
            )
        



    with tabPred_3:

        actual_ = df['y'].values
        predicted_ = forecast_['yhat'].values
        dates_ = df['ds'].values

        actualLen = len(actual_)
        predictedLen = len(predicted_)

        #365
        last_365th_row = df.iloc[actualLen - 365, :]
        # act
        ds_365 = last_365th_row['ds'].strftime('%Y-%m-%d')
        y_365 = round(last_365th_row['y'], 2)
        # st.write(ds_365, y_365)
        # pred
        last_365th_rowP = forecast_.iloc[actualLen - 365, :]
        dsp_365 = last_365th_rowP['ds'].strftime('%Y-%m-%d')
        yhat_365 = round(last_365th_rowP['yhat'], 2)
        # st.write(dsp_365, yhat_365)

        #700
        last_700th_row = df.iloc[actualLen - 700, :]
        # act
        ds_700 = last_700th_row['ds'].strftime('%Y-%m-%d')
        y_700 = round(last_700th_row['y'], 2)
        # st.write(ds_700, y_700)
        # pred
        last_700th_rowP = forecast_.iloc[actualLen - 700, :]
        dsp_700 = last_700th_rowP['ds'].strftime('%Y-%m-%d')
        yhat_700 = round(last_700th_rowP['yhat'], 2)
        # st.write(dsp_700, yhat_700)

        #1000
        last_1000th_row = df.iloc[actualLen - 1000, :]
        # act
        ds_1000 = last_1000th_row['ds'].strftime('%Y-%m-%d')
        y_1000 = round(last_1000th_row['y'], 2)
        # st.write(ds_1000, y_1000)
        # pred
        last_1000th_rowP = forecast_.iloc[actualLen - 1000, :]
        dsp_1000 = last_1000th_rowP['ds'].strftime('%Y-%m-%d')
        yhat_1000 = round(last_1000th_rowP['yhat'], 2)
        # st.write(dsp_1000, yhat_1000)

        finalDF = pd.DataFrame({
            'dates': [ds_365, ds_700, ds_1000],
            'actual': [y_365, y_700, y_1000],
            'predicted': [yhat_365, yhat_700, yhat_1000]
        })
        st.write(
                    f"""
                    <h5 style='color: #512da8; font-family: Georgia;'>
                    Some random numbers of {sel_company} past three years
                    </h5>
                    """
                    , unsafe_allow_html=True
                ) 

        plt.figure(figsize=(10, 6))
        plt.plot(finalDF['dates'], finalDF['actual'], color='blue', label='Actual')
        plt.plot(finalDF['dates'], finalDF['predicted'], color='red', label='Predicted')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title('Actual vs Predicted Values')
        plt.legend()
        st.pyplot(plt)


