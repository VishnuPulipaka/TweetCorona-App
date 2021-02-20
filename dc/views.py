from django.shortcuts import render,redirect
from django.http import HttpResponseRedirect
import time
import tweepy
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import pandas as pd
from .forms import location
import pickle
from geopy import Nominatim
from wordcloud import WordCloud
# import matplotlib.pyplot as plt
import re
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
import folium as fol
# nltk.download('wordnet')
# nltk.download('stopwords')
# Create your views here.
from PIL import Image
import numpy as np
from wordcloud import WordCloud,STOPWORDS
import matplotlib as plt

confirmedGlobal=pd.DataFrame()

allt=[]
def getNewData():

	db_tweets = pd.DataFrame(columns = ['username', 'acctdesc', 'location', 'following','followers', 'totaltweets', 'usercreatedts', 'tweetcreatedts',
	                                        'retweetcount', 'text', 'hashtags'])
	# del(db_tweets)
	db_tweets.drop(axis=1,index=db_tweets.index,inplace=True)

	# Your Twittter App Credentials
	access_token="your access_token_here"
	access_token_secret="your access_token_secret here"
	consumer_key="your consumer_key here"
	consumer_secret="your consumer_secret here"
	 
	# Calling API
	auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_token, access_token_secret)
	api = tweepy.API(auth)
	 
	# Provide the keyword you want to pull the data e.g. "Python".
	keyword = "covid-19"
	search_words=['covid-19 OR pandemic OR coronavirus OR vaccine']
	# Fetching tweet
	# We will time how long it takes to scrape tweets for each run:
	start_run = time.time()
	numTweets=250
	# Collect tweets using the Cursor object
	# .Cursor() returns an object that you can iterate or loop 
	#  over to access the data collected.
	# Each item in the iterator has various attributes 
	# that you can access to get information about each tweet
	tweets = tweepy.Cursor(api.search, q='covid-19 OR pandemic OR coronavirus OR vaccine',lang="en",
		geocode=locationString(l),exclude="retweets", 
		tweet_mode='extended').items(numTweets)
	# Store these tweets into a python list
	tweet_list = [tweet for tweet in tweets]
	noTweets=0
	 
	# script to re-format the twitter data       
	for tweet in tweet_list:
		username = tweet.user.screen_name
		acctdesc = tweet.user.description
		location = tweet.user.location
		following = tweet.user.friends_count
		followers = tweet.user.followers_count
		totaltweets = tweet.user.statuses_count
		usercreatedts = tweet.user.created_at
		tweetcreatedts = tweet.created_at
		retweetcount = tweet.retweet_count
		hashtags = tweet.entities['hashtags']
		try:
			text = tweet.retweeted_status.full_text
		except AttributeError:
			text = tweet.full_text
		
		#making dataframe row
		ith_tweet = [username, acctdesc, location, following, followers, totaltweets,usercreatedts, tweetcreatedts, retweetcount, text, hashtags]
		#appending to the dataframe
		db_tweets.loc[len(db_tweets)] = ith_tweet 
		noTweets += 1
		allt.append(text)
	end_run = time.time()
	duration_run = round((end_run-start_run)/60, 2)
	# print(db_tweets['usercreatedts'])
	# print(db_tweets['tweetcreatedts'])




	db_tweets['outText']=db_tweets['text']
	# print('no. of tweets scraped for run {} is {}'.format(i + 1, noTweets))
	# print('time take for {} run to complete is {} mins'.format(i+1, duration_run))
	

	# WORD CLOUD
	preprocess = preprocessor_Text(db_tweets)
	allwords = ' '.join([text for text in preprocess['text']])
	if len(allwords)!=0:
		stop=set(stopwords.words("english"))
		wc=WordCloud(background_color="black",max_words=300,stopwords=stop)
		wc.generate(allwords)
		wc.to_file("static/dc/images/WordCloud.png")

		# wc = WordCloud(background_color='white',width=800,height=500,random_state=21,max_font_size=110).generate(allwords)
		# plt.figure(figsize=(10,7))
		# plt.imshow(wc,interpolation="bilinear")
		# plt.axis('off')
		# plt.savefig('static/dc/images/words.png')
	else:
		pass
	

	print(len(db_tweets))
	return db_tweets

def preditionsHere(db_tweets):


	# unpickle stuff here
	vectorizer = pickle.load(open("twdata/tfidf.sav","rb"))
	model=pickle.load(open("twdata/model.sav","rb"))
	
	db_tweets['output']=model.predict(vectorizer.transform(db_tweets['text']))


	counts=pd.DataFrame(db_tweets['output'].value_counts())
	labels=list(counts.index)
	data=list(counts['output'])
	print(counts.columns)
	return labels,data,counts

test=""
def locationView(request):
	global loc
	global l
	global test
	
	geolocater=Nominatim(user_agent='geoapiExercises')
	mapd,totalCount,confirmedList,totalDeaths,deathList,totalRecovered,recoveredList,count_list,recover_list,deaths_list,todayConfirmed,todayRecovered,todayDeaths=mapBuild()
	if request.method=="POST":
		#form=location(request.POST)
		l=None
		loc=request.POST.get('entered_location')
		l=geolocater.geocode(str(loc))
		if l == None:
			test="1"
			return HttpResponseRedirect("/")
		else:
			test=""
			print(locationString(l))
				
			response=redirect('/home')
			return response
	else:
		form = location(auto_id=False)
	return render(request,"locform.html",{"comeOn":test,"form":form,"loc":loc,"l":l,"mapData":mapd,"tc":totalCount,"deaths_table":deathList,"totalDeaths":totalDeaths,"confirmedCases":confirmedList,"totalRecovered":totalRecovered,"recoveredCases":recoveredList,"cases_list":count_list,"recover_list":recover_list,"deaths_list":deaths_list,"todayConfirmed":todayConfirmed,"todayRecovered":todayRecovered,"todayDeaths":todayDeaths})

def locationString(l):
	fl=str(l.latitude)+','+str(l.longitude)+','+'200km'
	return fl

loc=""
l=''
	        
def home(request):
	 new_df = getNewData()
	 if new_df.empty:
	 	 return render(request,'error.html',{})
	 else:
		 labels,data,counts=preditionsHere(new_df)
		 outlist=[]
		 l1 = list(new_df['outText'])
		 l2 = list(new_df['output'])
		 l3=list(new_df['username'])
		 for i in range(len(l1)):
		 	try:
		 		t = {}
		 		t['text'] = l1[i]
		 		t['op'] = l2[i]
		 		t['user']=l3[i]
		 		outlist.append(t)
		 	except:
		 		pass
		 p,N,n,x=lineData(new_df)
		 one=False
		 if len(l)==0:
		 	one=True
		 lat=float(locationString(l).split(",")[0])
		 lon=float(locationString(l).split(",")[1])
		 label=l
		 m=fol.Map(location=[lat,lon],tiles="stamen toner",zoom_start=4)
		 fol.LayerControl().add_to(m)
		 fol.Marker(location=[lat,lon],popup=l).add_to(m)
		 m=m._repr_html_()

		 return render(request,'home.html',{'check':test,'tw':new_df['text'],'lab':labels,'dc':data,'test':counts,'p':p,'N':N,'n':n,'x':x,'myMap':m,'tweetList':outlist})

def mapBuild():
	global confirmedGlobal
	# TOTAL DEATHS CALCULATE
	deathsGlobal=pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv',encoding='utf-8',na_values=None)
	totalDeaths = deathsGlobal.iloc[:,-1].sum()
	todayDeaths = deathsGlobal.iloc[:,-1].sum()-deathsGlobal.iloc[:,-2].sum()
	# TABLE FOR GLOBAL DEATHS
	uniqueCNames=pd.unique(deathsGlobal['Country/Region'])
	deaths = deathsGlobal[list(deathsGlobal.columns[1:2])+list([deathsGlobal.columns[-2]])]
	deaths.columns = ['Country','values']
	deaths = deaths.sort_values(by=['values'],ascending=False)
	maxDeathCountry = deaths['Country'].tolist()
	maxDeathCountryCases = deaths['values'].tolist()
	for_table= {
			"Country":maxDeathCountry,
			"values": maxDeathCountryCases
	}
	disp_table = pd.DataFrame(for_table)
	disp_table.index += 1
	deathList = []
	for i in range(disp_table.shape[0]):
		temp = disp_table.iloc[i]
		deathList.append(dict(temp))
	# TABLE FOR GLOBAL DEATHS END
	# TOTAL DEATHS END

	# CONFIRMED CASES
	confirmedGlobal=pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv',encoding='utf-8',na_values=None)
	totalCount = confirmedGlobal.iloc[:,-1].sum()
	todayConfirmed = confirmedGlobal.iloc[:,-1].sum()-confirmedGlobal.iloc[:,-2].sum()

	# TABLE FOR CONFIRMED CASES
	uniqueCountryName=pd.unique(confirmedGlobal['Country/Region'])
	df2=confirmedGlobal[list(confirmedGlobal.columns[1:2])+list([confirmedGlobal.columns[-2]])]
	df2.columns=['Country','values']
	df2 = df2.sort_values(by=['values'],ascending=False)
	topCountry = df2['Country'].tolist()
	topCountryCases = df2['values'].tolist()
	d = {
		"Country": topCountry,
		"values": topCountryCases
	}
	lb = pd.DataFrame(d)
	lb.index += 1
	confirmedList = []
	for i in range(lb.shape[0]):
		temp1 = lb.iloc[i]
		confirmedList.append(dict(temp1))

	# RECOVERED CASES
	recoveredGlobal=pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv',encoding='utf-8',na_values=None)
	totalRecovered = recoveredGlobal.iloc[:,-1].sum()
	todayRecovered = recoveredGlobal.iloc[:,-1].sum()-recoveredGlobal.iloc[:,-2].sum()
	# TABLE FOR RECOVERED CASES
	uniqueCName = pd.unique(recoveredGlobal['Country/Region'])
	r_df = recoveredGlobal[list(recoveredGlobal.columns[1:2])+list([recoveredGlobal.columns[-2]])]
	r_df.columns=['Country','values']
	r_df = r_df.sort_values(by=['values'],ascending=False)
	topr_Country = r_df['Country'].tolist()
	topr_CountryCases = r_df['values'].tolist()
	r_dict={
		"Country": topr_Country,
		"values": topr_CountryCases
	}
	r_data = pd.DataFrame(r_dict)
	r_data.index += 1
	recoveredList = []
	for i in range(r_data.shape[0]):
		r_temp = r_data.iloc[i]
		recoveredList.append(dict(r_temp))

	#World Map 

	df3=pd.read_json('https://cdn.jsdelivr.net/gh/highcharts/highcharts@v7.0.0/samples/data/world-population-density.json')

	corrections=pd.DataFrame(uniqueCountryName)

	corrections_base=[
    ('Bahamas','Bahamas, The'),
    ('Brunei','Brunei Darussalam'),
    ('Burma','Myanmar'),
    ('Congo (Brazzaville)','Congo, Rep.'),
    ('Congo (Kinshasa)','Congo, Dem. Rep.'),
    ('Czechia','Czech Republic'),
    ('Egypt','Egypt, Arab Rep.'),
    ('Gambia', 'Gambia, The'),
    ('Iran', 'Iran, Islamic Rep.'),
    ('Russia', 'Russian Federation'),
    ('Syria', 'Syrian Arab Republic'),
    ('Venezuela', 'Venezuela, RB'),
    ('Yemen', 'Yemen, Rep.'),
    ('Korea, South','Korea, Rep.'),
    ('Laos','Lao PDR'),
    ('Kyrgyzstan','Kyrgyz Republic'),
    ('North Macedonia','Macedonia, FYR'),
    ('Saint Kitts and Nevis','St. Kitts and Nevis'),
    ( 'Saint Lucia','St. Lucia'),
    ('Saint Vincent and the Grenadines','St. Vincent and the Grenadines'),
    ( 'Slovakia','Slovak Republic'),
    ('Syria','Syrian Arab Republic'),
    ('US', 'United States'),
    ('Venezuela','Venezuela, RB'),
    ('Yemen','Yemen, Rep.')
	]

	for i in range(len(corrections_base)):
   		df3['name'].replace(corrections_base[i][1],corrections_base[i][0],inplace=True)
	uniqueCountryName=list(corrections[0])
	dataForMap=[]
	for i in uniqueCountryName:
	    try:
	        tempdf=df3[df3['name']==i]
	        temp={}
	        temp["code3"]=list(tempdf['code3'].values)[0]
	        #temp["name"]=i
	        temp["z"]=df2[df2['Country']==i]['values'].sum()
	        temp["code"]=list(tempdf['code'].values)[0]
	        dataForMap.append(temp)
	    except:
	        pass
	# LINE GRAPHS
	# lineGraph=pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv',encoding='utf-8',na_values=None)
	confirmedCasesLineGraph =pd.DataFrame(confirmedGlobal.iloc[:,4::29])
	count_list = confirmedCasesLineGraph.sum(axis=0).to_list()
	recoveredCasesLineGraph=pd.DataFrame(recoveredGlobal.iloc[:,4::29])
	recover_list = 	recoveredCasesLineGraph.sum(axis=0).to_list()
	deathCasesLineGraph=pd.DataFrame(deathsGlobal.iloc[:,4::29])
	deaths_list = deathCasesLineGraph.sum(axis=0).to_list()
	return dataForMap,totalCount,confirmedList,totalDeaths,deathList,totalRecovered,recoveredList,count_list,recover_list,deaths_list,todayConfirmed,todayRecovered,todayDeaths


def preprocessor_Text(df):

        def clean(x):
            x=' '.join(re.sub("(@[A-Za-z0-9]+)|([^A-Za-z0-9']+)|(\w+:\/\/\S+)"," ",x).split())
            return x
        
        stop=stopwords.words("english")
        stop.extend(["i'm","I'm"])
        # Removing Stopwords
        df.text=df.text.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
        # Removing Hyperlinks, userIDS
        df.text = df.text.apply(clean)
        # Applying Lemmatization
        wnl1 = WordNetLemmatizer()
        df.text=df.text.apply(lambda x:' '.join([wnl1.lemmatize(word,'v') for word in x.split()])) # v stands for verb
        df.text=df.text.apply(lambda x:' '.join([word.lower() for word in x.split()]))
        return df

def lineData(db_tweets):
	#lineGraph data for daywise sentiments:
	new_f=db_tweets.loc[:,['tweetcreatedts','output']].copy()

	nlis=[str(new_f['tweetcreatedts'][i]).split()[0]  for i in db_tweets.index]
	new_f['dates']=nlis
	srt=dict.fromkeys(nlis).keys()
	itera=list(srt)
	base={}
	for d in itera:
	   temp=new_f['output'][new_f['dates']==d]
	   temp=list(temp)
	   vals=[0]*(len(temp))
	   for i in range(len(temp)):
	           if(temp[i]=='Positive'):
	               vals[i]=1
	           if(temp[i]=='Neutral'):
	               vals[i]=0
	           if(temp[i]=='Negative'):
	               vals[i]=-1
	   row={}
	   row[-1]=vals.count(-1)
	   row[0]=vals.count(0)
	   row[1]=vals.count(1)
	   base[d]=row
	pos=[]
	neg=[]
	neu=[]
	for i in base:
	   pos.append(base[i][1])
	   neg.append(base[i][-1])
	   neu.append(base[i][0])
	return pos,neu,neg,itera