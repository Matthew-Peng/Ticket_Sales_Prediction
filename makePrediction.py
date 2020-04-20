import sys
import pandas as pd
import numpy as np
from datetime import date
from datetime import timedelta
import itertools
import lightgbm as lgb
from joblib import dump, load
from sklearn.ensemble import RandomForestRegressor

# config
days = 10 # predicting dates
today = date.today() # predicting from today
start = pd.to_datetime('2017-01-01').date() # trend start date
X_var = ["country_1", "channel_id", "iso_week", "iso_weekend", "trend"]

def create_feature_table(days, today, start):
	# build predicting dates
	dates = [today + timedelta(days=x) for x in range(days)]

	# load combinations of channels and countries
	combs = pd.read_csv("combinations.csv", index_col=0)

	# make feature table
	all_list = [combs["comb"].drop_duplicates().values,
	            dates]
	comb_dates = list(itertools.product(*all_list)) 
	comb_dates = pd.DataFrame(comb_dates, columns=["comb", "date"])
	features = combs.merge(comb_dates, on="comb")

	# create seasonality features
	features['iso_week'] = features['date'].map(lambda x: x.isocalendar()[1])
	features['iso_weekend'] = features['date'].map(lambda x: x.isocalendar()[2])

	# create trend feature
	features["trend"] = features["date"] - start 
	features["trend"] = features["trend"].map(lambda x: x.days)
	return(features)

def lgb_prediction(features):
	# make prediction with lgb
	a = lgb.Booster(model_file = 'Dell_q1_models/lgb_model.txt')
	predict = a.predict(features.values)
	predict[predict<0] = 0
	return(np.around(predict))

def rf_prediction(features):
	# make prediction with rf
	r = load('Dell_q1_models/rf_model.joblib')
	predict = r.predict(features.values)
	predict[predict<0] = 0
	return(np.around(predict))

if __name__ == '__main__':
	print("making prediction for ",  days, " days")

	# build freatues table
	features = create_feature_table(days, today, start)
	predictors =  features[X_var]

	# make prediction	
	features["lgb"] = lgb_prediction(predictors)
	features["rf"] = rf_prediction(predictors)
	# use ensemble
	features["prediction"] = [ max(features["lgb"].loc[x], features["rf"].loc[x]) for x in features["lgb"].index]

	# prepare output
	output = features[["date", "country_1", "channel_id", "prediction"]]
	output.columns = ["date", "country", "channel", "n_tickets"]
	output.to_csv("output.csv")
	print("Job done, please see the prediction in output.csv")