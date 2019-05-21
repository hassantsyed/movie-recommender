import sys, os
from collections import OrderedDict
import pandas as pd
from scipy.spatial.distance import euclidean, cosine
from scipy.stats import pearsonr
import math
from sklearn.metrics import mean_squared_error

cols = ['uid','mid','rating','na']

def evaluate():
	args = sys.argv[1:]
	
	info = OrderedDict([('command',None),('training',None),('k',None),('algo',None),('testing',None)])
	for k,v in zip(info,args):
		info[k] = v
	info['k'] = int(info['k'])
	funcs = {"average":average, "euclid":euclid, "cosine":cos, "pearson":pearson}
	info['algo'] = funcs[info['algo']]

	def algoName(name):
		for k, v in funcs.items():
			if name == v:
				return k

	#info = OrderedDict([('command','evaluate'),('training','t.test'),('testing','tt.test'),('k',20),('algo',funcs['cosine'])])

	try:
		dfTrain = pd.read_csv(info['training'], sep="\t", names = cols)
		dfTest = pd.read_csv(info['testing'], sep="\t", names = cols)
	except:
		print("Data is unreadable.")
		sys.exit()
	normalize(dfTrain)
	if not algoName(info['algo']) in funcs.keys():
		print("Algorithm specified is unsupported.")
		sys.exit()
	if not info['k'] >= 0:
		print("Invalid k.")
		sys.exit()

	preds = []
	actuals = []
	#print(mostSim(info['k'], info['algo'], info['training'], 1, 242, dfTrain))
	for user in dfTrain['uid'].unique():
		#all the movies to get predictions for
		for mov in dfTest.loc[dfTest['uid'] == user]['mid']:
			pred = mostSim(info['k'], info['algo'], info['training'], user, mov, dfTrain)
			actual = dfTest.loc[(dfTest['uid'] == user) & (dfTest['mid'] == mov)]['rating'].iloc[0]
			if pred == 0:
				continue
			preds.append(pred)
			actuals.append(actual)
	#mostSim(info['k'], info['algo'], info['training'], 7, 599, dfTrain)
	info['RMSE'] = math.sqrt(mean_squared_error(actuals, preds))

	

	print("myrex.command    = {}".format(info['command']))
	print("myrex.training   = {}".format(info['training']))
	print("myrex.testing    = {}".format(info['testing']))
	print("myrex.algorithm  = {}".format(algoName(info['algo'])))
	print("myrex.k          = {}".format(info['k']))
	print("myrex.RMSE       = {}".format(info['RMSE']))

def predict():
	args = sys.argv[1:]
	info = OrderedDict([('command',None),('training',None),('k',None),('algo',None),('uid',None),('mid',None)])
	for k,v in zip(info,args):
		info[k] = v
	info['k'] = int(info['k'])
	info['uid'] = int(info['uid'])
	info['mid'] = int(info['mid'])
	funcs = {"average":average, "euclid":euclid, "cosine":cos, "pearson":pearson}
	info['algo'] = funcs[info['algo']]

	def algoName(name):
		for k, v in funcs.items():
			if name == v:
				return k

	#info = OrderedDict([('command','predict'),('training','t.test'),('k',20),('algo',funcs['euclid']),('uid',6),('mid',5)])
	try:
		df = pd.read_csv(info['training'], sep="\t", names = cols)
	except:
		print("Data is unreadable.")
		sys.exit()
	if not (df['uid'] == info['uid']).any():
		print("Specified user does not exist in the data.")
		sys.exit()
	if not (df['mid'] == info['mid']).any():
		print("Specified movie has no ratings in the data.")
		sys.exit()
	if not algoName(info['algo']) in funcs.keys():
		print("Algorithm specified is unsupported.")
		sys.exit()
	if not info['k'] >= 0:
		print("Invalid k.")
		sys.exit()
	normalize(df)
	info['prediction'] = mostSim(info['k'], info['algo'], info['training'], info['uid'], info['mid'], df)
	if info['prediction'] == 0:
		sys.exit()
	
	
	print("myrex.command    = {}".format(info['command']))
	print("myrex.training   = {}".format(info['training']))
	print("myrex.algorithm  = {}".format(algoName(info['algo'])))
	print("myrex.k          = {}".format(info['k']))
	print("myrex.userID     = {}".format(info['uid']))
	print("myrex.movieID    = {}".format(info['mid']))
	print("myrex.prediction = {}".format(info['prediction']))

def mostSim(k, algo, file, uid, mid, df):
	#df = pd.read_csv(file, sep="\t", names = cols)
	#normalize(df)
	prediction = algo(uid, mid, k, df)
	return prediction

def cos(uid, mid, k, df):
	weights = {}
	curr = df.loc[df['uid'] == uid]
	users = set(df['uid'])
	for user in users:
		if user == uid:
			continue
		if not ((df['uid'] == user) & (df['mid'] == mid)).any():
			continue
		t = df.loc[df['uid'] == user]
		both = pd.merge(curr, t, how = 'inner', on = ['mid'])
		if len(both) == 0:
			continue
		#print(both)
		#print(both['normalized_x'], both['normalized_y'])
		try:
			dist = cosine(both['normalized_x'], both['normalized_y'])
		except:
			dist = 0
		if math.isnan(dist):
			dist = 0
		weights[user] = dist
	#print(weights)
	weights = sorted(weights.items(), key = lambda x: x[1], reverse = True)
	weights = weights[:k]
	if(len(weights) == 0):
		print("No valid ratings for the movie exist! The user had not ratings in common.")
		return 0
	#print(weights)
	predRating = 0
	wSum = 0.0
	ratings = df.loc[df['mid'] == mid]
	rates = {}
	for u, r in zip(ratings['uid'], ratings['normalized']):
		rates[u] = r
	for user in weights:
		if user[0] in rates:
			predRating += rates[user[0]] * user[1]
			wSum += abs(user[1])
	return denormalize(predRating/wSum)

def pearson(uid, mid, k, df):
	weights = {}
	curr = df.loc[df['uid'] == uid]
	users = set(df['uid'])
	for user in users:
		if user == uid:
			continue
		#doesn't compute similiarity if user hasnt rated desired movie
		if not ((df['uid'] == user) & (df['mid'] == mid)).any():
			continue
		t = df.loc[df['uid'] == user]
		both = pd.merge(curr, t, how = 'inner', on = ['mid'])
		if len(both) == 0:
			continue
		dist = pearsonr(both['normalized_x'], both['normalized_y'])
		val = dist[0]
		if math.isnan(val):
			val = 0
		weights[user] = val
	weights = sorted(weights.items(), key = lambda x: x[1], reverse = True)
	weights = weights[:k]
	if(len(weights) == 0):
		print("No valid ratings for the movie exist! The user had not ratings in common.")
		return 0
	predRating = 0
	wSum = 0.0
	ratings = df.loc[df['mid'] == mid]
	rates = {}
	for u, r in zip(ratings['uid'], ratings['normalized']):
		rates[u] = r
	for user in weights:
		if user[0] in rates:
			predRating += rates[user[0]] * user[1]
			wSum += abs(user[1])
	return denormalize(predRating/wSum) 

def euclid(uid, mid, k, df):
	weights = {}
	curr = df.loc[df['uid'] == uid]
	users = set(df['uid'])
	for user in users:
		if user == uid:
			continue
		if not ((df['uid'] == user) & (df['mid'] == mid)).any():
			continue
		t = df.loc[df['uid'] == user]
		both = pd.merge(curr, t, how = 'inner', on = ['mid'])
		if len(both) == 0:
			continue
		dist = euclidean(both['rating_x'], both['rating_y'])
		weights[user] = 1.0 / (1.0 + dist)
	#print(weights)
	weights = sorted(weights.items(), key = lambda x: x[1], reverse = True)
	weights = weights[:k]
	if(len(weights) == 0):
		print("No valid ratings for the movie exist! The user had not ratings in common.")
		return 0
	predRating = 0
	wSum = 0.0
	ratings = df.loc[df['mid'] == mid]
	rates = {}
	for u, r in zip(ratings['uid'], ratings['rating']):
		rates[u] = r
	for user in weights:
		if user[0] in rates:
			predRating += rates[user[0]] * user[1]
			wSum += user[1]
	return predRating/wSum

def average(uid, mid, k, df):
	mov = df.loc[df['mid'] == mid]
	return mov['rating'].mean()

def normalize(df):
	df['normalized'] = df.apply(lambda row: (row.rating-3)/2, axis = 1)

def denormalize(num):
	return 2 * num + 3

if __name__ == '__main__':
	if sys.argv[1] == 'predict':
		predict()
	elif sys.argv[1] == 'evaluate':
		evaluate()
