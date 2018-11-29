#! /usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import pandas
import os

from sklearn.cluster import KMeans

# Load in raw data files
county_data_filename = "county_facts.csv" # Census statistics
election_data_filename = "2016_US_County_Level_Presidential_Results.csv" # Election outcomes
county_data = pandas.read_csv(county_data_filename)
election_data = pandas.read_csv(election_data_filename)

# Join the data along fips and drop redundant columns, and remove all rows with nan.
joined_data = county_data.join(election_data.set_index('combined_fips'),on='fips')
joined_data = joined_data.drop(columns = "state_abbr")
joined_data = joined_data.drop(columns = "county_name")
joined_data = joined_data.drop(columns = "Unnamed: 0")
joined_data = joined_data.dropna();

print(joined_data.shape)

# Generate new columns of the dataframe

def party_win(row):
    """
    Determines if Clinton or Trump won the county
    
    row: row of the dataframe, extract 'per_dem' (percent of votes won by the democratic candidate
    Return: Dem or Rep or Tie
    """
    if row['votes_dem'] > row['votes_gop']:
        return 'Dem'
    elif row['votes_dem'] < row['votes_gop']:
        return 'Rep'
    else:
        return 'Tie'

def strength_of_dem_win(row):
    return row['votes_dem']/(row['votes_dem']+row['votes_gop'])

def strength_of_rep_win(row):
    return row['votes_gop']/(row['votes_dem']+row['votes_gop'])

def classify_strength(row):
    per_dem = row['votes_dem']/(row['votes_dem']+row['votes_gop'])
    if per_dem < 0.2:
        return 'Strong Rep'
    elif per_dem < 0.4:
        return 'Lean Rep'
    elif per_dem < 0.6:
        return 'Toss Up'
    elif per_dem < 0.8:
        return 'Lean Dem'
    else:
        return 'Strong Dem'

joined_data['Winner'] = joined_data.apply(party_win,axis=1)
joined_data['Dem_Win_Strength'] = joined_data.apply(strength_of_dem_win,axis=1)
joined_data['Rep_Win_Strength'] = joined_data.apply(strength_of_rep_win,axis=1)
joined_data['Strength_Class'] = joined_data.apply(classify_strength,axis=1)

print(joined_data['Winner'].value_counts())
print(joined_data['Strength_Class'].value_counts())
print(sum(joined_data.loc[:,'votes_dem']))
print(sum(joined_data.loc[:,'votes_gop']))

plt.hist(joined_data['Dem_Win_Strength'],bins=30)
plt.show()

plt.scatter(joined_data['Rep_Win_Strength'],joined_data['Dem_Win_Strength'])
plt.show()

kmeans = KMeans(n_clusters = 5, random_state = 0).fit(joined_data[['Rep_Win_Strength','Dem_Win_Strength']])

print(type(kmeans.labels_))

joined_data = joined_data.assign(Class = pandas.Series(kmeans.labels_))
print(kmeans.cluster_centers_)
print(joined_data['Class'].head())
print(joined_data['Class'].value_counts())

corr_data = joined_data.drop(['area_name','state_abbreviation','per_point_diff','Winner','Class','Strength_Class'],axis=1).corr()

print(type(corr_data))
print(corr_data['Dem_Win_Strength'])


# remove all columns with nan done
# classify D or R depending on 50% threshhold, then cluster data based on Strong R, mid R, toss up, mid D, and strong D // cluster or binning? binning is hard since we expect the races to lie all between 30 - 70%? done?? kind of
# bin the data to generate over the columns to find frequent data sets? // do we need to normalize the data?
# association rules

# look at correlations between all of the attributes Done? kind of 

# random forest
# svm
# nn
