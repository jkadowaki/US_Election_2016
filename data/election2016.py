#! /usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import pandas
import os

county_data_filename = "county_facts.csv"
election_data_filename = "2016_US_County_Level_Presidential_Results.csv"

county_data = pandas.read_csv(county_data_filename)
election_data = pandas.read_csv(election_data_filename)

joined_data = county_data.set_index('fips').join(election_data.set_index('combined_fips'))

joined_data = joined_data.drop(columns = "state_abbr")
joined_data = joined_data.drop(columns = "county_name")

print(county_data.head())
print(election_data.head())
print(joined_data.head())
print(type(county_data))
