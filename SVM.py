import requests
import simplejson
import re
import operator
import urllib.request
import os
import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as matplot

crypto_codes = ["BTC-USD"]
base_url = "http://ichart.finance.yahoo.com/table.csv?s="
index = 0

dataset_url = base_url + crypto_codes[index]

output_path = "C:/Users/ninja/Downloads/AP Research/Data/SVM/"
new_output_path = output_path + crypto_codes[index] + "_new.csv"
output_path = output_path + crypto_codes[index] + ".csv"

# try:
urllib.request.urlretrieve(dataset_url, output_path)
# except urllib.request.ContentTooShortError as p:
#    outfile = open(output_path, "w")
#    outfile.write(p.content)
#    outfile.close()
