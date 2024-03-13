from RF import RF
from SVM import SVM
import pandas as pd

data_list = ['Data/BCH-USD_daily.csv', 'Data/BTC-USD_daily.csv', 'Data/DASH-USD_daily.csv',
             'Data/EOS-USD_daily.csv', 'Data/ETC-USD_daily.csv', 'Data/ETH-USD_daily.csv',
             'Data/LTC-USD_daily.csv', 'Data/OMG-USD_daily.csv', 'Data/XMR-USD_daily.csv',
             'Data/XRP-USD_daily.csv', 'Data/ZEC-USD_daily.csv']

final_list = ['BCH-SVM-output', 'BTC-SVM-output', 'DASH-SVM-output',
              'EOS-SVM-output', 'ETC-SVM-output', 'ETH-SVM-output',
              'LTC-SVM-output', 'OMG-SVM-output', 'XMR-SVM-output',
              'XRP-SVM-output', 'ZEC-SVM-output',
              'BCH-RF-output', 'BTC-RF-output', 'DASH-RF-output',
              'EOS-RF-output', 'ETC-RF-output', 'ETH-RF-output',
              'LTC-RF-output', 'OMG-RF-output', 'XMR-RF-output',
              'XRP-RF-output', 'ZEC-RF-output']
writer = pd.ExcelWriter('output_file.xlsx', engine='xlsxwriter')

for x in range(len(data_list)):
    svm = SVM(data_list[x])
    rf = RF(data_list[x])
    svm.final_dataframe().to_excel(writer, sheet_name=final_list[x], index=False)
    rf.final_dataframe().to_excel(writer, sheet_name=final_list[x+11], index=False)
    print(data_list[x] + ' completed.')

writer.close()
