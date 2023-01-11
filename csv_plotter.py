import csv
import pandas as pd
import matplotlib

### test write
# def csv_write(data):
#     with open('csv_test.csv', 'w') as file:
#         writer = csv.writer(file)
        # writer.writerow(data)

### extract data [correct filepath etc]
df = pd.read_csv("csv_test_100.csv")
df_0 = df.iloc[:,0] # .astype(str)

print(type(df_0))
df_0 = df_0.str.extract("(?<=val = Array([) (.*) (?=])")
df_0

### plot