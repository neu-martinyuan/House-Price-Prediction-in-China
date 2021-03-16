import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from multiprocessing import Pool
import multiprocessing as mp
import time

def common_member(a, b):
    a_set = set([x.upper() for x in a])
    b_set = set([x.upper() for x in b])

    if (a_set & b_set):
        #print(len(a_set & b_set))
        return list(a_set & b_set)
    else:
        print("No common elements")

# update the time information
def update_year(year):
    sub_input = housing_price.loc[housing_price['Date of Transfer'] == year]
    sub_input['Year'] = year
    return sub_input['Year']

#parallel 改进
def regression(X, y, num, X_predict):
    return MultiOutputRegressor(GradientBoostingRegressor(random_state=num)).fit(X, y).predict(X_predict)

housing_price = pd.read_csv(os.path.join('data','price_paid_records.csv'))
housing_price = housing_price[['Price', 'Date of Transfer','Town/City', 'District', 'County']]
housing_price = housing_price[2855141:]
housing_price_name = housing_price['District'].unique().tolist()

gdp = pd.read_excel(os.path.join('data','regionalgrossdomesticproductgdpcityregions.xlsx'), sheet_name='Table 5')
gva = pd.read_excel(os.path.join('data','regionalgrossdomesticproductgdpcityregions.xlsx'), sheet_name='Table 1')
vat = pd.read_excel(os.path.join('data','regionalgrossdomesticproductgdpcityregions.xlsx'), sheet_name='Table 2')
other_tax = pd.read_excel(os.path.join('data','regionalgrossdomesticproductgdpcityregions.xlsx'), sheet_name='Table 3')
subsidies = pd.read_excel(os.path.join('data','regionalgrossdomesticproductgdpcityregions.xlsx'), sheet_name='Table 4')
population = pd.read_excel(os.path.join('data','regionalgrossdomesticproductgdpcityregions.xlsx'), sheet_name='Table 6')

gdp = gdp[gdp['Area type'] == 'LA']
gva = gva[gva['Area type'] == 'LA']
vat = vat[vat['Area type'] == 'LA']
other_tax = other_tax[other_tax['Area type'] == 'LA']
subsidies = subsidies[subsidies['Area type'] == 'LA']
population = population[population['Area type'] == 'LA']


hp_pop_name = common_member(housing_price_name, gdp['Area name'].unique().tolist())

# 将相同城市的数据保留
housing_price = housing_price[housing_price['District'].isin(hp_pop_name)]

# Serial Compute arrange the time information
start = time.perf_counter()
for i in range(1998, 2018):
    tmp = str(i)
    housing_price.loc[housing_price['Date of Transfer'].str.startswith(tmp), 'Date of Transfer'] = tmp
print("Serial compute, Time elapsed: ", (time.perf_counter() - start)*1000)
all_years = housing_price['Date of Transfer'].unique()

#规整城市名
gdp['Area name'] = gdp['Area name'].str.upper()
gva['Area name'] = gva['Area name'].str.upper()
vat['Area name'] = vat['Area name'].str.upper()
other_tax['Area name'] = other_tax['Area name'].str.upper()
subsidies['Area name'] = subsidies['Area name'].str.upper()
population['Area name'] = population['Area name'].str.upper()

y_data = [[]]
for name in hp_pop_name:
    tmp1 = housing_price[housing_price['District'] == name]
    for year in range(1998, 2018):
        tmp2 = tmp1[tmp1['Date of Transfer'] == str(year)]
        y_data.append(tmp2.Price.tolist())
del y_data[0]

x_data = [[]]
j = 0
for name in hp_pop_name:
    tmp1 = gdp[gdp['Area name'] == name]
    #tmp2 = gva[gva['Area name'] == name]
    #tmp3 = vat[vat['Area name'] == name]
    #tmp4 = other_tax[other_tax['Area name'] == name]
    #tmp5 = subsidies[subsidies['Area name'] == name]
    tmp6 = population[population['Area name'] == name]
    for year in range(1998, 2018):
        x_data[j].append(int(tmp1[year]))
        #x_data[j].append(int(tmp2[year]))
        #x_data[j].append(int(tmp3[year]))
        #x_data[j].append(int(tmp4[year]))
        #x_data[j].append(int(tmp5[year]))
        x_data[j].append(int(tmp6[year]))
        j = j + 1
        x_data.append([])
del x_data[2140]

b = []
for i in range(len(y_data)):
    if len(y_data[i]) == 0:
        b.append(i)

x_data = [x_data[i] for i in range(len(x_data)) if (i not in b)]
y_data = [y_data[j] for j in range(len(y_data)) if (j not in b)]

b = []
for i in range(len(y_data)):
    if len(y_data[i]) <= 885:
        b.append(i)

x_data = [x_data[i] for i in range(len(x_data)) if (i not in b)]
y_data = [y_data[j] for j in range(len(y_data)) if (j not in b)]

for i in range(len(y_data)):
    y_data[i] = y_data[i][:886]

# Tokyo
# They are the GDP and population data in Shinjuku ku in Tokyo for 1990, 1995, 2000, 2005, 2010 and 2015
shinjuku_x_data = [[17812.788 ,296860], [21573.73,279048],[18842.2,286726],
                   [15368.742,305716],[26615.52,326000],[19744.85,337556]]

shinjuku_y_actual = [590000, 414800, 406400, 410800, 548372, 505609]

# Men Tou Gou - Beijing District
# They are the GDP and population data in Men Tou Gou District, Beijing, from 2011 to 2019
Men_Tou_Gou_x_data = [[557.81,239000],[622.6,240000],[746.13,241000],[822.8,244000],
                      [963.215,290000],[1157.4287,294000],[1312.2032,298000],[1390.1877,303000],
                      [1485.44902,306000],[1616.3466,308000],[1736.471,311000],[1918.3868,322000],
                      [2136.33,331000],[2931.41,345000]]

Men_Tou_Gou_y_actual = [213402,176107,237469,263299,242465,334411,519424,468838,455204]

if __name__ == "__main__":
    start = time.perf_counter()
    pool = Pool(processes=8)
    results = pool.map(update_year, all_years)
    pool.close()
    pool.join()
    print("Parallel compute, CPU=8, Time elapsed: ", (time.perf_counter() - start)*1000)

    # concatenate results into a single pd.Series
    results = pd.concat(results)

    # join results with original df
    housing_price = housing_price.join(results)


    start = time.perf_counter()
    regression(x_data, y_data, 0, shinjuku_x_data)
    regression(x_data, y_data, 1, shinjuku_x_data)
    regression(x_data, y_data, 2, shinjuku_x_data)
    print("Serial compute in ml regression, Time elapsed: ", (time.perf_counter() - start)*1000)

    start = time.perf_counter()
    pool = Pool(processes=8)
    #args = [(x_data, y_data, num) for num in range(0, 5)]
    args = [(x_data, y_data, num, shinjuku_x_data) for num in range(0, 3)]
    shinjuku_results = pool.starmap(regression, args)
    pool.close()
    print("Parallel compute in ml regression, CPU=8, Time elapsed: ", (time.perf_counter() - start)*1000)

    start = time.perf_counter()
    pool = Pool(processes=8)
    #args = [(x_data, y_data, num) for num in range(0, 5)]
    args = [(x_data, y_data, num, Men_Tou_Gou_x_data) for num in range(0, 3)]
    Men_Tou_Gou_results = pool.starmap(regression, args)
    pool.close()
    print("Parallel compute in ml regression, CPU=8, Time elapsed: ", (time.perf_counter() - start)*1000)

    #print(results)
