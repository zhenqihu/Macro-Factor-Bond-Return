#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 09:59:37 2020

@author: Luke
"""

from fredapi import Fred
import datetime
import numpy as np
import pandas as pd
import wrds
from sklearn.preprocessing import StandardScaler


'---------------------------------------------------------'
''' Part1 '''
''' Bond Data From WRDS '''

## Connect to WRDS Database
Conn=wrds.Connection(wrds_username='sevenz')

## Download Fama-Bliss Dataset
data_fb=Conn.raw_sql('SELECT kytreasnox, mcaldt, tmnomprc FROM crsp.tfz_mth_fb',\
                     date_cols=['mcaldt']) 
data_fb['log_price']=data_fb['tmnomprc'].map(lambda x: np.log(x/100))
data_fb['kytreasnox']=data_fb['kytreasnox']-2000046
data_fb['log_yield']=data_fb.apply(lambda x: -1*x['log_price']/x['kytreasnox'], axis=1)
Conn.close()
## Divide into 5 sub-datasets based on their maturities
# kytreasnox=2000047-2000051 : Bonds 1-5 Year
bond_1 = data_fb.query('kytreasnox==1').set_index('mcaldt')
bond_2 = data_fb.query('kytreasnox==2').set_index('mcaldt')
bond_3 = data_fb.query('kytreasnox==3').set_index('mcaldt')
bond_4 = data_fb.query('kytreasnox==4').set_index('mcaldt')
bond_5 = data_fb.query('kytreasnox==5').set_index('mcaldt')

## Calculate risk-free rate & forward rates
risk_free_rate=bond_1['log_yield'].rename('risk_free_rate')
f_1_2 = (bond_1.log_price - bond_2.log_price).rename('forward_1_2')
f_2_3 = (bond_2.log_price - bond_3.log_price).rename('forward_2_3')
f_3_4 = (bond_3.log_price - bond_4.log_price).rename('forward_3_4')
f_4_5 = (bond_4.log_price - bond_5.log_price).rename('forward_4_5')
forward_factor = risk_free_rate.to_frame().join(f_1_2).join(f_2_3)\
    .join(f_3_4).join(f_4_5)

## Calculate Excess Bond Returns Across Maturities
def excess_return_calculation(b2,b1):
    b2=b2.join(b1['log_price'].rename('log_price_lag').shift(-12))
    b2=b2.join(risk_free_rate)
    b2['excess_return']=b2.apply(lambda x: x['log_price_lag']-x['log_price']-x['risk_free_rate'],axis=1)
    return b2['excess_return']

excess_return_2=excess_return_calculation(bond_2,bond_1)
excess_return_3=excess_return_calculation(bond_3,bond_2)
excess_return_4=excess_return_calculation(bond_4,bond_3)
excess_return_5=excess_return_calculation(bond_5,bond_4)

## Average Bond Excess Return
excess_return = pd.DataFrame({'excess_return_2':excess_return_2,
                              'excess_return_3':excess_return_3,
                              'excess_return_4':excess_return_4,
                              'excess_return_5':excess_return_5,
                              }).dropna()

excess_return['average_excess_return']=excess_return.mean(axis=1)
bond_returns=excess_return.join(forward_factor)
bond_returns['time_m']=bond_returns.index.map(lambda x: x.strftime('%Y-%m'))
bond_returns=bond_returns.set_index('time_m')
bond_returns=bond_returns.query("time_m >= '1982-03' & time_m <= '2015-11'")

bond_returns.to_csv('bond data.csv')



'---------------------------------------------------------'
''' Part2 '''
''' Econ Data From ALFRED Datasets '''

## FRED API Key: 3c63eb8e57a6e2cd7ab2fc4b913a5757
fred = Fred(api_key ='3c63eb8e57a6e2cd7ab2fc4b913a5757')

##Transformation Codes
def trans(series, trcode):
    if trcode == 1 :
        trseries = series
    elif trcode == 2:
        trseries = series.diff()
    elif trcode == 3:
        trseries = series.diff().diff()
    elif trcode == 4:
        trseries = series.map(lambda x: np.log(x))
    elif trcode == 5:
        trseries = series.map(lambda x: np.log(x)).diff()
    elif trcode == 6:
        trseries = series.map(lambda x: np.log(x)).diff().diff()
    return trseries
        
## List of Variables and Transformation Codes
list_of_vars = ['INDPRO','AWHMAN','AWHNONAG','AWOTMAN','DSPI','DSPIC96',
                'PI','CE16OV','CLF16OV','PAYEMS','MANEMP','DMANEMP',
                'NDMANEMP','USCONS','USFIRE','USGOOD','USGOVT','USMINE',
                'USPRIV','USSERV','USTPU','USTRADE','USWTRADE','SRVPRD',
                'UEMP5TO14','UEMP15OV','UEMP15T26','UEMP27OV','UEMPLT5',
                'UEMPMEAN','UEMPMED','UNEMPLOY','UNRATE','PCE','PCEDG',
                'PCEND','PCES','HOUST','HOUST1F','HOUST2F','CURRSL',
                'DEMDEPSL','M1SL','OCDSL','SAVINGSL','STDCBSL','STDSL',
                'STDTI','SVGCBSL','SVGTI','SVSTCBSL','TCDSL','CPIAUCSL',
                'PFCGEF','PPICPE','PPICRM','PPIFCF','PPIFGS','PPIIFF','PPIITM']

list_of_trcodes = [5 for i in range(60)]
list_of_trcodes[1] = 1
for i in [2,3,29,30,32]:
    list_of_trcodes[i] = 2 

for i in [37,38,39]:
    list_of_trcodes[i] = 4

for i in range(41,60):
    list_of_trcodes[i]= 6
    
#Standardize prior to estimation
def standardized(X):
    scaler = StandardScaler()
    X_scaler = scaler.fit_transform(X)
    X_standard = pd.DataFrame(X_scaler,index=X.index,columns=X.columns)
    return X_standard
    
## Get Final-Revised Data
def get_final_revised_series(var,trcode):
    series = fred.get_series_latest_release(var)
    trseries = trans(series,trcode).rename(var)
    trseries = trseries.reset_index()
    trseries = trseries.rename(columns={'index':'time'})
    trseries['time_m']=trseries['time'].map(lambda x: (x-datetime.timedelta(days=1)).strftime('%Y-%m'))
    trseries.drop('time',axis=1,inplace=True)
    trseries=trseries.set_index('time_m')
    return trseries

df = get_final_revised_series(list_of_vars[0],list_of_trcodes[0])\
    .query("time_m >= '1982-03' & time_m <= '2015-11'")
    
for i , j in zip(list_of_vars[1:], list_of_trcodes[1:]):
    df=df.join(get_final_revised_series(i,j))

df = df.fillna(method = 'ffill')
data_final_revised = standardized(df)
data_final_revised.to_csv('econ_data_final_revised.csv')


## Get Real-Time Data
def get_real_time_series(var,trcode):
    series_all = fred.get_series_all_releases(var)
    
    series_all['time_spread'] = series_all.apply(lambda x: (x['realtime_start']-x['date']).days, axis=1)
    series_all = series_all.query('time_spread<62')
    
    series_all['month_spread'] = series_all.apply(lambda x: x['realtime_start'].month-x['date'].month, axis=1)
    series_all = series_all.query('month_spread != 2 and month_spread != -10')
    
    series_all['month_diff'] = series_all['date'].map(lambda x: x.month).diff().shift(-1)
    series_all = series_all.query('month_diff != 0')    
    
    series_all['time_m']=series_all['date'].map(lambda x: x.strftime('%Y-%m')).shift(-1)
    series_all[var] = trans(series_all['value'],trcode)
    series_real_time = series_all[['time_m',var]].set_index('time_m')
    
    return series_real_time

dr = get_real_time_series(list_of_vars[0],list_of_trcodes[0])\
    .query("time_m >= '1982-03' & time_m <= '2015-11'")
    
for i , j in zip(list_of_vars[1:], list_of_trcodes[1:]):
    dr = dr.join(get_real_time_series(i,j))

dr = dr.fillna(method = 'ffill')
data_real_time = standardized(dr)
data_real_time.to_csv('econ_data_real_time.csv')
