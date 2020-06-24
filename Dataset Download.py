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

'---------------------------------------------------------'
''' Part1 '''
''' Bond Data From WRDS '''

## Connect to WRDS Database
Conn=wrds.Connection(wrds_username='sevenz')

## Download Fama-Bliss Dataset
data_fb=Conn.raw_sql('select * from crsp.tfz_mth_fb')
data_fb.drop(['tmnomprc_flg'], axis=1, inplace=True)
data_fb['log_price']=data_fb['tmnomprc'].map(lambda x: np.log(x/100))

## Divide into 5 sub-datasets based on their maturities
# kytreasnox=2000047 : Bonds 1 Year
# kytreasnox=2000048 : Bonds 2 Year
# kytreasnox=2000049 : Bonds 3 Year
# kytreasnox=2000050 : Bonds 4 Year
# kytreasnox=2000051 : Bonds 5 Year

bond_1 = data_fb[data_fb['kytreasnox']==2000047].set_index('mcaldt')
bond_2 = data_fb[data_fb['kytreasnox']==2000048].set_index('mcaldt')
bond_3 = data_fb[data_fb['kytreasnox']==2000049].set_index('mcaldt')
bond_4 = data_fb[data_fb['kytreasnox']==2000050].set_index('mcaldt')
bond_5 = data_fb[data_fb['kytreasnox']==2000051].set_index('mcaldt')
Bond = [bond_1, bond_2, bond_3, bond_4, bond_5]

## Set the log yield of a 1-year bond to be the risk-free rate
risk_free_rate=(-1)*bond_1['log_price'].rename('risk_free_rate')
 
## Calculate Bond Excess Returns Across Maturities
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
df=excess_return['average_excess_return'].to_frame()
df['time_m']=df.index.map(lambda x: x.strftime('%Y-%m'))
df=df.set_index('time_m')

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
        trseries = series.diff().map(lambda x: x**2)
    elif trcode == 4:
        trseries = series.map(lambda x: np.log(x))
    elif trcode == 5:
        trseries = series.map(lambda x: np.log(x)).diff()
    elif trcode == 6:
        trseries = series.map(lambda x: np.log(x)).diff().map(lambda x: x**2)
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
    
    
## Get Final-Revised Data
def get_final_revised_series(var,trcode):
    series=fred.get_series_latest_release(var)
    trseries = trans(series,trcode).rename(var)
    trseries = trseries.reset_index()
    trseries = trseries.rename(columns={'index':'time'})
    trseries['time_m']=trseries['time'].map(lambda x: (x-datetime.timedelta(days=1)).strftime('%Y-%m'))
    trseries.drop('time',axis=1,inplace=True)
    trseries=trseries.set_index('time_m')
    return trseries

data_final_revised=df
for i , j in zip(list_of_vars, list_of_trcodes):
    data_final_revised=data_final_revised.join(get_final_revised_series(i,j))
    
data_final_revised.to_csv('data_final_revised.csv')

## Get Real-Time Data
def get_real_time_series(var,trcode):
    series_all = fred.get_series_all_releases(var)
    
    series_all['time_spread'] = series_all.apply(lambda x: (x['realtime_start']-x['date']).days, axis=1)
    series_all = series_all[series_all['time_spread']<62]
    
    series_all['month_spread'] = series_all.apply(lambda x: x['realtime_start'].month-x['date'].month, axis=1)
    series_all = series_all[series_all['month_spread'] != 2]
    series_all = series_all[series_all['month_spread'] != -10]
    
    series_all['month_diff'] = series_all['date'].map(lambda x: x.month).diff().shift(-1)
    series_all = series_all[series_all['month_diff'] != 0]
    
    series_all['time_m']=series_all['date'].map(lambda x: x.strftime('%Y-%m')).shift(-1)
    series_all[var] = trans(series_all['value'],trcode)
    series_real_time = series_all[['time_m',var]].set_index('time_m')
    
    return series_real_time

data_real_time = df
for i , j in zip(list_of_vars, list_of_trcodes):
    data_real_time=data_real_time.join(get_real_time_series(i,j))
    
data_real_time.to_csv('data_real_time.csv')
