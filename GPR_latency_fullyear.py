
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.gaussian_process import GaussianProcessRegressor

def gaussian(df): 
    n=1020
    d=1
    pre=[]
    a=np.linspace(6,23,n)
    X = a.reshape(n, d)
    gp = GaussianProcessRegressor(kernel=RBF(length_scale=2, length_scale_bounds=(2, 1e3)))
    fiting=gp.fit(X,df)
    c=np.linspace(6,23,n)
    pre=gp.predict(c.reshape(n,1))
    return pre

picklepath = '/Users/zhaoyuheng/Downloads/OneDrive_1_6-15-2020/Data/df_new_ver5_20190401.pickle'
path0 = '/Users/zhaoyuheng/Downloads/OneDrive_1_6-15-2020/Data/'

#UPDATE 5 MINS MEDIAN LATENCY RESULT OF 2017###########
weekday=3
df = pd.read_pickle(picklepath)
df = df.loc[df.index.year == 2017]
fiveminmedian = pd.read_csv(path0+'5_med_latency1.csv')
# fiveminmedian_arrival = pd.read_csv(path0+'5_med_latency_arrival_339_509.csv')
fiveminmedian.set_index('Timestamp',inplace=True)
# fiveminmedian_arrival.set_index('Timestamp',inplace=True)
df1 = df.join(fiveminmedian)
# df2 = df.join(fiveminmedian_arrival)
# df2.rename(columns={'Latency_Raw':'5_med_latency_arrival'},inplace = True)
# df2 = df2.loc[df1.index.year == 2017].fillna(0)
# df2 = df2['5_med_latency_arrival']

df1.rename(columns={'Latency_Raw':'5_med_latency'},inplace = True)
df1 = df1.loc[df1.index.year == 2017].fillna(0)
df1 = df1['5_med_latency']

# # UPDATE GPR PREDICTION RESULT OF 2017

mon=df1[df1.index.weekday==0]
tue=df1[df1.index.weekday==1]
wes=df1[df1.index.weekday==2]
thu=df1[df1.index.weekday==3]
fri=df1[df1.index.weekday==4]
sat=df1[df1.index.weekday==5]
sun=df1[df1.index.weekday==6]
mon_mean=gaussian(mon.groupby([mon.index.hour,mon.index.minute]).agg('mean'))
tue_mean=gaussian(tue.groupby([tue.index.hour,tue.index.minute]).agg('mean'))
wes_mean=gaussian(wes.groupby([wes.index.hour,wes.index.minute]).agg('mean'))
thu_mean=gaussian(thu.groupby([thu.index.hour,thu.index.minute]).agg('mean'))
fri_mean=gaussian(fri.groupby([fri.index.hour,fri.index.minute]).agg('mean'))
sat_mean=gaussian(sat.groupby([sat.index.hour,sat.index.minute]).agg('mean'))
sun_mean=gaussian(sun.groupby([sun.index.hour,sun.index.minute]).agg('mean'))

result=pd.DataFrame(index=df.index)
df00=pd.DataFrame(np.tile(mon_mean,52),index=result[result.index.weekday==0].index)
df01=pd.DataFrame(np.tile(tue_mean,52),index=result[result.index.weekday==1].index)
df02=pd.DataFrame(np.tile(wes_mean,52),index=result[result.index.weekday==2].index)
df03=pd.DataFrame(np.tile(thu_mean,52),index=result[result.index.weekday==3].index)
df04=pd.DataFrame(np.tile(fri_mean,52),index=result[result.index.weekday==4].index)
df05=pd.DataFrame(np.tile(sat_mean,52),index=result[result.index.weekday==5].index)
df06=pd.DataFrame(np.tile(sun_mean,53),index=result[result.index.weekday==6].index)
df000=pd.concat([df00,df01,df02,df03,df04,df05,df06])
result=result.join(df000)


# UPDATE PAST MEDIANS DATA

# past_medians = pd.read_csv(path0+'past_result_2017_arrival_339_509.csv')
# past_medians.set_index('Timestamp',inplace=True)
# past_medians.columns = ['past_5_med_arrival', 'past_10_med_arrival', 'past_15_med_arrival', 'past_20_med_arrival']
# df = df.join(past_medians)
# past_medians_376 = pd.read_csv(path0+'past_result_2017_arrival_376_509.csv')
# past_medians_376.set_index('Timestamp',inplace=True)
# past_medians_376.columns = ['376_past_5_median','376_past_10_median','376_past_15_median','376_past_20_median']
# df = df.join(past_medians_376)
# past_medians_413 = pd.read_csv(path0+'past_result_2017_arrival_413_509.csv')
# past_medians_413.set_index('Timestamp',inplace=True)
# past_medians_413.columns = ['413_past_5_median','413_past_10_median','413_past_15_median', '413_past_20_median']
# df = df.join(past_medians_413)
# past_medians_467 = pd.read_csv(path0+'past_result_2017_arrival_467_509.csv')
# past_medians_467.set_index('Timestamp',inplace=True)
# past_medians_467.columns = ['467_past_5_median','467_past_10_median','467_past_15_median','467_past_20_median']
# df = df.join(past_medians_467)



df['p_l_median']=result
# /Users/zhaoyuheng/Downloads/OneDrive_1_6-15-2020/Data/df_median_speed_starttime.pickle
# # df['5_med_latency']=df1.values
# # df['5_med_latency_arrival']=df2.values
# df.to_pickle('/Users/zhaoyuheng/Downloads/OneDrive_1_6-15-2020/Data/df_median_version_arrival.pickle')





    




