import os
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import shutil
from multiprocessing import Process, Value, Lock

#search all files with your keyword under your path folder
def search(path, word):
    filelist = []
    for filename in os.listdir(path):
        fp = os.path.join(path, filename)
        if os.path.isfile(fp) and 'TDCS_M06A_'+ word in filename:
            #print(fp)
            filelist.append(fp)
    return filelist

def select_raw_latency_by_segment(df_input, entrance_sensor, exit_sensor):
    df = df_input.xs((entrance_sensor, exit_sensor), level=('Entrance', 'Exit'))
    return pd.DataFrame(df['Latency_Raw'])
#delete /latency_by_start_time folder underpath and create a new onej
def DeleteAndCreate(path,s_month):
    
    if os.path.exists(path+'/latency_by_start_time'):
        shutil.rmtree(path+'/latency_by_start_time')
    if not os.path.exists(path+'/latency_by_start_time'):
        os.makedirs(path+'/latency_by_start_time')
        os.makedirs(path+'/latency_by_start_time/fig')
    
#latency_by_start_time latencies of certain dates from rawdata file, 
#save latency of every monday between every pair in one csv file. 4*10=40 files in total
#and also save latency of all Monday between every pair in one csv file 10 file in total
#Input weekday as list of strings like weekday = ['20170306','20170313','20170320','20170327'],and path of rawdata file

def extract_latency_by_start_time(l_s_month,path0,names,pairs):
    for s_month in l_s_month:
        path = path0 + s_month
        DeleteAndCreate(path, s_month)
    print('DeleteAndCreate Success!')
    for pi in pairs:
        # LatencyAbnormalThreshold = 3000 #second #Using median we don't consider abnormal values
        df_3months = pd.read_pickle(path0 + 'L2017' + l_s_month[0] + '.pickle')
        df_3months = pd.DataFrame(index=df_3months.index, columns=['Latency_Raw'])
        # 删掉多余的index 只保留timestamp
        entrance_sensor, exit_sensor = '01F0339S', '01F0509S'
        df_3months = select_raw_latency_by_segment(df_3months, entrance_sensor, exit_sensor)
        # 每个Raw Latency的元素初始化为[]
        for index, row in df_3months.iterrows():
            row[0] = []
        for s_month in l_s_month:
            path = path0 + s_month
            # 直接每个文件循环
            # 筛选出所有表
            fl = os.listdir(path)
            l_del = []
            for ni, fi in enumerate(fl):
                if '.csv' not in fi:
                    l_del.append(fi)
            [fl.remove(fi) for fi in l_del]

            for fi in fl:
                print('processing', fi)
                df = pd.read_csv(path + '/' + fi, names=names)
                # for pi in pairs:

                bol0 = df.iloc[:, 7].str.contains(pi[0])
                df1 = df[bol0]
                bol1 = df1.iloc[:, 7].str.contains(pi[1])
                df1 = df1[bol1].copy()
                # 计算InTime Outtime
                df1['Intime'] = '' * len(df1)
                df1['Outtime'] = '' * len(df1)
                di = fi[-15:-11]  # date, like'0301'
                for si in range(len(df1)):  # regular expression
                    s = df1.iloc[si, 7]
                    s = s.split(';')
                    for ti in s:
                        if pi[0] in ti and di[0:2] + '-' + di[2:4] in ti:
                            df1.iloc[si, 8] = (ti.split('+'))[0]
                        elif pi[1] in ti and di[0:2] + '-' + di[2:4] in ti:
                            df1.iloc[si, 9] = (ti.split('+'))[0]
                # 计算latency
                df1.iloc[:, 8] = pd.to_datetime(df1.iloc[:, 8])
                df1.iloc[:, 9] = pd.to_datetime(df1.iloc[:, 9])
                df1['latency'] = df1.iloc[:, 9] - df1.iloc[:, 8]
                df1.iloc[:, 10] = (df1.iloc[:, 10]).dt.total_seconds()  # latency
                # df1 = df1.loc[df1['latency']<LatencyAbnormalThreshold]#剔除异常值
                # 这里也考虑一下下一个小时才到达的
                l_hour=[int(fi[-10:-8])]
                #l_hour = [int(fi[-10:-8]), int(fi[-10:-8]) + 1] if int(fi[-10:-8]) < 23 else [int(fi[-10:-8])]
                for each_hour in l_hour:
                    for each_minute in range(60):
                        # 筛选出发时间intime等于指定时分的
                        df3 = df1[(df1.iloc[:, 8].dt.minute == each_minute) & (df1.iloc[:, 8].dt.hour == each_hour)]
                        df_3months.loc[pd.Timestamp(2017, int(s_month), int(fi[-13:-11]), each_hour,
                                                    each_minute), 'Latency_Raw'] += df3['latency'].tolist()
        df_3months.to_pickle(path + '/' + pi[0] + '_' + pi[1] + '_latency_by_start_time' + '2017' + l_s_month[0] + '.pickle')


def run():
    l_s_month_arr = [['10', '11', '12']]
    path0 = '/Users/zhaoyuheng/Downloads/6980/'
    names = ['Vehicle Type', 'First Detection Time ', 'Origin Detector ID', 'Last Detection Time',
             'Destination Detector ID', 'Trip Length', 'Trip End', 'Trip Information']
    ll_pairs = [
        # ['01F0339S', '01F0376S'],
        # ['01F0339S', '01F0413S'],
        # ['01F0339S', '01F0467S'],
        # ['01F0339S', '01F0509S']
        ['01F0376S', '01F0413S'],
        # ['01F0376S', '01F0467S'],
        ['01F0376S', '01F0509S'],
        ['01F0413S', '01F0467S'],
        # ['01F0413S', '01F0509S'],
        # ['01F0467S', '01F0509S']
    ]
    
    process_num = len(l_s_month_arr)
    processes = []
    for i in range(process_num):
        p = Process(target=extract_latency_by_start_time, args=(l_s_month_arr[i],path0,names,ll_pairs,))
        p.start() 
        processes.append(p)
    pi = 0
    for p in processes:
        p.join()
        print("latency_by_start_timeion {} success!".format(pi))
        pi += 1
#     extract_latency_by_start_time(l_s_month,path0,names,ll_pairs)
    print("latency_by_start_timeion all success!")
    #Draw the plots of latency of all Monday between every pair,i.e. 10 plots.
if __name__ == '__main__':
    run()