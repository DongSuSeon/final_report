import requests
import os
import json
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime as dt


skey1 = '09XYFOgTu4I5%2BM1bSMR%2F63wJY0tFA%2Beqq3hNnoST6xqUa2HhqC8SGFZsyJOdTyTva0NspRO0XnCh0TVLNggLdA%3D%3D'
skey2 = '09XYFOgTu4I5+M1bSMR/63wJY0tFA+eqq3hNnoST6xqUa2HhqC8SGFZsyJOdTyTva0NspRO0XnCh0TVLNggLdA=='

start_idx = [0, 1440*45, 1440*51, 1440*54, 1440*59, 1440*67, 1440*84, 1440*94, 1440*103, 1440*113, 1440*116, 1440*119, 1440*124, 1440*129, 1440*136]
end_idx = [1440*43, 1440*50, 1440*53, 1440*57, 1440*66, 1440*83, 1440*93, 1440*100, 1440*112, 1440*115, 1440*118, 1440*123, 1440*128, 1440*135, -1]
date = []
fric = []
t_fric = []
hum = []
state = []
x_vol = []
y_vol = []
atm_tem = []
road_tem = []
lens = []
second_tem = []
eqip_tem = []
for i in range(200):
    print(i)
    # road state
    url = 'http://apis.data.go.kr/C100006/zerocity/getIotRoadList'
    type = ''
    nrow = '1000'
    npage = str(int(i+1))
    sdate = '2019-01-01'
    edate = '2021-12-01'
    params = {'serviceKey': skey2, 'type': type, 'numOfRows': nrow, 'pageNo': npage, 'startDt': sdate, 'endDt': edate}

    # Get Result
    deter = True
    # while deter:
    #     try:
    #         response = requests.get(url, params=params)
    #         result = response.content
    #         dict_str = result.decode("UTF-8")
    #         my_data = json.loads(dict_str)
    #
    #         deter = False
    #     except:
    #         print('exception!')
    #         pass
    # io = open('{}.json'.format(i),'w')
    # json.dump(my_data, io)
    # io.close()
    io = open('./jsons/{}.json'.format(i), 'r')
    my_data = json.load(io)
    io.close()
    for j in range(int(nrow)):
        temp = my_data[0]['iotRoadFileList'][j]
        date.append(temp['ocrr_dt'])
        fric.append(float(temp['avg_rdsrfc_frct_nmvl']))
        t_fric.append(float(temp['trinspct_rdsrfc_frct_nmvl']))
        hum.append(float(temp['rltv_hmdt']))
        state.append(float(temp['avg_road_stts_cd']))
        x_vol.append(float(temp['x_volt']))
        y_vol.append(float(temp['y_volt']))
        atm_tem.append(float(temp['atmp_tmpr']))
        road_tem.append(float(temp['road_tmpr']))
        lens.append(float(temp['lens_stts_cd']))
        second_tem.append(float(temp['scnd_atmp_tmpr']))
        eqip_tem.append(float(temp['eqpm_tmpr']))
time_stamp = []
for i in range(len(date)):
    time_stamp.append(dt.strptime(date[i][:16],"%Y-%m-%d %H:%M").timestamp())

obs_time = []
obs_rain = []
for i in range(1,17):
    dum = np.genfromtxt('./gong/{}.csv'.format(i), skip_header=1, delimiter=',', dtype=None)
    for j in dum:
        obs_time.append(j[1].decode('utf-8'))
        obs_rain.append(j[4])

obs_timestamp = []
for i in range(len(obs_time)):
    obs_timestamp.append(dt.strptime(obs_time[i],"%Y-%m-%d %H:%M").timestamp())

obs_timestamp2 = []
obs_rain2 = []
for i in range(len(obs_rain)-1):
    obs_timestamp2.append((obs_timestamp[i+1]+obs_timestamp[i])/2)
    obs_rain2.append((obs_rain[i+1]-obs_rain[i]) / 10)
obs_rain2 = np.maximum(obs_rain2, 0)
rain_interp = np.interp(time_stamp, obs_timestamp2, obs_rain2)

obs_time2 = []
obs_rain3 = []
obs_rain_tf = []
dum = np.genfromtxt('./aws/1.csv', skip_header=1, delimiter=',', dtype=None)
for j in dum:
    obs_time2.append(j[1].decode('utf-8'))
    obs_rain3.append(float(j[3]))
    obs_rain_tf.append(float(j[4]) / 10)

obs_rain3 = np.array(obs_rain3) + 0.0
obs_rain_tf = np.array(obs_rain_tf) + 0.0
obs_timestamp3 = []
for i in range(len(obs_time2)):
    obs_timestamp3.append(dt.strptime(obs_time2[i],"%Y-%m-%d %H:%M").timestamp())

time_lastrain = []
tdum = 0
not_obs_rain_tf = np.invert((obs_rain_tf > 0))
for i in range(len(not_obs_rain_tf)):
    tdum = not_obs_rain_tf[i] * (tdum + not_obs_rain_tf[i])
    time_lastrain.append(tdum/3600)
rain_interp2 = np.interp(time_stamp, obs_timestamp3, obs_rain3)
raintf_interp = np.interp(time_stamp, obs_timestamp3, obs_rain_tf)
lastrain_interp = np.interp(time_stamp, obs_timestamp3, time_lastrain)

fric_save = []
hum_save = []
atm_tem_save = []
road_tem_save = []
rain_save = []
rain_save2 = []
raintf_save = []
lastrain_save = []
x_vol_save = []
y_vol_save = []
state_save = []
lens_save = []
second_tem_save = []
eqip_tem_save = []
# for i in range(len(start_idx)):
#     idx = slice(start_idx[i], end_idx[i])
#     fric_save.append(fric[idx])
#     hum_save.append(hum[idx])
#     atm_tem_save.append(atm_tem[idx])
#     road_tem_save.append(road_tem[idx])
#     rain_save.append(rain_interp[idx])
#     x_vol_save.append(x_vol[idx])
#     y_vol_save.append(y_vol[idx])
#     state_save.append(state[idx])
#     lens_save.append(lens[idx])
#     second_tem_save.append(second_tem[idx])
#     eqip_tem_save.append(eqip_tem[idx])
fric_save.append(fric)
hum_save.append(hum)
atm_tem_save.append(atm_tem)
road_tem_save.append(road_tem)
rain_save.append(rain_interp)
rain_save2.append(rain_interp2)
raintf_save.append(raintf_interp)
lastrain_save.append(lastrain_interp)
x_vol_save.append(x_vol)
y_vol_save.append(y_vol)
state_save.append(state)
lens_save.append(lens)
second_tem_save.append(second_tem)
eqip_tem_save.append(eqip_tem)
fric_save = [x for xs in fric_save for x in xs]
hum_save = [x for xs in hum_save for x in xs]
atm_tem_save = [x for xs in atm_tem_save for x in xs]
road_tem_save = [x for xs in road_tem_save for x in xs]
rain_save = [x for xs in rain_save for x in xs]
rain_save2 = [x for xs in rain_save2 for x in xs]
raintf_save = [x for xs in raintf_save for x in xs]
lastrain_save = [x for xs in lastrain_save for x in xs]
x_vol_save = [x for xs in x_vol_save for x in xs]
y_vol_save = [x for xs in y_vol_save for x in xs]
state_save = [x for xs in state_save for x in xs]
lens_save = [x for xs in lens_save for x in xs]
second_tem_save = [x for xs in second_tem_save for x in xs]
eqip_tem_save = [x for xs in eqip_tem_save for x in xs]

ratio = []
for i in range(len(x_vol_save)):
    ratio.append(y_vol_save[i]/x_vol_save[i])

input = np.array([rain_save, rain_save2, raintf_save, lastrain_save, hum_save, atm_tem_save, road_tem_save, state_save, lens_save, second_tem_save, eqip_tem_save]).T
lable = np.array([(np.array(fric_save)==0.0),
                  (np.array(fric_save)==0.1),
                  (np.array(fric_save)==0.2),
                  (np.array(fric_save)==0.3),
                  (np.array(fric_save)==0.4),
                  (np.array(fric_save)==0.5),
                  (np.array(fric_save)==0.6),
                  (np.array(fric_save)==0.7),
                  (np.array(fric_save)==0.8),
                  (np.array(fric_save)==0.9)]).T

# sample_idx = []
# maximum = np.max(input, axis=0)
# minimum = np.min(input, axis=0)
# n_div = 10
# delta = (maximum - minimum) / n_div
# dum0 = np.where((np.array(fric_save) == 0.5) + (np.array(fric_save) == 0.8))[0]
# for i1 in range(n_div):
#     print('i1 = ', i1)
#     nv = 0
#     dum1 = np.where(((minimum[nv] + i1 * delta[nv]) <= input[dum0,nv]) * (input[dum0,nv]< (minimum[nv] + (i1 + 1) * delta[nv])))[0]
#     for i2 in range(n_div):
#         nv = 1
#         dum2 = np.where(((minimum[nv] + i2 * delta[nv]) <= input[dum1,nv]) * (input[dum1,nv]< (minimum[nv] + (i2 + 1) * delta[nv])))[0]
#         dum2 = dum1[dum2]
#         for i3 in range(n_div):
#             nv = 2
#             dum3 = np.where(((minimum[nv] + i3 * delta[nv]) <= input[dum2, nv]) * (input[dum2, nv] < (minimum[nv] + (i3 + 1) * delta[nv])))[0]
#             dum3 = dum2[dum3]
#             for i4 in range(n_div):
#                 nv = 3
#                 dum4 = np.where(((minimum[nv] + i4 * delta[nv]) <= input[dum3, nv]) * (input[dum3, nv] < (minimum[nv] + (i4 + 1) * delta[nv])))[0]
#                 dum4 = dum3[dum4]
#                 for i5 in range(n_div):
#                     nv = 4
#                     dum5 = np.where(((minimum[nv] + i5 * delta[nv]) <= input[dum4, nv]) * (input[dum4, nv] < (minimum[nv] + (i5 + 1) * delta[nv])))[0]
#                     dum5 = dum4[dum5]
#                     for i6 in range(n_div):
#                         nv = 5
#                         dum6 = np.where(((minimum[nv] + i6 * delta[nv]) <= input[dum5, nv]) * (input[dum5, nv] < (minimum[nv] + (i6 + 1) * delta[nv])))[0]
#                         dum6 = dum5[dum6]
#                         for i7 in range(n_div):
#                             nv = 6
#                             dum7 = np.where(((minimum[nv] + i7 * delta[nv]) <= input[dum6, nv]) * (input[dum6, nv] < (minimum[nv] + (i7 + 1) * delta[nv])))[0]
#                             dum7 = dum6[dum7]
#                             try:
#                                 sample_idx.append(np.random.choice(dum7, 1))
#                             except:
#                                 pass
# sample_idx.append(np.where((np.array(fric_save) == 0.0)
#                            + (np.array(fric_save) == 0.1)
#                            + (np.array(fric_save) == 0.2)
#                            + (np.array(fric_save) == 0.3)
#                            + (np.array(fric_save) == 0.4)
#                            + (np.array(fric_save) == 0.6)
#                            + (np.array(fric_save) == 0.7)
#                            + (np.array(fric_save) == 0.9))[0])
# sample_idx = [x for xs in sample_idx for x in xs]
# sample_idx = np.unique(sample_idx)


# deletion = np.unique(np.hstack([np.where(np.array(x_vol_save) == 5000.0)[0],
#                                 np.where(np.array(y_vol_save) == 5000.0)[0],
#                                 np.where(np.array(fric_save) == 0.5)[0][np.random.permutation(len(np.where(np.array(fric_save) == 0.5)[0]))][:-1000],
#                                 np.where(np.array(fric_save) == 0.8)[0][np.random.permutation(len(np.where(np.array(fric_save) == 0.8)[0]))][:-1000]]))
#
# input2 = np.delete(input, deletion, axis=0)
# lable2 = np.delete(lable, deletion, axis=0)

# np.save('input.npy', input)
# np.save('lable.npy', lable)


# plt.hist(fric_save, bins=10)
# plt.grid()
# plt.xlabel('friction coefficient')
# plt.ylabel('# of data')
i = 1
plt.plot(fric_save[3600*i:3600*(i+1)], label='fric')
# plt.plot(rain_save[3600*i:3600*(i+1)], label='rain1')
# plt.plot(rain_save2[3600*i:3600*(i+1)], label='rain2')
plt.plot(raintf_save[3600*i:3600*(i+1)], label='raintf')
plt.plot(lastrain_save[3600*i:3600*(i+1)], label='last rain')
plt.legend()