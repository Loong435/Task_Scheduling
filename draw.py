import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
# RRdata=pd.read_csv("./experiment4/result/node4_acc/noclip_RRlog.csv")
# greedy_waitdata=pd.read_csv("./experiment4/result/node4_acc/noclip_greedy_waitlog.csv")
# greedy_responsedata=pd.read_csv("./experiment4/result/node4_acc/noclip_greedy_responselog.csv")
# RLdata=pd.read_csv("./experiment4/result/node4_acc/noclip_RLlog.csv")
# job_num=RLdata[['fin_job']]
# #responseT
# # RRresponseT=RRdata[['avg_responseT']]
# # greedy_wait_responseT=greedy_waitdata[['avg_responseT']]
# # greedy_response_responseT=greedy_responsedata[['avg_responseT']]
# # RLresponseT=RLdata[['avg_responseT']]
# #acc_reward
# RRacc_reward=RRdata[['acc_reward']]
# greedy_wait_acc_reward=greedy_waitdata[['acc_reward']]
# greedy_response_acc_reward=greedy_responsedata[['acc_reward']]
# RLacc_reward=RLdata[['acc_reward']]
# x=np.array(job_num)
# # y1=np.array(RRresponseT)
# y1=np.array(RLacc_reward)
# y2=np.array(greedy_wait_acc_reward)
# y3=np.array(greedy_response_acc_reward)
# y4=np.array(RRacc_reward)
# plt.plot(x,y1,label="RL",color='red',marker='*')
# plt.plot(x,y2,label="greedy_wait",marker='*')
# plt.plot(x,y3,label="greedy_response",marker='*')
# plt.plot(x,y4,label="RR",color='blue',marker='*')
# plt.title("acc_reward") 
# plt.xlabel('job_num')
# plt.ylabel('acc_reward')
# plt.legend()   #显示标签
# plt.savefig("./experiment4/resultshow/noclip_node4_acc_reward")



#消融实验模拟仿真图
data=pd.read_csv("./experiment4/result/node4_acc/simulateEnvTestLog.csv")
globalStep=data[['global_step']]
RL_agent_grade=data[['RL_agent']]
DDQN_agent_grade=data[['DDQN_agent']]
DDuel_DQN_agent_grade=data[['DDuel_DQN_agent']]
DDuelDQN_pri_agent_grade=data[['DDuelDQN_pri_agent']]
x=np.array(globalStep)
y1=np.array(RL_agent_grade)
#数据Savitzky-Golay滤波器 进行平滑处理
y1=scipy.signal.savgol_filter(y1.reshape(-1),53,3)
y2=np.array(DDQN_agent_grade)
y2=scipy.signal.savgol_filter(y2.reshape(-1),53,3)
y3=np.array(DDuel_DQN_agent_grade)
y3=scipy.signal.savgol_filter(y3.reshape(-1),53,3)
y4=np.array(DDuelDQN_pri_agent_grade)
y4=scipy.signal.savgol_filter(y4.reshape(-1),53,3)
plt.plot(x,y1,label="RL",color='red')
plt.plot(x,y2,label="DDQN")
plt.plot(x,y3,label="D3QN")
plt.plot(x,y4,label="D3QN_PER",color='blue')
plt.title("acc_reward") 
plt.xlabel('steps')
plt.ylabel('acc_reward')
plt.legend()   #显示标签
plt.savefig("./experiment4/resultshow/testSimulateReward1", dpi=300 )





#固定任务数目，改变负载时的平均响应时间图
# data=pd.read_csv("./experiment4/result/node4_acc/EvaulationMethodLog2.csv")
# data=np.array(data)
# x=np.array([10,11,12,13])
# RL_avg_responseT=np.zeros(len(x))
# RLavg_wait_responseT=np.zeros(len(x))
# RLslow_down_responseT=np.zeros(len(x))
# greedy_wait_avg_responseT=np.zeros(len(x))
# greedy_response_avg_responseT=np.zeros(len(x))
# RL_index=0
# RR_index=0
# Random_index=0
# greedy_wait_index=0
# greedy_response_index=0
# for i in range(len(data)):
#     if data[i,0]=='RL':
#         RL_avg_responseT[RL_index]=data[i,2]
#         RL_index+=1
#     elif data[i,0]=='RLavg_wait_responseT':
#         RLavg_wait_responseT[RR_index]=data[i,2]
#         RR_index+=1
#     elif data[i,0]=='RLslow_down_responseT':
#         RLslow_down_responseT[Random_index]=data[i,2]
#         Random_index+=1
#     elif data[i,0]=='greedy_wait':
#         greedy_wait_avg_responseT[greedy_wait_index]=data[i,2]
#         greedy_wait_index+=1
#     elif data[i,0]=='greedy_response':
#         greedy_response_avg_responseT[greedy_response_index]=data[i,2]
#         greedy_response_index+=1
# plt.plot(x,RL_avg_responseT,label="RL_avg_response",color='red',marker='*')
# plt.plot(x,greedy_wait_avg_responseT,label="Greedy_wait",marker='*')
# plt.plot(x,greedy_response_avg_responseT,label="Greedy_response",marker='*')
# plt.plot(x,RLslow_down_responseT,label="RL_slow_down",marker='*')
# plt.plot(x,RLavg_wait_responseT,label="RL_avg_wait",marker='*')
# plt.title("Average Response Time") 
# plt.xlabel('Mean Arrival Rate')
# plt.ylabel('avg_responseT')
# plt.legend()   #显示标签
# plt.savefig("./experiment4/resultshow/EvaulationMethodLog2_avg_responseT", dpi=300)
