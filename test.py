import torch
import numpy as np
from full_env import SchedulingEnv
from model import RL_model0,baselines
from sum_tree import SumTree
from para import parameter_parser
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from datetime import datetime
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def plot_figure_acc_reward(x,y1,y2,y3,y4,x_name,y_name):
    plt.title(y_name)  # 添加图片标题
    plt.ion()
    plt.xlabel(x_name)#x轴上的名字
    plt.ylabel(y_name)#y轴上的名字
    plt.plot(x,y1,'red',x,y2,'yellow',x,y3,'blue',x,y4,'black',marker='*')
    plt.legend(["myRL", "RR","greedy_idleT","greedy_responseT"], loc='lower right')  # loc='upper left'
    plt.savefig(y_name)
    plt.ioff()
    plt.close()
def plot_figure_avg(x,y1,y2,y3,y4,x_name,y_name):
    plt.title(y_name)  # 添加图片标题
    plt.ion()
    plt.xlabel(x_name)#x轴上的名字
    plt.ylabel(y_name)#y轴上的名字
    plt.plot(x,y1,'red',x,y2,'yellow',x,y3,'blue',marker='*')
    plt.legend(["myRL", "greedy","RR"], loc='lower right')  # loc='upper left'
    plt.savefig(y_name)
    plt.ioff()
    plt.close()
args=parameter_parser()
#gen env
env = SchedulingEnv(args)
#build model
RL_agent = RL_model0(env.s_features, env.actionNum, args).to(device)
RL_agent.Dqn.load_state_dict(torch.load('./experiment4/resultRL/myRL_weights_global_step34500.pth'))
RL_agent.target_Dqn.load_state_dict(torch.load('./experiment4/resultRL/myRL_weights_global_step34500.pth'))
RL_agent1 = RL_model0(env.s_features, env.actionNum, args).to(device)
RL_agent1.Dqn.load_state_dict(torch.load('./experiment4/resultreward_wait/myRL_weights_global_step37000.pth'))
RL_agent1.target_Dqn.load_state_dict(torch.load('./experiment4/resultreward_wait/myRL_weights_global_step37000.pth'))
RL_agent2= RL_model0(env.s_features, env.actionNum, args).to(device)
RL_agent2.Dqn.load_state_dict(torch.load('./experiment4/resultreward_slowdown/myRL_weights_global_step240500.pth'))
RL_agent2.target_Dqn.load_state_dict(torch.load('./experiment4/resultreward_slowdown/myRL_weights_global_step240500.pth'))
brainOthers = baselines(env.actionNum)
#创建列
# df = pd.DataFrame(columns=['fin_job','avg_responseT','finT','acc_reward'])#列名
# df.to_csv("./result/node4_acc/noclip_RLlog.csv",index=False) #路径可以根据需要更改
# df.to_csv("./result/node4_acc/noclip_RRlog.csv",index=False) #路径可以根据需要更改
# df.to_csv("./result/node4_acc/noclip_greedy_waitlog.csv",index=False) #路径可以根据需要更改
# df.to_csv("./result/node4_acc/noclip_greedy_responselog.csv",index=False) #路径可以根据需要更改
df = pd.DataFrame(columns=['algorithm','lamda','avg_responseT'])#列名
# df.to_csv("./experiment4/result/node4_acc/EvaulationMethodLog2.csv",index=False) #路径可以根据需要更改
global_step = 0
# result
x=[]
y_low=(int)(args.Job_Num/500)
#avg_responceT
yR1=np.zeros((5,y_low))
yR2=np.zeros((5,y_low))
yR3=np.zeros((5,y_low))
yR4=np.zeros((5,y_low))
yR5=np.zeros((5,y_low))
# yR1=np.zeros(5)
# yR2=np.zeros(5)
# yR3=np.zeros(5)
# yR4=np.zeros(5)
# yR5=np.zeros(5)
for episode in range(5):
    print('----------------------------Episode----------------------------')
    job_c = 1  # job counter
    performance_c = 0
    y_lowidx=0
    env.reset()  # attention: whether generate new workload, if yes, don't forget to modify reset() function
    while True:
        global_step += 1
        finish, job_attrs = env.workload(job_c)
        #greedy with resposeT
        for node in range(args.Node_num):
            env.update_node(job_attrs,node,policyID=5)
        # thoughout_test=env.success_num[5]/job_attrs[1]
        greedy_state=env.get_states(job_attrs,policyID=5)
        greedy_action=brainOthers.greedy_choose_action_respose(greedy_state)
        reward_greedy = env.feedback(job_attrs, greedy_action,policyID=5)
        # thoughout_test1=env.get_thoughout(job_c)[5]
        # print(f'thoughout_test:{thoughout_test},thoughout_test1:{thoughout_test1}')
        #greedy with idleT
        for node in range(args.Node_num):
            env.update_node(job_attrs,node,policyID=4)
        greedy_state=env.get_states(job_attrs,policyID=4)
        greedy_action=brainOthers.greedy_choose_action_wait(greedy_state)
        reward_greedy = env.feedback(job_attrs, greedy_action,policyID=4)
        #random
        for node in range(args.Node_num):
            env.update_node(job_attrs,node,policyID=1)
        random_action=brainOthers.greedy_action(job_attrs,env.DDQN_Node_events[0],env.DDQN_Node_events[1])
        random_reward=env.feedback(job_attrs, random_action,policyID=1)
        # myRL   reward=avg_responseT
        for node in range(args.Node_num):
            env.update_node(job_attrs,node,policyID=7)
        my_DQN_state=env.get_states(job_attrs)
        myDQN_action=RL_agent.choose_action_max(torch.FloatTensor(my_DQN_state))
        my_reward_DQN = env.feedback(job_attrs, myDQN_action,policyID=7)
        #myRL reward=avg_waitT
        # for node in range(args.Node_num):
        #     env.update_node(job_attrs,node,policyID=1)
        # my_DQN_state1=env.get_states(job_attrs,policyID=1)
        # myDQN_action1=RL_agent1.choose_action_max(torch.FloatTensor(my_DQN_state1))
        # my_reward_DQN1 = env.feedback(job_attrs, myDQN_action1,policyID=1)
        #myRL reward=slow_down
        for node in range(args.Node_num):
            env.update_node(job_attrs,node,policyID=2)
        my_DQN_state2=env.get_states(job_attrs,policyID=2)
        myDQN_action2=RL_agent2.choose_action_max(torch.FloatTensor(my_DQN_state2))
        my_reward_DQN2 = env.feedback(job_attrs, myDQN_action2,policyID=2)
        #round robin
        # for node in range(args.Node_num):
        #     env.update_node(job_attrs,node,policyID=2)
        # roundrobin_action=brainOthers.RR_choose_action(job_attrs,env.DDuelDQN_Node_events[0],env.DDuelDQN_Node_events[1])
        # roundrobin_reward=env.feedback(job_attrs, roundrobin_action,policyID=2)
        if job_c % 500 == 0:
            acc_Rewards = env.get_accumulateRewards(performance_c, job_c)
            finishTs = env.get_FinishTimes(performance_c, job_c)
            avg_exeTs = env.get_executeTs(performance_c, job_c)
            avg_waitTs = env.get_waitTs(performance_c, job_c)
            avg_respTs = env.get_responseTs(performance_c, job_c)
            # successTs2 = env.get_successTimes1(performance_c, job_c)
            print(f'myRL: start_job:{performance_c},end_job:{job_c}\nacc_Rewards:{acc_Rewards[0]}     \
                    finishTs:{finishTs[0]} avg_exeTs:{avg_exeTs[0]} avg_waitTs:{avg_waitTs[0]}          \
                    avg_respTs:{avg_respTs[0]} ')
            # print(f'random: start_job:{performance_c},end_job:{job_c}\nacc_Rewards:{acc_Rewards[1]}     \
            #         finishTs:{finishTs[1]} avg_exeTs:{avg_exeTs[1]} avg_waitTs:{avg_waitTs[1]}          \
            #         avg_respTs:{avg_respTs[1]} successTs:{successTs[1]}')
            # print(f'RR: start_job:{performance_c},end_job:{job_c}\nacc_Rewards:{acc_Rewards[2]}     \
            #         finishTs:{finishTs[2]} avg_exeTs:{avg_exeTs[2]} avg_waitTs:{avg_waitTs[2]}          \
            #         avg_respTs:{avg_respTs[2]} successTs:{successTs1[2]}')
            # print(f'DQN: start_job:{performance_c},end_job:{job_c}\nacc_Rewards:{acc_Rewards[3]}     \
            #         finishTs:{finishTs[3]} avg_exeTs:{avg_exeTs[3]} avg_waitTs:{avg_waitTs[3]}          \
            #         avg_respTs:{avg_respTs[3]} successTs:{successTs[3]}')
            print(f'greedy with idleT: start_job:{performance_c},end_job:{job_c}\nacc_Rewards:{acc_Rewards[4]}     \
                   finishTs:{finishTs[4]} avg_exeTs:{avg_exeTs[4]} avg_waitTs:{avg_waitTs[4]}          \
                   avg_respTs:{avg_respTs[4]} ')
            print(f'greedy  with resposeT: start_job:{performance_c},end_job:{job_c}\nacc_Rewards:{acc_Rewards[5]}     \
                   finishTs:{finishTs[5]} avg_exeTs:{avg_exeTs[5]} avg_waitTs:{avg_waitTs[5]}          \
                   avg_respTs:{avg_respTs[5]} ')
            # plot figure
            if episode==0:
                x.append(job_c)
            yR1[episode,y_lowidx]=avg_respTs[0]
            yR2[episode,y_lowidx]=avg_respTs[2]
            yR3[episode,y_lowidx]=avg_respTs[4]
            yR4[episode,y_lowidx]=avg_respTs[5]
            yR5[episode,y_lowidx]=avg_respTs[1]
            y_lowidx+=1
            # performance_c = job_c
        if finish:
            avg_respTs = env.get_responseTs(performance_c, job_c)
            finishTs = env.get_FinishTimes(performance_c, job_c)
            avg_exeTs = env.get_executeTs(performance_c, job_c)
            avg_waitTs = env.get_waitTs(performance_c, job_c)
            successTs1 = env.get_successTimes1(performance_c, job_c)
            successTs2 = env.get_successTimes2(performance_c, job_c)
            # avg_slowdown = env.get_avg_slowdown(performance_c, job_c)
            # yR1[episode]=avg_respTs[0]
            # yR2[episode]=avg_respTs[2]
            # yR3[episode]=avg_respTs[4]
            # yR4[episode]=avg_respTs[5]
            # yR5[episode]=avg_respTs[1]
            break
        job_c+=1
#responseT
# RL_responseT=np.mean(yR1)
# RR_responseT=np.mean(yR2)
# greedy_wait_responseT=np.mean(yR3)
# greedy_response_responseT=np.mean(yR4)
# Random_responseT=np.mean(yR5)
# columns=['algorithm','lamda','avg_responseT']
# listRL = ['RL',env.lamda,RL_responseT]
# data = pd.DataFrame([listRL])
# data.to_csv('./experiment4/result/node4_acc/EvaulationMethodLog2.csv',mode='a',header=False,index=False)#mode设为a,就可以向csv文件追加数据了
# listrandom = ['RLavg_wait_responseT',env.lamda,Random_responseT]
# data = pd.DataFrame([listrandom])
# data.to_csv('./experiment4/result/node4_acc/EvaulationMethodLog2.csv',mode='a',header=False,index=False)#mode设为a,就可以向csv文件追加数据了
# listRR = ['RLslow_down_responseT',env.lamda,RR_responseT]
# data = pd.DataFrame([listRR])
# data.to_csv('./experiment4/result/node4_acc/EvaulationMethodLog2.csv',mode='a',header=False,index=False)#mode设为a,就可以向csv文件追加数据了
# listgreedy_wait = ['greedy_wait',env.lamda,greedy_wait_responseT]
# data = pd.DataFrame([listgreedy_wait])
# data.to_csv('./experiment4/result/node4_acc/EvaulationMethodLog2.csv',mode='a',header=False,index=False)#mode设为a,就可以向csv文件追加数据了
# listgreedy_response = ['greedy_response',env.lamda,greedy_response_responseT]
# data = pd.DataFrame([listgreedy_response])
# data.to_csv('./experiment4/result/node4_acc/EvaulationMethodLog2.csv',mode='a',header=False,index=False)#mode设为a,就可以向csv文件追加数据了
#plot
RL_responseT=np.mean(yR1,axis=0)
RLslow_down_responseT=np.mean(yR2,axis=0)
Greedy_wait_responseT=np.mean(yR3,axis=0)
Greedy_resT_responseT=np.mean(yR4,axis=0)
RLavg_wait_responseT=np.mean(yR5,axis=0)
#save data
# for i in range(len(x)):
#     listRL = [x[i],RL_responseT[i],RL_finT[i],RL_acc_reward[i]]
#     data = pd.DataFrame([listRL])
#     data.to_csv('./result/node4_acc/noclip_RLlog.csv',mode='a',header=False,index=False)#mode设为a,就可以向csv文件追加数据了
#     listRR = [x[i],RR_responseT[i],RR_finT[i],RR_acc_reward[i]]
#     data = pd.DataFrame([listRR])
#     data.to_csv('./result/node4_acc/noclip_RRlog.csv',mode='a',header=False,index=False)#mode设为a,就可以向csv文件追加数据了\
#     listgreedy_wait=[x[i],Greedy_wait_responseT[i],Greedy_wait_finT[i],Greedy_wait_acc_reward[i]]
#     data = pd.DataFrame([listgreedy_wait])
#     data.to_csv('./result/node4_acc/noclip_greedy_waitlog.csv',mode='a',header=False,index=False)#mode设为a,就可以向csv文件追加数据了
#     listgreedy_response = [x[i],Greedy_resT_responseT[i],Greedy_resT_finT[i],Greedy_resT_acc_reward[i]]
#     data = pd.DataFrame([listgreedy_response])
#     data.to_csv('./result/node4_acc/noclip_greedy_responselog.csv',mode='a',header=False,index=False)#mode设为a,就可以向csv文件追加数据了
plt.plot(x,RL_responseT,label="RL_avg_response",color='red',marker='*')
plt.plot(x,Greedy_wait_responseT,label="Greedy_wait",marker='*')
plt.plot(x,Greedy_resT_responseT,label="Greedy_response",marker='*')
plt.plot(x,RLslow_down_responseT,label="RL_slow_down",marker='*')
plt.plot(x,RLavg_wait_responseT,label="RL_avg_wait",marker='*')
plt.title("Average Response Time") 
plt.xlabel('job_num')
plt.ylabel('avg_responseT')
plt.legend()   #显示标签
plt.savefig("./experiment4/resultshow/final_node4_avg_responseT", dpi=300)
