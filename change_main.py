import torch
import numpy as np
import copy
from multiprocessing import Pool
import time
# from change_env_v2 import SchedulingEnv
import sys,os
# current_directory = os.getcwd()
# print(current_directory)
# result_path = os.path.join(current_directory, 'experiment4')
# print(result_path)
# sys.path.append(result_path)
from experiment4.change_env_v0_change_reward import SchedulingEnv
from mymodelDD_withpre import RL_model,baselines
from sum_tree import SumTree
from change_new_para import parameter_parser
import pickle
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args=parameter_parser()
#gen env
env = SchedulingEnv(args)
with open('train_env.pkl', 'rb') as file:
    tmp_env = pickle.load(file)
env.store_exeT=copy.deepcopy(tmp_env.store_exeT)
env.store_num=tmp_env.store_num
del tmp_env
print(f'store_exeT:{env.store_exeT},store_num:{env.store_num}')
#build model
RL_agent = RL_model(env.s_features, env.actionNum, args).to(device)
baseline_RL=RL_model(env.s_features, env.actionNum, args).to(device)
baseline_RL.Dqn.load_state_dict(torch.load('./experimentnoacc/resultRL/myRL_weights_global_step235500.pth'))
baseline_RL.target_Dqn.load_state_dict(torch.load('./experimentnoacc/resultRL/myRL_weights_global_step235500.pth'))
brainOthers = baselines(env.actionNum)
global_step = 0
my_learn_step = 0
better_rewards_RL=0
better_rewards_DQN=0
#result
x=[]
y1=[]
y2=[]
y3=[]
y4=[]
y5=[]
for episode in range(args.Epoch):
    print('----------------------------Episode', episode, '----------------------------')
    job_c = 1  # job counter
    performance_c = 0
    env.reset(args)  # attention: whether generate new workload, if yes, don't forget to modify reset() function
    while True:
        global_step += 1
        finish, job_attrs = env.workload(job_c)
        #greedy
        for node in range(args.Node_num):
            env.update_node(job_attrs,node,policyID=5)
        greedy_state=env.get_states(job_attrs,policyID=5)
        if global_step != 1:
            with torch.no_grad():
                q_value=RL_agent.Dqn(torch.FloatTensor(greedy_last_state).to(device))[greedy_last_action]
                q_nextvalue_idx=torch.argmax(RL_agent.Dqn(torch.FloatTensor(greedy_state).to(device)))
                q_nextvalue=RL_agent.target_Dqn(torch.FloatTensor(greedy_state).to(device))[q_nextvalue_idx]
                td_error=greedy_last_reward+RL_agent.gamma*q_nextvalue-q_value
            RL_agent.store_buffer(torch.pow(torch.abs(td_error),RL_agent.alpha).item(),greedy_last_state, greedy_last_action, greedy_last_reward, greedy_state)
        greedy_action=baseline_RL.choose_action_max(torch.FloatTensor(greedy_state))
        reward_greedy = env.feedback(job_attrs, greedy_action,policyID=5)
        greedy_last_state=greedy_state
        greedy_last_action=greedy_action
        greedy_last_reward=reward_greedy
        #myRL
        for node in range(args.Node_num):
            env.update_node(job_attrs,node,policyID=7)
        my_DQN_state=env.get_states(job_attrs)
        if global_step != 1:
            #computing td-error
            with torch.no_grad():
                q_value=RL_agent.Dqn(torch.FloatTensor(my_last_state).to(device))[my_last_action]
                q_nextvalue_idx=torch.argmax(RL_agent.Dqn(torch.FloatTensor(my_DQN_state).to(device)))
                q_nextvalue=RL_agent.target_Dqn(torch.FloatTensor(my_DQN_state).to(device))[q_nextvalue_idx]
                # print(f'q_value:{q_value},q_nextvalue_idx:{q_nextvalue_idx},q_nextvalue:{q_nextvalue}')
                td_error=my_last_reward+RL_agent.gamma*q_nextvalue-q_value
            RL_agent.store_buffer(torch.pow(torch.abs(td_error),RL_agent.alpha).item(),my_last_state, my_last_action, my_last_reward, my_DQN_state)
        if episode!=args.Epoch-1:
            myDQN_action=RL_agent.choose_action_max(torch.FloatTensor(my_DQN_state))
        else:
            myDQN_action=RL_agent.choose_action(torch.FloatTensor(my_DQN_state))
        my_reward_DQN = env.feedback(job_attrs, myDQN_action,policyID=7)
        # if (global_step > args.myRL_start_learn) and (global_step % args.myRL_learn_interval == 0) and episode!=args.Epoch-1:  # learn
        if global_step% RL_agent.batch_size ==0:
            RL_agent.learn()
        my_last_state=my_DQN_state
        my_last_action=myDQN_action
        my_last_reward=my_reward_DQN
        #random
        # for node in range(args.Node_num):
        #     env.update_node(job_attrs,node,policyID=1)
        # random_action=brainOthers.random_choose_action()
        # random_reward=env.feedback(job_attrs, random_action,policyID=1)
        #round robin
        # for node in range(args.Node_num):
        #     env.update_node(job_attrs,node,policyID=2)
        # roundrobin_action=brainOthers.RR_choose_action(job_attrs,env.roundrobin_Node_events[0],env.roundrobin_Node_events[1])
        # roundrobin_reward=env.feedback(job_attrs, roundrobin_action,policyID=2)
        if job_c % 500 == 0:
            acc_Rewards = env.get_accumulateRewards(performance_c, job_c)
            finishTs = env.get_FinishTimes(performance_c, job_c)
            avg_exeTs = env.get_executeTs(performance_c, job_c)
            avg_waitTs = env.get_waitTs(performance_c, job_c)
            avg_respTs = env.get_responseTs(performance_c, job_c)
            successTs = env.get_successTimes1(performance_c, job_c)
            print(f'myRL: start_job:{performance_c},end_job:{job_c}\nacc_Rewards:{acc_Rewards[0]}     \
                   finishTs:{finishTs[0]} avg_exeTs:{avg_exeTs[0]} avg_waitTs:{avg_waitTs[0]}          \
                   avg_respTs:{avg_respTs[0]} successTs:{successTs[0]}')
            # print(f'random: start_job:{performance_c},end_job:{job_c}\nacc_Rewards:{acc_Rewards[1]}     \
            #        finishTs:{finishTs[1]} avg_exeTs:{avg_exeTs[1]} avg_waitTs:{avg_waitTs[1]}          \
            #        avg_respTs:{avg_respTs[1]} successTs:{successTs[1]}')
            # print(f'RR: start_job:{performance_c},end_job:{job_c}\nacc_Rewards:{acc_Rewards[2]}     \
            #        finishTs:{finishTs[2]} avg_exeTs:{avg_exeTs[2]} avg_waitTs:{avg_waitTs[2]}          \
            #        avg_respTs:{avg_respTs[2]} successTs:{successTs[2]}')
            # print(f'DQN: start_job:{performance_c},end_job:{job_c}\nacc_Rewards:{acc_Rewards[3]}     \
            #        finishTs:{finishTs[3]} avg_exeTs:{avg_exeTs[3]} avg_waitTs:{avg_waitTs[3]}          \
            #        avg_respTs:{avg_respTs[3]} successTs:{successTs[3]}')
            print(f'greedy: start_job:{performance_c},end_job:{job_c}\nacc_Rewards:{acc_Rewards[5]}     \
                   finishTs:{finishTs[5]} avg_exeTs:{avg_exeTs[5]} avg_waitTs:{avg_waitTs[5]}          \
                   avg_respTs:{avg_respTs[5]} successTs:{successTs[5]}')
            #plot figure
            if episode==args.Epoch-1:
                x.append(job_c)
                y1.append(avg_respTs[0])
                y2.append(avg_respTs[4])
                y3.append(avg_respTs[2])
                y4.append(avg_respTs[3])
                # y5.append(avg_respTs[4])
            performance_c = job_c
            if better_rewards_RL<acc_Rewards[0] and episode>=50:
                # better_rewards_RL=acc_Rewards[0]
                torch.save(RL_agent.target_Dqn.state_dict(),f'./experimentnoacc/resultRL_self/myRL_weights_global_step{global_step}.pth')
        job_c+=1
        if finish:
            break
torch.save(RL_agent.target_Dqn.state_dict(),f'./experimentnoacc/resultRL_self/myRL_weights_global_step{global_step}.pth')
with open('train_env.pkl', 'wb') as file:
    pickle.dump(env, file)