import torch
import numpy as np
import copy
from multiprocessing import Pool
import time
from full_env import SchedulingEnv
from model import RL_model0,DDQN,DDuel_DQN,baselines
from sum_tree import SumTree
from para import parameter_parser
from simulateEnvTest import simulateEnvTest
import matplotlib.pyplot as plt
import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def smooth(y, radius, mode='two_sided', valid_only=False):
    '''
    Smooth signal y, where radius is determines the size of the window

    mode='twosided':
        average over the window [max(index - radius, 0), min(index + radius, len(y)-1)]
    mode='causal':
        average over the window [max(index - radius, 0), index]

    valid_only: put nan in entries where the full-sized window is not available

    '''
    assert mode in ('two_sided', 'causal')
    if len(y) < 2*radius+1:
        return np.ones_like(y) * y.mean()
    elif mode == 'two_sided':
        convkernel = np.ones(2 * radius+1)
        out = np.convolve(y, convkernel,mode='same') / np.convolve(np.ones_like(y), convkernel, mode='same')
        if valid_only:
            out[:radius] = out[-radius:] = np.nan
    elif mode == 'causal':
        convkernel = np.ones(radius)
        out = np.convolve(y, convkernel,mode='full') / np.convolve(np.ones_like(y), convkernel, mode='full')
        out = out[:-radius+1]
        if valid_only:
            out[:radius] = np.nan
    return out

args=parameter_parser()
#gen env
env = SchedulingEnv(args)
#build model
RL_agent = RL_model0(env.s_features, env.actionNum, args).to(device)
DDQN_agent= DDQN(env.s_features, env.actionNum, args).to(device)
DDuel_DQN_agent=DDuel_DQN(env.s_features, env.actionNum, args).to(device)
DDuelDQN_pri_agent=RL_model0(env.s_features, env.actionNum, args).to(device)
brainOthers = baselines(env.actionNum)
global_step = 0
my_learn_step = 0
old_RL_grade=0
#result
df = pd.DataFrame(columns=['global_step','RL_agent','DDQN_agent','DDuel_DQN_agent','DDuelDQN_pri_agent'])#列名
df.to_csv("./experiment4/result/node4_acc/simulateEnvTestLog.csv",index=False) #路径可以根据需要更改
for episode in range(args.Epoch):
    print('----------------------------Episode', episode, '----------------------------')
    job_c = 1  # job counter
    performance_c = 0
    env.reset()  # attention: whether generate new workload, if yes, don't forget to modify reset() function
    while True:
        global_step += 1
        finish, job_attrs = env.workload(job_c)
        #greedy with resposeT
        for node in range(args.Node_num):
            env.update_node(job_attrs,node,policyID=5)
        greedy_resposeT_state=env.get_states(job_attrs,policyID=5)
        if global_step != 1:
            with torch.no_grad():
                q_value=RL_agent.Dqn(torch.FloatTensor(greedy_resposeT_last_state).to(device))[greedy_resposeT_last_action]
                q_nextvalue_idx=torch.argmax(RL_agent.Dqn(torch.FloatTensor(greedy_resposeT_state).to(device)))
                q_nextvalue=RL_agent.target_Dqn(torch.FloatTensor(greedy_resposeT_state).to(device))[q_nextvalue_idx]
                td_error=greedy_resposeT_last_reward+RL_agent.gamma*q_nextvalue-q_value
            RL_agent.store_buffer(torch.pow(torch.abs(td_error),RL_agent.alpha).item(),greedy_resposeT_last_state, greedy_resposeT_last_action, greedy_resposeT_last_reward, greedy_resposeT_state)
        greedy_resposeT_action=brainOthers.greedy_choose_action_respose(greedy_resposeT_state)
        reward_greedy_resposeT = env.feedback(job_attrs, greedy_resposeT_action,policyID=5)
        greedy_resposeT_last_state=greedy_resposeT_state
        greedy_resposeT_last_action=greedy_resposeT_action
        greedy_resposeT_last_reward=reward_greedy_resposeT
        #greedy with idleT
        for node in range(args.Node_num):
            env.update_node(job_attrs,node,policyID=4)
        greedy_idleT_state=env.get_states(job_attrs,policyID=4)
        #store experience replay
        # save_greedy_idleT_state=env.get_states(job_attrs,policyID=7)
        # if global_step != 1:
        #     with torch.no_grad():
        #         q_value=RL_agent.Dqn(torch.FloatTensor(greedy_idleT_last_state).to(device))[greedy_idleT_last_action]
        #         q_nextvalue_idx=torch.argmax(RL_agent.Dqn(torch.FloatTensor(save_greedy_idleT_state).to(device)))
        #         q_nextvalue=RL_agent.target_Dqn(torch.FloatTensor(save_greedy_idleT_state).to(device))[q_nextvalue_idx]
        #         td_error=greedy_idleT_last_reward+RL_agent.gamma*q_nextvalue-q_value
        #     RL_agent.store_buffer(torch.pow(torch.abs(td_error),RL_agent.alpha).item(),greedy_idleT_last_state, greedy_idleT_last_action, greedy_idleT_last_reward, save_greedy_idleT_state)
        greedy_idle_action=brainOthers.greedy_choose_action_wait(greedy_idleT_state)
        reward_greedy_idle = env.feedback(job_attrs, greedy_idle_action,policyID=4)
        #
        # greedy_idleT_last_state=save_greedy_idleT_state
        # greedy_idleT_last_action=greedy_idle_action
        # greedy_idleT_last_reward=reward_greedy_idle
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
                td_error=my_last_reward+RL_agent.gamma*q_nextvalue-q_value
            RL_agent.store_buffer(torch.pow(torch.abs(td_error),RL_agent.alpha).item(),my_last_state, my_last_action, my_last_reward, my_DQN_state)
        myDQN_action=RL_agent.choose_action(torch.FloatTensor(my_DQN_state))
        my_reward_DQN = env.feedback(job_attrs, myDQN_action,policyID=7)
        # if (global_step > args.myRL_start_learn) and (global_step % args.myRL_learn_interval == 0):  # learn
        # if global_step% RL_agent.batch_size ==0:
        #     RL_agent.learn()
        my_last_state=my_DQN_state
        my_last_action=myDQN_action
        my_last_reward=my_reward_DQN
        #DDQN
        for node in range(args.Node_num):
            env.update_node(job_attrs,node,policyID=1)
        DDQN_state=env.get_states(job_attrs,policyID=1)
        if global_step != 1:
            DDQN_agent.store_buffer(DDQN_last_state, DDQN_last_action, DDQN_last_reward, DDQN_state)
        DDQN_action=DDQN_agent.choose_action(torch.FloatTensor(DDQN_state))
        DDQN_reward = env.feedback(job_attrs, DDQN_action,policyID=1)
        # if (global_step > args.myRL_start_learn) and (global_step % args.myRL_learn_interval == 0):  # learn
        # # if global_step% DDQN_agent.batch_size ==0:
        #     DDQN_agent.learn()
        DDQN_last_state=DDQN_state
        DDQN_last_action=DDQN_action
        DDQN_last_reward=DDQN_reward
        #DDuel-DQN
        for node in range(args.Node_num):
            env.update_node(job_attrs,node,policyID=2)
        DDuelDQN_state=env.get_states(job_attrs,policyID=2)
        if global_step != 1:
            DDuel_DQN_agent.store_buffer(DDuelDQN_last_state, DDuelDQN_last_action, DDuelDQN_last_reward, DDuelDQN_state)
        DDuelDQN_action=DDuel_DQN_agent.choose_action(torch.FloatTensor(DDuelDQN_state))
        DDuelDQN_reward = env.feedback(job_attrs, DDuelDQN_action,policyID=2)
        # if (global_step > args.myRL_start_learn) and (global_step % args.myRL_learn_interval == 0):  # learn
        # # if global_step% DDuel_DQN_agent.batch_size ==0:
        #     DDuel_DQN_agent.learn()
        DDuelDQN_last_state=DDuelDQN_state
        DDuelDQN_last_action=DDuelDQN_action
        DDuelDQN_last_reward=DDuelDQN_reward
        #D Duel-DQN with priority
        for node in range(args.Node_num):
            env.update_node(job_attrs,node,policyID=3)
        DDuelDQN_pri_state=env.get_states(job_attrs,policyID=3)
        if global_step != 1:
            #computing td-error
            with torch.no_grad():
                q_value=DDuelDQN_pri_agent.Dqn(torch.FloatTensor(DDuelDQN_pri_last_state).to(device))[DDuelDQN_pri_last_action]
                q_nextvalue_idx=torch.argmax(DDuelDQN_pri_agent.Dqn(torch.FloatTensor(DDuelDQN_pri_state).to(device)))
                q_nextvalue=DDuelDQN_pri_agent.target_Dqn(torch.FloatTensor(DDuelDQN_pri_state).to(device))[q_nextvalue_idx]
                td_error=my_last_reward+DDuelDQN_pri_agent.gamma*q_nextvalue-q_value
            DDuelDQN_pri_agent.store_buffer(torch.pow(torch.abs(td_error),DDuelDQN_pri_agent.alpha).item(),DDuelDQN_pri_last_state, DDuelDQN_pri_last_action, DDuelDQN_pri_last_reward, DDuelDQN_pri_state)
        DDuelDQN_pri_action=DDuelDQN_pri_agent.choose_action(torch.FloatTensor(DDuelDQN_pri_state))
        DDuelDQN_pri_reward= env.feedback(job_attrs, DDuelDQN_pri_action,policyID=3)
        # if (global_step > args.myRL_start_learn) and (global_step % args.myRL_learn_interval == 0):  # learn
        # # if global_step% DDuelDQN_pri_agent.batch_size ==0:
        #     DDuelDQN_pri_agent.learn()
        DDuelDQN_pri_last_state=DDuelDQN_pri_state
        DDuelDQN_pri_last_action=DDuelDQN_pri_action
        DDuelDQN_pri_last_reward=DDuelDQN_pri_reward
        #uniform learn
        if (global_step > args.myRL_start_learn) and (global_step % args.myRL_learn_interval == 0):
            RL_agent.learn(env)
            DDQN_agent.learn(env)
            DDuel_DQN_agent.learn(env)
            DDuelDQN_pri_agent.learn(env)
            grade_RL_agent,grade_DDQN_agent,grade_DDuel_DQN_agent,grade_DDuelDQN_pri_agent=simulateEnvTest(env,brainOthers,RL_agent,DDQN_agent,DDuel_DQN_agent,DDuelDQN_pri_agent)
            list = [global_step,grade_RL_agent,grade_DDQN_agent,grade_DDuel_DQN_agent,grade_DDuelDQN_pri_agent]
            data = pd.DataFrame([list])
            data.to_csv('./experiment4/result/node4_acc/simulateEnvTestLog.csv',mode='a',header=False,index=False)#mode设为a,就可以向csv文件追加数据了
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
            print(f'DDQN: start_job:{performance_c},end_job:{job_c}\nacc_Rewards:{acc_Rewards[1]}     \
                   finishTs:{finishTs[1]} avg_exeTs:{avg_exeTs[1]} avg_waitTs:{avg_waitTs[1]}          \
                   avg_respTs:{avg_respTs[1]} successTs:{successTs[1]}')
            print(f'DDuel-DQN: start_job:{performance_c},end_job:{job_c}\nacc_Rewards:{acc_Rewards[2]}     \
                   finishTs:{finishTs[2]} avg_exeTs:{avg_exeTs[2]} avg_waitTs:{avg_waitTs[2]}          \
                   avg_respTs:{avg_respTs[2]} successTs:{successTs[2]}')
            print(f'DDuel-DQN with priority: start_job:{performance_c},end_job:{job_c}\nacc_Rewards:{acc_Rewards[3]}     \
                   finishTs:{finishTs[3]} avg_exeTs:{avg_exeTs[3]} avg_waitTs:{avg_waitTs[3]}          \
                   avg_respTs:{avg_respTs[3]} successTs:{successTs[3]}')
            print(f'greedy with idleT: start_job:{performance_c},end_job:{job_c}\nacc_Rewards:{acc_Rewards[4]}     \
                   finishTs:{finishTs[4]} avg_exeTs:{avg_exeTs[4]} avg_waitTs:{avg_waitTs[4]}          \
                   avg_respTs:{avg_respTs[4]} successTs:{successTs[4]}')
            print(f'greedy  with resposeT: start_job:{performance_c},end_job:{job_c}\nacc_Rewards:{acc_Rewards[5]}     \
                   finishTs:{finishTs[5]} avg_exeTs:{avg_exeTs[5]} avg_waitTs:{avg_waitTs[5]}          \
                   avg_respTs:{avg_respTs[5]} successTs:{successTs[5]}')
        job_c+=1
        if finish:
            break
