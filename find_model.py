import torch
import numpy as np
import copy
import time
from full_env import SchedulingEnv
from model import RL_model0,baselines
from sum_tree import SumTree
from para import parameter_parser
import matplotlib.pyplot as plt
import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
args=parameter_parser()
#gen env
env = SchedulingEnv(args)
#build model
RL_agent = RL_model0(env.s_features, env.actionNum, args).to(device)
brainOthers = baselines(env.actionNum)
init_model=200500
fin_model=250500
#最好的模型是210140，已经遍历到234000
best_model=200100
best_reward=8805.47
#创建列
global_step = 0
df = pd.DataFrame(columns=['algorithm','model','avg_responseT','acc_rewards','greedy_response'])#列名
# df.to_csv("./experiment4/result/node4_acc/finalRL.csv",index=False) #路径可以根据需要更改
while init_model<=fin_model:
    RL_agent.Dqn.load_state_dict(torch.load('./experiment4/resultTest/myRL_weights_global_step'+str(init_model)+'.pth'))
    RL_agent.target_Dqn.load_state_dict(torch.load('./experiment4/resultTest/myRL_weights_global_step'+str(init_model)+'.pth'))
    print('----------------------------Episode start'+str(init_model)+ '----------------------------')
    for episode in range(5):
        job_c = 1  # job counter
        performance_c = 0
        average_responses=0
        env.reset()  # attention: whether generate new workload, if yes, don't forget to modify reset() function
        while True:
            global_step += 1
            finish, job_attrs = env.workload(job_c)
            #greedy with resposeT
            for node in range(args.Node_num):
                env.update_node(job_attrs,node,policyID=5)
            greedy_state=env.get_states(job_attrs,policyID=5)
            greedy_action=brainOthers.greedy_choose_action_respose(greedy_state)
            reward_greedy = env.feedback(job_attrs, greedy_action,policyID=5)
            #greedy with idleT
            for node in range(args.Node_num):
                env.update_node(job_attrs,node,policyID=4)
            greedy_state=env.get_states(job_attrs,policyID=4)
            greedy_action=brainOthers.greedy_choose_action_wait(greedy_state)
            reward_greedy = env.feedback(job_attrs, greedy_action,policyID=4)
            for node in range(args.Node_num):
                env.update_node(job_attrs,node,policyID=7)
            my_DQN_state=env.get_states(job_attrs)
            myDQN_action=RL_agent.choose_action_max(torch.FloatTensor(my_DQN_state))
            my_reward_DQN = env.feedback(job_attrs, myDQN_action,policyID=7)
            if finish:
                avg_respTs = env.get_responseTs(performance_c, job_c)
                finishTs = env.get_FinishTimes(performance_c, job_c)
                avg_exeTs = env.get_executeTs(performance_c, job_c)
                avg_waitTs = env.get_waitTs(performance_c, job_c)
                average_responses+=avg_respTs[0]
                break
            job_c+=1
    average_responses=average_responses/5
    if(best_reward>average_responses):
        best_reward=average_responses
        best_model=init_model
    print(f'best model:{best_model},average_rewards:{average_responses}')
    init_model+=1000























            
