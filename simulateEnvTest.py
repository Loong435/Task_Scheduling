import copy
import torch
def simulateEnvTest(env,brainOthers,RL_agent,DDQN_agent,DDuel_DQN_agent,DDuelDQN_pri_agent):
    simulate_env=copy.deepcopy(env)
    simulate_env.reset()
    job_c = 1  # job counter
    global_step = 0
    grade_RL_agent=0
    grade_DDQN_agent=0
    grade_DDuel_DQN_agent=0
    grade_DDuelDQN_pri_agent=0
    while True:
        global_step += 1
        finish, job_attrs = simulate_env.workload(job_c)
        #greedy with resposeT
        for node in range(simulate_env.Node_num):
            simulate_env.update_node(job_attrs,node,policyID=5)
        greedy_resposeT_state=simulate_env.get_states(job_attrs,policyID=5)
        greedy_resposeT_action=brainOthers.greedy_choose_action_respose(greedy_resposeT_state)
        reward_greedy_resposeT = simulate_env.feedback(job_attrs, greedy_resposeT_action,policyID=5)
        #myRL
        for node in range(simulate_env.Node_num):
            simulate_env.update_node(job_attrs,node,policyID=7)
        my_DQN_state=simulate_env.get_states(job_attrs)
        myDQN_action=RL_agent.choose_action_max(torch.FloatTensor(my_DQN_state))
        my_reward_DQN = simulate_env.feedback(job_attrs, myDQN_action,policyID=7)
        grade_RL_agent+=my_reward_DQN
        #ddqn
        for node in range(simulate_env.Node_num):
            simulate_env.update_node(job_attrs,node,policyID=1)
        DDQN_state=simulate_env.get_states(job_attrs,policyID=1)
        DDQN_action=DDQN_agent.choose_action_max(torch.FloatTensor(DDQN_state))
        DDQN_reward = simulate_env.feedback(job_attrs, DDQN_action,policyID=1)
        grade_DDQN_agent+=DDQN_reward
        #DDuel-DQN
        for node in range(simulate_env.Node_num):
            simulate_env.update_node(job_attrs,node,policyID=2)
        DDuelDQN_state=simulate_env.get_states(job_attrs,policyID=2)
        DDuelDQN_action=DDuel_DQN_agent.choose_action_max(torch.FloatTensor(DDuelDQN_state))
        DDuelDQN_reward = simulate_env.feedback(job_attrs, DDuelDQN_action,policyID=2)
        grade_DDuel_DQN_agent+=DDuelDQN_reward
        #D Duel-DQN with priority
        for node in range(simulate_env.Node_num):
            simulate_env.update_node(job_attrs,node,policyID=3)
        DDuelDQN_pri_state=simulate_env.get_states(job_attrs,policyID=3)
        DDuelDQN_pri_action=DDuelDQN_pri_agent.choose_action_max(torch.FloatTensor(DDuelDQN_pri_state))
        DDuelDQN_pri_reward= simulate_env.feedback(job_attrs, DDuelDQN_pri_action,policyID=3)
        grade_DDuelDQN_pri_agent+=DDuelDQN_pri_reward
        job_c+=1
        if finish:
            break
    return grade_RL_agent,grade_DDQN_agent,grade_DDuel_DQN_agent,grade_DDuelDQN_pri_agent

