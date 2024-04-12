import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import random
from collections import deque
from sum_tree import SumTree
random.seed(6)
np.random.seed(6)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Dueling_DQN(nn.Module):
    def __init__(self, in_channels, num_actions):
        super(Dueling_DQN, self).__init__()
        self.num_actions = num_actions
        self.ln_f = nn.LayerNorm(in_channels)
        self.fc1_adv = nn.Linear(in_features=in_channels, out_features=20)
        self.fc2_adv = nn.Linear(in_features=20, out_features=num_actions)
        self.fc1_val = nn.Linear(in_features=in_channels, out_features=20)
        self.fc2_val = nn.Linear(in_features=20, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x=self.ln_f(x)
        adv = self.relu(self.fc1_adv(x))
        val = self.relu(self.fc1_val(x))
        adv = self.fc2_adv(adv)
        val = self.fc2_val(val).expand(x.size(0), self.num_actions)
        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.num_actions)
        x = x.squeeze()
        return x
class DQN(nn.Module):
    def __init__(self, in_channels, num_actions):
        super(DQN, self).__init__()
        self.num_actions = num_actions
        self.fc1 = nn.Linear(in_features=in_channels, out_features=20)
        self.fc2 = nn.Linear(in_features=20,out_features=20)
        self.fc3 = nn.Linear(in_features=20, out_features=num_actions)
        self.relu = nn.ReLU()

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.squeeze()
        return x
#prioritize double duel-DQN
class RL_model0(nn.Module):
    #in_channels ==states
    def __init__(self, in_channels, num_actions, args, memory_size=512, batch_size=32):
        self.args=args
        super(RL_model0, self).__init__()
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.memory_size = memory_size
        self.replay_buffer = SumTree(self.memory_size)
        self.alpha=args.alpha
        self.beta=args.beta
        self.Dqn = Dueling_DQN(in_channels, num_actions)
        self.target_Dqn = Dueling_DQN(in_channels, num_actions)
        self.epsilon_max = args.myRL_e_greedy
        self.replace_target_iter=args.replace_target_iter
        self.e_greedy_increment=args.e_greedy_increment
        self.epsilon = 0.01 if args.e_greedy_increment is not None else self.epsilon_max
        self.learn_step_counter = 0  # total learning step
        self.gamma=0.999
        self.optimizer = torch.optim.Adam(self.Dqn.parameters(), lr=args.Lr_DDQN)
        self.criterion = nn.L1Loss()
    def forward(self, x):
        x = self.Dqn(x)
        return x
    def update_target(self):
        self.target_Dqn.load_state_dict(self.Dqn.state_dict())

    def choose_action(self,states):
        pro = np.random.uniform()
        if pro < self.epsilon:
            with torch.no_grad():
                q_values=self.Dqn(states.to(device))
            action=np.argmax(q_values.cpu()).item()
        else:
            action = np.random.randint(0, self.num_actions)
        return action
    def choose_action_max(self,states):
        with torch.no_grad():
            q_values=self.target_Dqn(states.to(device))
        action=np.argmax(q_values.cpu()).item()
        return action
    def choose_action_max1(self,states):
        with torch.no_grad():
            q_values=self.Dqn(states.to(device))
        action=np.argmax(q_values.cpu()).item()
        return action
    def learn(self):
        minibatch,minibatch_priority,minibatch_idx = self.sample_buffer()
        minibatch_priority=torch.from_numpy(np.array(minibatch_priority)).type(torch.FloatTensor).to(device)
        minibatch_priority=minibatch_priority/self.replay_buffer.total()
        weight=torch.pow((self.memory_size*minibatch_priority),-self.beta)
        weight=weight/torch.max(weight)
        state_batch = torch.from_numpy(np.array([data[0] for data in minibatch])).type(torch.FloatTensor).to(device)
        action_batch = torch.from_numpy(np.array([data[1] for data in minibatch])).type(torch.LongTensor).to(device)
        reward_batch = torch.from_numpy(np.array([data[2] for data in minibatch])).type(torch.FloatTensor).to(device)
        next_state_batch = torch.from_numpy(np.array([data[3] for data in minibatch])).type(torch.FloatTensor).to(device)
        q_values = self.Dqn(state_batch)
        q_s_a = q_values.gather(1, action_batch.unsqueeze(1))
        q_s_a = q_s_a.squeeze()
        q_tp1_values = self.Dqn(next_state_batch).detach()
        _, a_prime = q_tp1_values.max(1)
        a_prime=a_prime.unsqueeze(1)
        q_s_a_prime=self.target_Dqn(next_state_batch)
        q_s_a_prime=q_s_a_prime.gather(1,a_prime)
        q_s_a_prime=q_s_a_prime.squeeze()
        # error = self.criterion(q_s_a, reward_batch + self.gamma*q_s_a_prime)
        TD_error=reward_batch + self.gamma*q_s_a_prime-q_s_a
        self.update_buffer(torch.pow(torch.abs(TD_error),self.alpha),minibatch_idx)
        error=0.5*weight*(TD_error**2)
        error=torch.sum(error)
        self.optimizer.zero_grad()
        error.backward()
        # nn.utils.clip_grad_value_(self.Dqn.parameters(),5)
        self.optimizer.step()
        # increasing epsilon
        if self.epsilon < self.epsilon_max:
            self.epsilon = self.epsilon + self.e_greedy_increment
        else:
            self.epsilon = self.epsilon_max
        # print('epsilon:', self.epsilon)
        self.learn_step_counter += 1
        if self.learn_step_counter % self.replace_target_iter==0:
            self.update_target()
        return error
    def store_buffer(self, priority,state, action, reward, _state):
        one_hot_action = np.zeros(self.num_actions)
        one_hot_action[action] = 1
        self.replay_buffer.add(priority,(state, action, reward, _state))
    def sample_buffer(self):
        minibatch=[]
        minibatch_priority=[]
        minibatch_idx=[]
        for i in range(self.batch_size):
            random_number=np.random.uniform(0,self.replay_buffer.total())
            idx,priority,data=self.replay_buffer.get(random_number)
            minibatch.append(data)
            minibatch_priority.append(priority)
            minibatch_idx.append(idx)
        return minibatch,minibatch_priority,minibatch_idx
    def update_buffer(self,minibatch_priority,minibatch_idx):
        for i in range(len(minibatch_idx)):
            self.replay_buffer.update(minibatch_idx[i],minibatch_priority[i])
class baselines:
    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.roundrobin_idx=0
    def random_choose_action(self):  # random policy
        action = np.random.randint(self.n_actions)  # [0, n_actions)
        return action
    def greedy_action(self,job_attrs,node_cpu,node_mem):
        cpu = job_attrs[3]
        mem = job_attrs[4]
        for i in range(self.n_actions):
            if(node_cpu[i]>=cpu and node_mem[i]>=mem):
                return i
        return 0
    def greedy_choose_action_wait(self,wait_time):
        action=np.argmin(wait_time)
        return action
    def greedy_choose_action_respose(self,respose_time):
        action=np.argmin(respose_time)
        return action
    def RR_choose_action(self, job_attrs,node_cpu,node_mem):  # round robin policy
        cpu = job_attrs[3]
        mem = job_attrs[4]
        idx=self.roundrobin_idx
        while True:
            if cpu<=node_cpu[self.roundrobin_idx] and mem<=node_mem[self.roundrobin_idx]:
                break
            else:
                self.roundrobin_idx+=1
                if self.roundrobin_idx%(self.n_actions)==0:
                    self.roundrobin_idx=0
                if self.roundrobin_idx==idx:
                    break
        action=self.roundrobin_idx
        self.roundrobin_idx+=1
        if self.roundrobin_idx%(self.n_actions)==0:
            self.roundrobin_idx=0
        return action
#double DQN
class DDQN(nn.Module):
    def __init__(self, in_channels, num_actions, args, memory_size=128, batch_size=32):
        super(DDQN, self).__init__()
        self.args=args
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.memory_size = memory_size
        self.replay_buffer = deque()
        self.Dqn = DQN(in_channels, num_actions)
        self.target_Dqn = DQN(in_channels, num_actions)
        self.epsilon_max = args.myRL_e_greedy
        self.replace_target_iter=args.replace_target_iter
        self.e_greedy_increment=args.e_greedy_increment
        self.epsilon = 0.01 if args.e_greedy_increment is not None else self.epsilon_max
        self.learn_step_counter = 0  # total learning step
        self.gamma=0.999
        self.optimizer = torch.optim.Adam(self.Dqn.parameters(), lr=args.Lr_DDQN)
        self.criterion = nn.L1Loss()
    def forward(self,x):
        x = self.Dqn(x)
        return x
    def update_target(self):
        self.target_Dqn.load_state_dict(self.Dqn.state_dict())
    def choose_action(self,states):
        pro = np.random.uniform()
        if pro < self.epsilon:
            with torch.no_grad():
                q_values=self.Dqn(states.to(device))
            action=np.argmax(q_values.cpu()).item()
        else:
            action = np.random.randint(0, self.num_actions)
        return action
    def choose_action_max(self,states):
        with torch.no_grad():
            q_values=self.target_Dqn(states.to(device))
        action=np.argmax(q_values.cpu()).item()
        return action
    def choose_action_max1(self,states):
        with torch.no_grad():
            q_values=self.Dqn(states.to(device))
        action=np.argmax(q_values.cpu()).item()
        return action
    def compare(self,env):
        test_env=copy.deepcopy(env)
        test_env.reset()
        job_c = 1  # job counter
        oldpolicy_grade=0
        policy_grade=0
        while True:
            finish, job_attrs = test_env.workload(job_c)
            for node in range(self.args.Node_num):
                test_env.update_node(job_attrs,node,policyID=5)
            oldpolicy_state=test_env.get_states(job_attrs,policyID=5)
            oldpolicy_action=self.choose_action_max(torch.FloatTensor(oldpolicy_state))
            oldpolicy_reward = test_env.feedback(job_attrs, oldpolicy_action,policyID=5)
            oldpolicy_grade+=oldpolicy_reward
            for node in range(self.args.Node_num):
                test_env.update_node(job_attrs,node,policyID=7)
            policy_state=test_env.get_states(job_attrs,policyID=7)
            policy_action=self.choose_action_max1(torch.FloatTensor(policy_state))
            policy_reward = test_env.feedback(job_attrs, policy_action,policyID=7)
            policy_grade+=policy_reward
            if finish:
                break
            job_c+=1
        del test_env
        if policy_grade>oldpolicy_grade:
            return 1
        return 0
    def learn(self,env):
        minibatch = random.sample(self.replay_buffer, self.batch_size)
        state_batch = torch.from_numpy(np.array([data[0] for data in minibatch])).type(torch.FloatTensor).to(device)
        action_batch = torch.from_numpy(np.array([data[1] for data in minibatch])).type(torch.LongTensor).to(device)
        reward_batch = torch.from_numpy(np.array([data[2] for data in minibatch])).type(torch.FloatTensor).to(device)
        next_state_batch = torch.from_numpy(np.array([data[3] for data in minibatch])).type(torch.FloatTensor).to(device)
        q_values = self.Dqn(state_batch)
        q_s_a = q_values.gather(1, action_batch.unsqueeze(1))
        q_s_a = q_s_a.squeeze()
        q_tp1_values = self.target_Dqn(next_state_batch).detach()
        q_s_a_prime, a_prime = q_tp1_values.max(1)
        error = self.criterion(q_s_a, reward_batch + self.gamma*q_s_a_prime)
        # q_values = self.Dqn(state_batch)
        # q_s_a = q_values.gather(1, action_batch.unsqueeze(1))
        # q_s_a = q_s_a.squeeze()
        # q_tp1_values = self.Dqn(next_state_batch).detach()
        # _, a_prime = q_tp1_values.max(1)
        # a_prime=a_prime.unsqueeze(1)
        # q_s_a_prime=self.target_Dqn(next_state_batch)
        # q_s_a_prime=q_s_a_prime.gather(1,a_prime)
        # q_s_a_prime=q_s_a_prime.squeeze()
        # error = self.criterion(q_s_a, reward_batch + self.gamma*q_s_a_prime)
        self.optimizer.zero_grad()
        error.backward()
        self.optimizer.step()
        # increasing epsilon
        if self.epsilon < self.epsilon_max:
            self.epsilon = self.epsilon + self.e_greedy_increment
        else:
            self.epsilon = self.epsilon_max
        # print('epsilon:', self.epsilon)
        self.learn_step_counter += 1
        if self.learn_step_counter % self.replace_target_iter==0:
            tag=self.compare(env)
            if tag==1:
                self.update_target()
    def store_buffer(self, state, action, reward, _state):
        one_hot_action = np.zeros(self.num_actions)
        one_hot_action[action] = 1
        self.replay_buffer.append((state, action, reward, _state))
        if len(self.replay_buffer) > self.memory_size:
            self.replay_buffer.popleft()
#double duel-DQN
class DDuel_DQN(nn.Module):
    #in_channels ==states
    def __init__(self, in_channels, num_actions, args, memory_size=128, batch_size=32):
        super(DDuel_DQN, self).__init__()
        self.args=args
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.memory_size = memory_size
        self.replay_buffer = deque()
        self.Dqn = Dueling_DQN(in_channels, num_actions)
        self.target_Dqn = Dueling_DQN(in_channels, num_actions)
        self.epsilon_max = args.myRL_e_greedy
        self.replace_target_iter=args.replace_target_iter
        self.e_greedy_increment=args.e_greedy_increment
        self.epsilon = 0.01 if args.e_greedy_increment is not None else self.epsilon_max
        self.learn_step_counter = 0  # total learning step
        self.gamma=0.999
        self.optimizer = torch.optim.Adam(self.Dqn.parameters(), lr=args.Lr_DDQN)
        self.criterion = nn.L1Loss()
    def forward(self, x):
        x = self.Dqn(x)
        return x
    def update_target(self):
        self.target_Dqn.load_state_dict(self.Dqn.state_dict())
    def choose_action(self,states):
        pro = np.random.uniform()
        if pro < self.epsilon:
            with torch.no_grad():
                q_values=self.Dqn(states.to(device))
            action=np.argmax(q_values.cpu()).item()
        else:
            action = np.random.randint(0, self.num_actions)
        return action
    def choose_action_max(self,states):
        with torch.no_grad():
            q_values=self.target_Dqn(states.to(device))
        action=np.argmax(q_values.cpu()).item()
        return action
    def choose_action_max1(self,states):
        with torch.no_grad():
            q_values=self.Dqn(states.to(device))
        action=np.argmax(q_values.cpu()).item()
        return action
    def compare(self,env):
        test_env=copy.deepcopy(env)
        test_env.reset()
        job_c = 1  # job counter
        oldpolicy_grade=0
        policy_grade=0
        while True:
            finish, job_attrs = test_env.workload(job_c)
            for node in range(self.args.Node_num):
                test_env.update_node(job_attrs,node,policyID=5)
            oldpolicy_state=test_env.get_states(job_attrs,policyID=5)
            oldpolicy_action=self.choose_action_max(torch.FloatTensor(oldpolicy_state))
            oldpolicy_reward = test_env.feedback(job_attrs, oldpolicy_action,policyID=5)
            oldpolicy_grade+=oldpolicy_reward
            for node in range(self.args.Node_num):
                test_env.update_node(job_attrs,node,policyID=7)
            policy_state=test_env.get_states(job_attrs,policyID=7)
            policy_action=self.choose_action_max1(torch.FloatTensor(policy_state))
            policy_reward = test_env.feedback(job_attrs, policy_action,policyID=7)
            policy_grade+=policy_reward
            if finish:
                break
            job_c+=1
        del test_env
        if policy_grade>oldpolicy_grade:
            return 1
        return 0
    def learn(self,env):
        minibatch = random.sample(self.replay_buffer, self.batch_size)
        state_batch = torch.from_numpy(np.array([data[0] for data in minibatch])).type(torch.FloatTensor).to(device)
        action_batch = torch.from_numpy(np.array([data[1] for data in minibatch])).type(torch.LongTensor).to(device)
        reward_batch = torch.from_numpy(np.array([data[2] for data in minibatch])).type(torch.FloatTensor).to(device)
        next_state_batch = torch.from_numpy(np.array([data[3] for data in minibatch])).type(torch.FloatTensor).to(device)
        q_values = self.Dqn(state_batch)
        q_s_a = q_values.gather(1, action_batch.unsqueeze(1))
        q_s_a = q_s_a.squeeze()
        q_tp1_values = self.Dqn(next_state_batch).detach()
        _, a_prime = q_tp1_values.max(1)
        a_prime=a_prime.unsqueeze(1)
        q_s_a_prime=self.target_Dqn(next_state_batch)
        q_s_a_prime=q_s_a_prime.gather(1,a_prime)
        q_s_a_prime=q_s_a_prime.squeeze()
        error = self.criterion(q_s_a, reward_batch + self.gamma*q_s_a_prime)
        self.optimizer.zero_grad()
        error.backward()
        nn.utils.clip_grad_value_(self.Dqn.parameters(),5)
        self.optimizer.step()
        # increasing epsilon
        if self.epsilon < self.epsilon_max:
            self.epsilon = self.epsilon + self.e_greedy_increment
        else:
            self.epsilon = self.epsilon_max
        # print('epsilon:', self.epsilon)
        self.learn_step_counter += 1
        if self.learn_step_counter % self.replace_target_iter==0:
            tag=self.compare(env)
            if tag==1:
                self.update_target()
        return error
    def store_buffer(self, state, action, reward, _state):
        one_hot_action = np.zeros(self.num_actions)
        one_hot_action[action] = 1
        self.replay_buffer.append((state, action, reward, _state))
        if len(self.replay_buffer) > self.memory_size:
            self.replay_buffer.popleft()
