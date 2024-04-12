import numpy as np
from scipy import stats
from collections import deque
np.random.seed(3)
from change_new_para import parameter_parser
class SchedulingEnv:
    def __init__(self, args):
        #policy number
        self.policies=6
        #Node Setting
        self.Node_performance = args.Node_performance#和基准机器对比的处理速度
        self.Node_num = args.Node_num
        assert self.Node_num == len(self.Node_performance)
        self.Node_cpu_capacity = args.Node_cpu_capacity
        self.Node_mem_capacity = args.Node_mem_capacity#total number of cpu,mem
        assert self.Node_num == len(self.Node_cpu_capacity)
        assert self.Node_num == len(self.Node_mem_capacity)
        #state action
        self.actionNum = args.Node_num
        self.s_features = args.Node_num#3 + args.Node_num #VMnum and job length
        #Job Setting
        #job type(根据不同的任务类型) 
        self.job_type=args.Job_Type
        self.job_type_prob=args.Job_Type_prob
        self.cpu_MI=args.cpu_mean
        self.cpu_std=args.cpu_std
        self.mem_MI=args.mem_mean
        self.mem_std=args.mem_std
        self.duration_MI=args.duration_mean
        self.duration_std=args.duration_std

        self.jobNum = args.Job_Num
        self.lamda = args.lamda
        self.arrival_Times = np.zeros(self.jobNum)
        self.type = np.zeros(self.jobNum)
        #self.duration = np.zeros(self.jobNum)
        self.cpu = np.zeros(self.jobNum)
        self.mem = np.zeros(self.jobNum)
        # self.ddl = np.ones(self.jobNum) * args.Job_ddl  # 250ms = waitT + exeT
        #node store exe simulate
        self.store_exeT=np.zeros(len(args.Job_Type))
        self.store_num=1
        #generate workload
        self.gen_workload(self.lamda)
        #myRL
        # 1-Node id  2-start time  3-wait time  4-waitT+exeT  5-leave time  6-reward  7-actual_exeT  8- success1  9-success2
        self.myRL_events = np.zeros((9, self.jobNum))
        #1-cpu availabe resource 2-mem availabe
        self.myRL_Node_events = np.ones((2, self.Node_num))
        self.myRL_Node_events[0]=self.Node_cpu_capacity
        self.myRL_Node_events[1]=self.Node_mem_capacity
        self.myRL_exe_queue_array = [deque() for _ in range(self.Node_num)]
        self.myRL_wait_queue_array = [deque() for _ in range(self.Node_num)]
        
        #DDQN
        self.DDQN_events = np.zeros((9, self.jobNum))
        self.DDQN_Node_events = np.ones((2, self.Node_num))
        self.DDQN_Node_events[0]=self.Node_cpu_capacity
        self.DDQN_Node_events[1]=self.Node_mem_capacity
        self.DDQN_exe_queue_array = [deque() for _ in range(self.Node_num)]
        self.DDQN_wait_queue_array = [deque() for _ in range(self.Node_num)]
        #D Duel-DQN
        self.DDuelDQN_events = np.zeros((9, self.jobNum))
        self.DDuelDQN_Node_events = np.ones((2, self.Node_num))
        self.DDuelDQN_Node_events[0]=self.Node_cpu_capacity
        self.DDuelDQN_Node_events[1]=self.Node_mem_capacity
        self.DDuelDQN_exe_queue_array = [deque() for _ in range(self.Node_num)]
        self.DDuelDQN_wait_queue_array = [deque() for _ in range(self.Node_num)]
        #D Duel-DQN with priority
        self.DDuelDQN_pri_events = np.zeros((9, self.jobNum))
        self.DDuelDQN_pri_Node_events = np.ones((2, self.Node_num))
        self.DDuelDQN_pri_Node_events[0]=self.Node_cpu_capacity
        self.DDuelDQN_pri_Node_events[1]=self.Node_mem_capacity
        self.DDuelDQN_pri_exe_queue_array = [deque() for _ in range(self.Node_num)]
        self.DDuelDQN_pri_wait_queue_array = [deque() for _ in range(self.Node_num)]
        #greedy with idleT
        self.greedy_idleT_events = np.zeros((9, self.jobNum))
        self.greedy_idleT_Node_events = np.ones((2, self.Node_num))
        self.greedy_idleT_Node_events[0]=self.Node_cpu_capacity
        self.greedy_idleT_Node_events[1]=self.Node_mem_capacity
        self.greedy_idleT_exe_queue_array = [deque() for _ in range(self.Node_num)]
        self.greedy_idleT_wait_queue_array = [deque() for _ in range(self.Node_num)]
        #greedy with idleT with resposeT
        self.greedy_resposeT_events = np.zeros((9, self.jobNum))
        self.greedy_resposeT_Node_events = np.ones((2, self.Node_num))
        self.greedy_resposeT_Node_events[0]=self.Node_cpu_capacity
        self.greedy_resposeT_Node_events[1]=self.Node_mem_capacity
        self.greedy_resposeT_exe_queue_array = [deque() for _ in range(self.Node_num)]
        self.greedy_resposeT_wait_queue_array = [deque() for _ in range(self.Node_num)]
        #job success num
        self.success_num=np.zeros(9)

    def gen_workload(self, lamda):
        #Generate arrival time of jobs (poisson distribution)
        intervalT = stats.expon.rvs(scale=1 / lamda, size=self.jobNum)
        print("intervalT mean: ", round(np.mean(intervalT), 3),
              '  intervalT SD:', round(np.std(intervalT, ddof=1), 3))
        self.arrival_Times = np.around(intervalT.cumsum(), decimals=3)
        last_arrivalT = self.arrival_Times[- 1]
        print('last job arrivalT:', round(last_arrivalT, 3))

        # generate jobs' type
        types = np.zeros(self.jobNum)
        type_prob=np.array(self.job_type_prob)
        for i in range(self.jobNum):
            j_type=np.random.choice(self.job_type, p=type_prob.ravel())
            types[i]=j_type
            self.cpu[i]=int(np.random.normal(self.cpu_MI[j_type],self.cpu_std[j_type]))
            if self.cpu[i]>self.cpu_MI[j_type]+2*self.cpu_std[j_type]:
                self.cpu[i]=self.cpu_MI[j_type]+2*self.cpu_std[j_type]
            elif self.cpu[i]<self.cpu_MI[j_type]-2*self.cpu_std[j_type]:
                self.cpu[i]=self.cpu_MI[j_type]-2*self.cpu_std[j_type]
            self.mem[i]=int(np.random.normal(self.mem_MI[j_type],self.mem_std[j_type]))
            if self.mem[i]>self.mem_MI[j_type]+2*self.mem_std[j_type]:
                self.mem[i]=self.mem_MI[j_type]+2*self.mem_std[j_type]
            elif self.mem[i]<self.mem_MI[j_type]-2*self.mem_std[j_type]:
                self.mem[i]=self.mem_MI[j_type]-2*self.mem_std[j_type]
        self.type = types
        
    def reset(self, args):
        # if each episode generates new workload
        self.policies=6
        self.lamda = args.lamda
        self.jobNum = args.Job_Num
        self.arrival_Times = np.zeros(self.jobNum)
        # self.duration = np.zeros(self.jobNum)
        self.type = np.zeros(self.jobNum)
        self.cpu = np.zeros(self.jobNum)
        self.mem = np.zeros(self.jobNum)
        self.gen_workload(args.lamda)
        # 0-Node id  1-start time  2-wait time  3-waitT+exeT  4-leave time  5-reward 6-actual_exeT  7- success 
        self.myRL_events = np.zeros((9, self.jobNum))
        #1-cpu availabe resource 2-mem availabe
        self.myRL_Node_events = np.zeros((2, self.Node_num))
        self.myRL_Node_events[0]=self.Node_cpu_capacity
        self.myRL_Node_events[1]=self.Node_mem_capacity
        self.myRL_exe_queue_array = [deque() for _ in range(self.Node_num)]
        self.myRL_wait_queue_array = [deque() for _ in range(self.Node_num)]
        #DDQN
        self.DDQN_events = np.zeros((9, self.jobNum))
        self.DDQN_Node_events = np.ones((2, self.Node_num))
        self.DDQN_Node_events[0]=self.Node_cpu_capacity
        self.DDQN_Node_events[1]=self.Node_mem_capacity
        self.DDQN_exe_queue_array = [deque() for _ in range(self.Node_num)]
        self.DDQN_wait_queue_array = [deque() for _ in range(self.Node_num)]
        #D Duel-DQN
        self.DDuelDQN_events = np.zeros((9, self.jobNum))
        self.DDuelDQN_Node_events = np.ones((2, self.Node_num))
        self.DDuelDQN_Node_events[0]=self.Node_cpu_capacity
        self.DDuelDQN_Node_events[1]=self.Node_mem_capacity
        self.DDuelDQN_exe_queue_array = [deque() for _ in range(self.Node_num)]
        self.DDuelDQN_wait_queue_array = [deque() for _ in range(self.Node_num)]
        #D Duel-DQN with priority
        self.DDuelDQN_pri_events = np.zeros((9, self.jobNum))
        self.DDuelDQN_pri_Node_events = np.ones((2, self.Node_num))
        self.DDuelDQN_pri_Node_events[0]=self.Node_cpu_capacity
        self.DDuelDQN_pri_Node_events[1]=self.Node_mem_capacity
        self.DDuelDQN_pri_exe_queue_array = [deque() for _ in range(self.Node_num)]
        self.DDuelDQN_pri_wait_queue_array = [deque() for _ in range(self.Node_num)]
        #greedy with idleT
        self.greedy_idleT_events = np.zeros((9, self.jobNum))
        self.greedy_idleT_Node_events = np.ones((2, self.Node_num))
        self.greedy_idleT_Node_events[0]=self.Node_cpu_capacity
        self.greedy_idleT_Node_events[1]=self.Node_mem_capacity
        self.greedy_idleT_exe_queue_array = [deque() for _ in range(self.Node_num)]
        self.greedy_idleT_wait_queue_array = [deque() for _ in range(self.Node_num)]
        #greedy  with resposeT
        self.greedy_resposeT_events = np.zeros((9, self.jobNum))
        self.greedy_resposeT_Node_events = np.ones((2, self.Node_num))
        self.greedy_resposeT_Node_events[0]=self.Node_cpu_capacity
        self.greedy_resposeT_Node_events[1]=self.Node_mem_capacity
        self.greedy_resposeT_exe_queue_array = [deque() for _ in range(self.Node_num)]
        self.greedy_resposeT_wait_queue_array = [deque() for _ in range(self.Node_num)]
        #job success num
        self.success_num=np.zeros(9)
        
    def workload(self, job_count):
        arrival_time = self.arrival_Times[job_count-1]
        # duration = self.duration[job_count-1]
        type=self.type[job_count-1]
        cpu = self.cpu[job_count-1]
        mem = self.mem[job_count-1]
        #ddl = self.ddl[job_count-1]
        if job_count == self.jobNum:
            finish = True
        else:
            finish = False
        job_attributes = [job_count-1, arrival_time, type, cpu, mem]
        return finish, job_attributes
    def update_node(self,job_attrs,action,policyID = 7):
        check_point=job_attrs[1]
        #myRL node state update
        if policyID==7:
            #release all resouce of execuated job
            exeQ=np.array(self.myRL_exe_queue_array[action])
            # print(f'check_point:{check_point},Node:{action},exeQ:{exeQ}')
            for i in range(len(exeQ)):
                if self.myRL_events[4,exeQ[i]]<=check_point:
                    self.myRL_Node_events[0,action]+=self.cpu[exeQ[i]]
                    self.myRL_Node_events[1,action]+=self.mem[exeQ[i]]
                    self.myRL_exe_queue_array[action].remove(exeQ[i])
                    #update store exeT
                    j_type=int(self.type[exeQ[i]])
                    self.store_exeT[j_type]=self.store_exeT[j_type]+1/self.store_num*(self.myRL_events[6,exeQ[i]]*self.Node_performance[action]-self.store_exeT[j_type])
                    self.store_num+=1
                    #job success num +1
                    self.success_num[policyID]+=1
            waitQ=np.array(self.myRL_wait_queue_array[action])
            # print(f'Node:{action},waitQ:{waitQ}')
            #FCFS for node
            for i in range(len(waitQ)):
                if self.cpu[waitQ[i]]<=self.myRL_Node_events[0,action] and self.mem[waitQ[i]]<=self.myRL_Node_events[1,action]:
                    if self.myRL_events[1,waitQ[i]]<=check_point:
                        if self.myRL_events[4,waitQ[i]]>check_point:
                            self.myRL_Node_events[0,action]-=self.cpu[waitQ[i]]
                            self.myRL_Node_events[1,action]-=self.mem[waitQ[i]]
                            self.myRL_exe_queue_array[action].append(waitQ[i])
                        self.myRL_wait_queue_array[action].remove(waitQ[i])
                else:
                    break
        #DDQN node state update
        elif policyID==1:
            #release all resouce of execuated job
            exeQ=np.array(self.DDQN_exe_queue_array[action])
            for i in range(len(exeQ)):
                if self.DDQN_events[4,exeQ[i]]<=check_point:
                    self.DDQN_Node_events[0,action]+=self.cpu[exeQ[i]]
                    self.DDQN_Node_events[1,action]+=self.mem[exeQ[i]]
                    self.DDQN_exe_queue_array[action].remove(exeQ[i])
                    #update store exeT
                    j_type=int(self.type[exeQ[i]])
                    self.store_exeT[j_type]=self.store_exeT[j_type]+1/self.store_num*(self.DDQN_events[6,exeQ[i]]*self.Node_performance[action]-self.store_exeT[j_type])
                    self.store_num+=1
                    #job success num +1
                    self.success_num[policyID]+=1
            waitQ=np.array(self.DDQN_wait_queue_array[action])
            #FCFS for node
            for i in range(len(waitQ)):
                if self.cpu[waitQ[i]]<=self.DDQN_Node_events[0,action] and self.mem[waitQ[i]]<=self.DDQN_Node_events[1,action]:
                    if self.DDQN_events[1,waitQ[i]]<=check_point:
                        if self.DDQN_events[4,waitQ[i]]>check_point:
                            self.DDQN_Node_events[0,action]-=self.cpu[waitQ[i]]
                            self.DDQN_Node_events[1,action]-=self.mem[waitQ[i]]
                            self.DDQN_exe_queue_array[action].append(waitQ[i])
                        self.DDQN_wait_queue_array[action].remove(waitQ[i])
                else:
                    break
        elif policyID==2:
            #release all resouce of execuated job
            exeQ=np.array(self.DDuelDQN_exe_queue_array[action])
            for i in range(len(exeQ)):
                if self.DDuelDQN_events[4,exeQ[i]]<=check_point:
                    self.DDuelDQN_Node_events[0,action]+=self.cpu[exeQ[i]]
                    self.DDuelDQN_Node_events[1,action]+=self.mem[exeQ[i]]
                    self.DDuelDQN_exe_queue_array[action].remove(exeQ[i])
                    #update store exeT
                    j_type=int(self.type[exeQ[i]])
                    self.store_exeT[j_type]=self.store_exeT[j_type]+1/self.store_num*(self.DDuelDQN_events[6,exeQ[i]]*self.Node_performance[action]-self.store_exeT[j_type])
                    self.store_num+=1
                    #job success num +1
                    self.success_num[policyID]+=1
            waitQ=np.array(self.DDuelDQN_wait_queue_array[action])
            #FCFS for node
            for i in range(len(waitQ)):
                if self.cpu[waitQ[i]]<=self.DDuelDQN_Node_events[0,action] and self.mem[waitQ[i]]<=self.DDuelDQN_Node_events[1,action]:
                    if self.DDuelDQN_events[1,waitQ[i]]<=check_point:
                        if self.DDuelDQN_events[4,waitQ[i]]>check_point:
                            self.DDuelDQN_Node_events[0,action]-=self.cpu[waitQ[i]]
                            self.DDuelDQN_Node_events[1,action]-=self.mem[waitQ[i]]
                            self.DDuelDQN_exe_queue_array[action].append(waitQ[i])
                        self.DDuelDQN_wait_queue_array[action].remove(waitQ[i])
                else:
                    break
        elif policyID==3:
            #release all resouce of execuated job
            exeQ=np.array(self.DDuelDQN_pri_exe_queue_array[action])
            for i in range(len(exeQ)):
                if self.DDuelDQN_pri_events[4,exeQ[i]]<=check_point:
                    self.DDuelDQN_pri_Node_events[0,action]+=self.cpu[exeQ[i]]
                    self.DDuelDQN_pri_Node_events[1,action]+=self.mem[exeQ[i]]
                    self.DDuelDQN_pri_exe_queue_array[action].remove(exeQ[i])
                    #update store exeT
                    j_type=int(self.type[exeQ[i]])
                    self.store_exeT[j_type]=self.store_exeT[j_type]+1/self.store_num*(self.DDuelDQN_pri_events[6,exeQ[i]]*self.Node_performance[action]-self.store_exeT[j_type])
                    self.store_num+=1
                    #job success num +1
                    self.success_num[policyID]+=1
            waitQ=np.array(self.DDuelDQN_pri_wait_queue_array[action])
            #FCFS for node
            for i in range(len(waitQ)):
                if self.cpu[waitQ[i]]<=self.DDuelDQN_pri_Node_events[0,action] and self.mem[waitQ[i]]<=self.DDuelDQN_pri_Node_events[1,action]:
                    if self.DDuelDQN_pri_events[1,waitQ[i]]<=check_point:
                        if self.DDuelDQN_pri_events[4,waitQ[i]]>check_point:
                            self.DDuelDQN_pri_Node_events[0,action]-=self.cpu[waitQ[i]]
                            self.DDuelDQN_pri_Node_events[1,action]-=self.mem[waitQ[i]]
                            self.DDuelDQN_pri_exe_queue_array[action].append(waitQ[i])
                        self.DDuelDQN_pri_wait_queue_array[action].remove(waitQ[i])
                else:
                    break
        elif policyID==4:
            #release all resouce of execuated job
            exeQ=np.array(self.greedy_idleT_exe_queue_array[action])
            for i in range(len(exeQ)):
                if self.greedy_idleT_events[4,exeQ[i]]<=check_point:
                    self.greedy_idleT_Node_events[0,action]+=self.cpu[exeQ[i]]
                    self.greedy_idleT_Node_events[1,action]+=self.mem[exeQ[i]]
                    self.greedy_idleT_exe_queue_array[action].remove(exeQ[i])
                    #update store exeT
                    j_type=int(self.type[exeQ[i]])
                    self.store_exeT[j_type]=self.store_exeT[j_type]+1/self.store_num*(self.greedy_idleT_events[6,exeQ[i]]*self.Node_performance[action]-self.store_exeT[j_type])
                    self.store_num+=1
                    #job success num +1
                    self.success_num[policyID]+=1
            waitQ=np.array(self.greedy_idleT_wait_queue_array[action])
            #FCFS for node
            for i in range(len(waitQ)):
                if self.cpu[waitQ[i]]<=self.greedy_idleT_Node_events[0,action] and self.mem[waitQ[i]]<=self.greedy_idleT_Node_events[1,action]:
                    if self.greedy_idleT_events[1,waitQ[i]]<=check_point:
                        if self.greedy_idleT_events[4,waitQ[i]]>check_point:
                            self.greedy_idleT_Node_events[0,action]-=self.cpu[waitQ[i]]
                            self.greedy_idleT_Node_events[1,action]-=self.mem[waitQ[i]]
                            self.greedy_idleT_exe_queue_array[action].append(waitQ[i])
                        self.greedy_idleT_wait_queue_array[action].remove(waitQ[i])
                else:
                    break
        elif policyID==5:
            #release all resouce of execuated job
            exeQ=np.array(self.greedy_resposeT_exe_queue_array[action])
            for i in range(len(exeQ)):
                if self.greedy_resposeT_events[4,exeQ[i]]<=check_point:
                    self.greedy_resposeT_Node_events[0,action]+=self.cpu[exeQ[i]]
                    self.greedy_resposeT_Node_events[1,action]+=self.mem[exeQ[i]]
                    self.greedy_resposeT_exe_queue_array[action].remove(exeQ[i])
                    #update store exeT
                    j_type=int(self.type[exeQ[i]])
                    self.store_exeT[j_type]=self.store_exeT[j_type]+1/self.store_num*(self.greedy_resposeT_events[6,exeQ[i]]*self.Node_performance[action]-self.store_exeT[j_type])
                    self.store_num+=1
                    #job success num +1
                    self.success_num[policyID]+=1
            waitQ=np.array(self.greedy_resposeT_wait_queue_array[action])
            #FCFS for node
            for i in range(len(waitQ)):
                if self.cpu[waitQ[i]]<=self.greedy_resposeT_Node_events[0,action] and self.mem[waitQ[i]]<=self.greedy_resposeT_Node_events[1,action]:
                    if self.greedy_resposeT_events[1,waitQ[i]]<=check_point:
                        if self.greedy_resposeT_events[4,waitQ[i]]>check_point:
                            self.greedy_resposeT_Node_events[0,action]-=self.cpu[waitQ[i]]
                            self.greedy_resposeT_Node_events[1,action]-=self.mem[waitQ[i]]
                            self.greedy_resposeT_exe_queue_array[action].append(waitQ[i])
                        self.greedy_resposeT_wait_queue_array[action].remove(waitQ[i])
                else:
                    break
    def computing_idleT_simulate(self,job_attrs, action,policyID = 7):
        job_id=job_attrs[0]
        check_point=job_attrs[1]
        cpu = job_attrs[3]
        mem = job_attrs[4]
        #myRL
        if policyID==7:
            if cpu<=self.myRL_Node_events[0,action] and mem<=self.myRL_Node_events[1,action] and len(self.myRL_wait_queue_array[action])==0:
                idleT=check_point
            else:
                #gain cur cpu mem info
                idleT=check_point
                cur_node_cpu=self.myRL_Node_events[0,action]
                cur_node_mem=self.myRL_Node_events[1,action]
                #init cur exeQ_dict
                cur_exeQ=dict()
                wait_idx=0
                exeQ=np.array(self.myRL_exe_queue_array[action])
                for i in range(len(exeQ)):
                    # if self.type[exeQ[i]]==0:
                    #     duration=self.duration1_MI/1000
                    # else:
                    #     duration=self.duration2_MI/1000
                    j_type=int(self.type[exeQ[i]])
                    duration=self.store_exeT[j_type]
                    cur_exeQ.setdefault(exeQ[i],self.myRL_events[1,exeQ[i]]+duration)
                while True:
                    cur_exeQ=dict(sorted(cur_exeQ.items(), key=lambda x: x[1], reverse=True))
                    #update
                    fin_job,fin_t= cur_exeQ.popitem()
                    check_point=fin_t
                    cur_node_cpu+=self.cpu[fin_job]
                    cur_node_mem+=self.mem[fin_job]
                    while wait_idx<len(self.myRL_wait_queue_array[action]):
                        if cur_node_cpu>=self.cpu[self.myRL_wait_queue_array[action][wait_idx]] and cur_node_mem>=self.mem[self.myRL_wait_queue_array[action][wait_idx]]:
                            j_type=int(self.type[self.myRL_wait_queue_array[action][wait_idx]])
                            duration=self.store_exeT[j_type]
                            cur_exeQ.setdefault(self.myRL_wait_queue_array[action][wait_idx],check_point+duration)
                            cur_node_cpu-=self.cpu[self.myRL_wait_queue_array[action][wait_idx]]
                            cur_node_mem-=self.mem[self.myRL_wait_queue_array[action][wait_idx]]
                            wait_idx+=1
                        else:
                            break
                    if wait_idx==len(self.myRL_wait_queue_array[action]):
                        if cur_node_cpu>=cpu and cur_node_mem>=mem:
                            if check_point>idleT:
                                idleT=check_point
                            break
        elif policyID==1:
            if cpu<=self.DDQN_Node_events[0,action] and mem<=self.DDQN_Node_events[1,action] and len(self.DDQN_wait_queue_array[action])==0:
                idleT=check_point
            else:
                #gain cur cpu mem info
                idleT=check_point
                cur_node_cpu=self.DDQN_Node_events[0,action]
                cur_node_mem=self.DDQN_Node_events[1,action]
                #init cur exeQ_dict
                cur_exeQ=dict()
                wait_idx=0
                exeQ=np.array(self.DDQN_exe_queue_array[action])
                for i in range(len(exeQ)):
                    j_type=int(self.type[exeQ[i]])
                    duration=self.store_exeT[j_type]
                    cur_exeQ.setdefault(exeQ[i],self.DDQN_events[1,exeQ[i]]+duration)
                while True:
                    cur_exeQ=dict(sorted(cur_exeQ.items(), key=lambda x: x[1], reverse=True))
                    #update
                    fin_job,fin_t= cur_exeQ.popitem()
                    check_point=fin_t
                    cur_node_cpu+=self.cpu[fin_job]
                    cur_node_mem+=self.mem[fin_job]
                    while wait_idx<len(self.DDQN_wait_queue_array[action]):
                        if cur_node_cpu>=self.cpu[self.DDQN_wait_queue_array[action][wait_idx]] and cur_node_mem>=self.mem[self.DDQN_wait_queue_array[action][wait_idx]]:
                            j_type=int(self.type[self.DDQN_wait_queue_array[action][wait_idx]])
                            duration=self.store_exeT[j_type]
                            cur_exeQ.setdefault(self.DDQN_wait_queue_array[action][wait_idx],check_point+duration)
                            cur_node_cpu-=self.cpu[self.DDQN_wait_queue_array[action][wait_idx]]
                            cur_node_mem-=self.mem[self.DDQN_wait_queue_array[action][wait_idx]]
                            wait_idx+=1
                        else:
                            break
                    if wait_idx==len(self.DDQN_wait_queue_array[action]):
                        if cur_node_cpu>=cpu and cur_node_mem>=mem:
                            if check_point>idleT:
                                idleT=check_point
                            break
        elif policyID==2:
            if cpu<=self.DDuelDQN_Node_events[0,action] and mem<=self.DDuelDQN_Node_events[1,action] and len(self.DDuelDQN_wait_queue_array[action])==0:
                idleT=check_point
            else:
                #gain cur cpu mem info
                idleT=check_point
                cur_node_cpu=self.DDuelDQN_Node_events[0,action]
                cur_node_mem=self.DDuelDQN_Node_events[1,action]
                #init cur exeQ_dict
                cur_exeQ=dict()
                wait_idx=0
                exeQ=np.array(self.DDuelDQN_exe_queue_array[action])
                for i in range(len(exeQ)):
                    j_type=int(self.type[exeQ[i]])
                    duration=self.store_exeT[j_type]
                    cur_exeQ.setdefault(exeQ[i],self.DDuelDQN_events[1,exeQ[i]]+duration)
                while True:
                    cur_exeQ=dict(sorted(cur_exeQ.items(), key=lambda x: x[1], reverse=True))
                    #update
                    fin_job,fin_t= cur_exeQ.popitem()
                    check_point=fin_t
                    cur_node_cpu+=self.cpu[fin_job]
                    cur_node_mem+=self.mem[fin_job]
                    while wait_idx<len(self.DDuelDQN_wait_queue_array[action]):
                        if cur_node_cpu>=self.cpu[self.DDuelDQN_wait_queue_array[action][wait_idx]] and cur_node_mem>=self.mem[self.DDuelDQN_wait_queue_array[action][wait_idx]]:
                            j_type=int(self.type[self.DDuelDQN_wait_queue_array[action][wait_idx]])
                            duration=self.store_exeT[j_type]
                            cur_exeQ.setdefault(self.DDuelDQN_wait_queue_array[action][wait_idx],check_point+duration)
                            cur_node_cpu-=self.cpu[self.DDuelDQN_wait_queue_array[action][wait_idx]]
                            cur_node_mem-=self.mem[self.DDuelDQN_wait_queue_array[action][wait_idx]]
                            wait_idx+=1
                        else:
                            break
                    if wait_idx==len(self.DDuelDQN_wait_queue_array[action]):
                        if cur_node_cpu>=cpu and cur_node_mem>=mem:
                            if check_point>idleT:
                                idleT=check_point
                            break
        elif policyID==3:
            if cpu<=self.DDuelDQN_pri_Node_events[0,action] and mem<=self.DDuelDQN_pri_Node_events[1,action] and len(self.DDuelDQN_pri_wait_queue_array[action])==0:
                idleT=check_point
            else:
                #gain cur cpu mem info
                idleT=check_point
                cur_node_cpu=self.DDuelDQN_pri_Node_events[0,action]
                cur_node_mem=self.DDuelDQN_pri_Node_events[1,action]
                #init cur exeQ_dict
                cur_exeQ=dict()
                wait_idx=0
                exeQ=np.array(self.DDuelDQN_pri_exe_queue_array[action])
                for i in range(len(exeQ)):
                    j_type=int(self.type[exeQ[i]])
                    duration=self.store_exeT[j_type]
                    cur_exeQ.setdefault(exeQ[i],self.DDuelDQN_pri_events[1,exeQ[i]]+duration)
                while True:
                    cur_exeQ=dict(sorted(cur_exeQ.items(), key=lambda x: x[1], reverse=True))
                    #update
                    fin_job,fin_t= cur_exeQ.popitem()
                    check_point=fin_t
                    cur_node_cpu+=self.cpu[fin_job]
                    cur_node_mem+=self.mem[fin_job]
                    while wait_idx<len(self.DDuelDQN_pri_wait_queue_array[action]):
                        if cur_node_cpu>=self.cpu[self.DDuelDQN_pri_wait_queue_array[action][wait_idx]] and cur_node_mem>=self.mem[self.DDuelDQN_pri_wait_queue_array[action][wait_idx]]:
                            j_type=int(self.type[self.DDuelDQN_pri_wait_queue_array[action][wait_idx]])
                            duration=self.store_exeT[j_type]
                            cur_exeQ.setdefault(self.DDuelDQN_pri_wait_queue_array[action][wait_idx],check_point+duration)
                            cur_node_cpu-=self.cpu[self.DDuelDQN_pri_wait_queue_array[action][wait_idx]]
                            cur_node_mem-=self.mem[self.DDuelDQN_pri_wait_queue_array[action][wait_idx]]
                            wait_idx+=1
                        else:
                            break
                    if wait_idx==len(self.DDuelDQN_pri_wait_queue_array[action]):
                        if cur_node_cpu>=cpu and cur_node_mem>=mem:
                            if check_point>idleT:
                                idleT=check_point
                            break
        elif policyID==4:
            if cpu<=self.greedy_idleT_Node_events[0,action] and mem<=self.greedy_idleT_Node_events[1,action] and len(self.greedy_idleT_wait_queue_array[action])==0:
                idleT=check_point
            else:
                #gain cur cpu mem info
                idleT=check_point
                cur_node_cpu=self.greedy_idleT_Node_events[0,action]
                cur_node_mem=self.greedy_idleT_Node_events[1,action]
                #init cur exeQ_dict
                cur_exeQ=dict()
                wait_idx=0
                exeQ=np.array(self.greedy_idleT_exe_queue_array[action])
                for i in range(len(exeQ)):
                    j_type=int(self.type[exeQ[i]])
                    duration=self.store_exeT[j_type]
                    cur_exeQ.setdefault(exeQ[i],self.greedy_idleT_events[1,exeQ[i]]+duration)
                while True:
                    cur_exeQ=dict(sorted(cur_exeQ.items(), key=lambda x: x[1], reverse=True))
                    #update
                    fin_job,fin_t= cur_exeQ.popitem()
                    check_point=fin_t
                    cur_node_cpu+=self.cpu[fin_job]
                    cur_node_mem+=self.mem[fin_job]
                    while wait_idx<len(self.greedy_idleT_wait_queue_array[action]):
                        if cur_node_cpu>=self.cpu[self.greedy_idleT_wait_queue_array[action][wait_idx]] and cur_node_mem>=self.mem[self.greedy_idleT_wait_queue_array[action][wait_idx]]:
                            j_type=int(self.type[self.greedy_idleT_wait_queue_array[action][wait_idx]])
                            duration=self.store_exeT[j_type]
                            cur_exeQ.setdefault(self.greedy_idleT_wait_queue_array[action][wait_idx],check_point+duration)
                            cur_node_cpu-=self.cpu[self.greedy_idleT_wait_queue_array[action][wait_idx]]
                            cur_node_mem-=self.mem[self.greedy_idleT_wait_queue_array[action][wait_idx]]
                            wait_idx+=1
                        else:
                            break
                    if wait_idx==len(self.greedy_idleT_wait_queue_array[action]):
                        if cur_node_cpu>=cpu and cur_node_mem>=mem:
                            if check_point>idleT:
                                idleT=check_point
                            break
        elif policyID==5:
            if cpu<=self.greedy_resposeT_Node_events[0,action] and mem<=self.greedy_resposeT_Node_events[1,action] and len(self.greedy_resposeT_wait_queue_array[action])==0:
                idleT=check_point
            else:
                #gain cur cpu mem info
                idleT=check_point
                cur_node_cpu=self.greedy_resposeT_Node_events[0,action]
                cur_node_mem=self.greedy_resposeT_Node_events[1,action]
                #init cur exeQ_dict
                cur_exeQ=dict()
                wait_idx=0
                exeQ=np.array(self.greedy_resposeT_exe_queue_array[action])
                for i in range(len(exeQ)):
                    j_type=int(self.type[exeQ[i]])
                    duration=self.store_exeT[j_type]
                    cur_exeQ.setdefault(exeQ[i],self.greedy_resposeT_events[1,exeQ[i]]+duration)
                while True:
                    cur_exeQ=dict(sorted(cur_exeQ.items(), key=lambda x: x[1], reverse=True))
                    #update
                    fin_job,fin_t= cur_exeQ.popitem()
                    check_point=fin_t
                    cur_node_cpu+=self.cpu[fin_job]
                    cur_node_mem+=self.mem[fin_job]
                    while wait_idx<len(self.greedy_resposeT_wait_queue_array[action]):
                        if cur_node_cpu>=self.cpu[self.greedy_resposeT_wait_queue_array[action][wait_idx]] and cur_node_mem>=self.mem[self.greedy_resposeT_wait_queue_array[action][wait_idx]]:
                            j_type=int(self.type[self.greedy_resposeT_wait_queue_array[action][wait_idx]])
                            duration=self.store_exeT[j_type]
                            cur_exeQ.setdefault(self.greedy_resposeT_wait_queue_array[action][wait_idx],check_point+duration)
                            cur_node_cpu-=self.cpu[self.greedy_resposeT_wait_queue_array[action][wait_idx]]
                            cur_node_mem-=self.mem[self.greedy_resposeT_wait_queue_array[action][wait_idx]]
                            wait_idx+=1
                        else:
                            break
                    if wait_idx==len(self.greedy_resposeT_wait_queue_array[action]):
                        if cur_node_cpu>=cpu and cur_node_mem>=mem:
                            if check_point>idleT:
                                idleT=check_point
                            break
        return idleT

    def computing_idleT(self,job_attrs, action,policyID = 7):
        job_id=job_attrs[0]
        check_point=job_attrs[1]
        cpu = job_attrs[3]
        mem = job_attrs[4]
        #myRL
        if policyID==7:
            if cpu<=self.myRL_Node_events[0,action] and mem<=self.myRL_Node_events[1,action] and len(self.myRL_wait_queue_array[action])==0:
                idleT=check_point
            else:
                #now node finally exe job
                if len(self.myRL_wait_queue_array[action])!=0:
                    node_fin_job=self.myRL_wait_queue_array[action][-1]
                else:
                    if len(self.myRL_exe_queue_array[action])!=0:
                        node_fin_job=self.myRL_exe_queue_array[action][-1]
                    else:
                        node_fin_job=job_id-1
                        print('error')
                idleT=self.myRL_events[1,node_fin_job]
                # print(f'node:{action},idleT:{idleT},node_fin_job:{node_fin_job}')
                cur_node_cpu=self.myRL_Node_events[0,action]
                cur_node_mem=self.myRL_Node_events[1,action]
                fin_exeQ=dict()
                exeQ=np.array(self.myRL_exe_queue_array[action])
                for i in range(len(exeQ)):
                    if self.myRL_events[4,exeQ[i]]<=idleT:
                        cur_node_cpu+=self.cpu[exeQ[i]]
                        cur_node_mem+=self.mem[exeQ[i]]
                    else:
                        fin_exeQ.setdefault(exeQ[i],self.myRL_events[4,exeQ[i]])
                waitQ=np.array(self.myRL_wait_queue_array[action])
                for i in range(len(waitQ)):
                    if self.myRL_events[4,waitQ[i]]>idleT:
                        cur_node_cpu-=self.cpu[waitQ[i]]
                        cur_node_mem-=self.mem[waitQ[i]]
                        fin_exeQ.setdefault(waitQ[i],self.myRL_events[4,waitQ[i]])
                fin_exeQ=dict(sorted(fin_exeQ.items(),key = lambda kv:(kv[1], kv[0])))
                for j_id,j_leaveT in fin_exeQ.items():
                    if cur_node_cpu>=cpu and cur_node_mem>=mem:
                        break
                    cur_node_cpu+=self.cpu[j_id]
                    cur_node_mem+=self.mem[j_id]
                    idleT=j_leaveT
        elif policyID==1:
            if cpu<=self.DDQN_Node_events[0,action] and mem<=self.DDQN_Node_events[1,action] and len(self.DDQN_wait_queue_array[action])==0:
                idleT=check_point
            else:
                #now node finally exe job
                if len(self.DDQN_wait_queue_array[action])!=0:
                    node_fin_job=self.DDQN_wait_queue_array[action][-1]
                else:
                    if len(self.DDQN_exe_queue_array[action])!=0:
                        node_fin_job=self.DDQN_exe_queue_array[action][-1]
                    else:
                        node_fin_job=job_id-1
                idleT=self.DDQN_events[1,node_fin_job]
                cur_node_cpu=self.DDQN_Node_events[0,action]
                cur_node_mem=self.DDQN_Node_events[1,action]
                fin_exeQ=dict()
                exeQ=np.array(self.DDQN_exe_queue_array[action])
                for i in range(len(exeQ)):
                    if self.DDQN_events[4,exeQ[i]]<=idleT:
                        cur_node_cpu+=self.cpu[exeQ[i]]
                        cur_node_mem+=self.mem[exeQ[i]]
                    else:
                        fin_exeQ.setdefault(exeQ[i],self.DDQN_events[4,exeQ[i]])
                waitQ=np.array(self.DDQN_wait_queue_array[action])
                for i in range(len(waitQ)):
                    if self.DDQN_events[4,waitQ[i]]>idleT:
                        cur_node_cpu-=self.cpu[waitQ[i]]
                        cur_node_mem-=self.mem[waitQ[i]]
                        fin_exeQ.setdefault(waitQ[i],self.DDQN_events[4,waitQ[i]])
                fin_exeQ=dict(sorted(fin_exeQ.items(),key = lambda kv:(kv[1], kv[0])))
                for j_id,j_leaveT in fin_exeQ.items():
                    if cur_node_cpu>=cpu and cur_node_mem>=mem:
                        break
                    cur_node_cpu+=self.cpu[j_id]
                    cur_node_mem+=self.mem[j_id]
                    idleT=j_leaveT
        elif policyID==2:
            if cpu<=self.DDuelDQN_Node_events[0,action] and mem<=self.DDuelDQN_Node_events[1,action] and len(self.DDuelDQN_wait_queue_array[action])==0:
                idleT=check_point
            else:
                #now node finally exe job
                if len(self.DDuelDQN_wait_queue_array[action])!=0:
                    node_fin_job=self.DDuelDQN_wait_queue_array[action][-1]
                else:
                    if len(self.DDuelDQN_exe_queue_array[action])!=0:
                        node_fin_job=self.DDuelDQN_exe_queue_array[action][-1]
                    else:
                        node_fin_job=job_id-1
                        print('error')
                idleT=self.DDuelDQN_events[1,node_fin_job]
                cur_node_cpu=self.DDuelDQN_Node_events[0,action]
                cur_node_mem=self.DDuelDQN_Node_events[1,action]
                fin_exeQ=dict()
                exeQ=np.array(self.DDuelDQN_exe_queue_array[action])
                for i in range(len(exeQ)):
                    if self.DDuelDQN_events[4,exeQ[i]]<=idleT:
                        cur_node_cpu+=self.cpu[exeQ[i]]
                        cur_node_mem+=self.mem[exeQ[i]]
                    else:
                        fin_exeQ.setdefault(exeQ[i],self.DDuelDQN_events[4,exeQ[i]])
                waitQ=np.array(self.DDuelDQN_wait_queue_array[action])
                for i in range(len(waitQ)):
                    if self.DDuelDQN_events[4,waitQ[i]]>idleT:
                        cur_node_cpu-=self.cpu[waitQ[i]]
                        cur_node_mem-=self.mem[waitQ[i]]
                        fin_exeQ.setdefault(waitQ[i],self.DDuelDQN_events[4,waitQ[i]])
                fin_exeQ=dict(sorted(fin_exeQ.items(),key = lambda kv:(kv[1], kv[0])))
                for j_id,j_leaveT in fin_exeQ.items():
                    if cur_node_cpu>=cpu and cur_node_mem>=mem:
                        break
                    cur_node_cpu+=self.cpu[j_id]
                    cur_node_mem+=self.mem[j_id]
                    idleT=j_leaveT
        elif policyID==3:
            if cpu<=self.DDuelDQN_pri_Node_events[0,action] and mem<=self.DDuelDQN_pri_Node_events[1,action] and len(self.DDuelDQN_pri_wait_queue_array[action])==0:
                idleT=check_point
            else:
                #now node finally exe job
                if len(self.DDuelDQN_pri_wait_queue_array[action])!=0:
                    node_fin_job=self.DDuelDQN_pri_wait_queue_array[action][-1]
                else:
                    if len(self.DDuelDQN_pri_exe_queue_array[action])!=0:
                        node_fin_job=self.DDuelDQN_pri_exe_queue_array[action][-1]
                    else:
                        node_fin_job=job_id-1
                        print('error')
                idleT=self.DDuelDQN_pri_events[1,node_fin_job]
                cur_node_cpu=self.DDuelDQN_pri_Node_events[0,action]
                cur_node_mem=self.DDuelDQN_pri_Node_events[1,action]
                fin_exeQ=dict()
                exeQ=np.array(self.DDuelDQN_pri_exe_queue_array[action])
                for i in range(len(exeQ)):
                    if self.DDuelDQN_pri_events[4,exeQ[i]]<=idleT:
                        cur_node_cpu+=self.cpu[exeQ[i]]
                        cur_node_mem+=self.mem[exeQ[i]]
                    else:
                        fin_exeQ.setdefault(exeQ[i],self.DDuelDQN_pri_events[4,exeQ[i]])
                waitQ=np.array(self.DDuelDQN_pri_wait_queue_array[action])
                for i in range(len(waitQ)):
                    if self.DDuelDQN_pri_events[4,waitQ[i]]>idleT:
                        cur_node_cpu-=self.cpu[waitQ[i]]
                        cur_node_mem-=self.mem[waitQ[i]]
                        fin_exeQ.setdefault(waitQ[i],self.DDuelDQN_pri_events[4,waitQ[i]])
                fin_exeQ=dict(sorted(fin_exeQ.items(),key = lambda kv:(kv[1], kv[0])))
                for j_id,j_leaveT in fin_exeQ.items():
                    if cur_node_cpu>=cpu and cur_node_mem>=mem:
                        break
                    cur_node_cpu+=self.cpu[j_id]
                    cur_node_mem+=self.mem[j_id]
                    idleT=j_leaveT
        elif policyID==4:
            if cpu<=self.greedy_idleT_Node_events[0,action] and mem<=self.greedy_idleT_Node_events[1,action] and len(self.greedy_idleT_wait_queue_array[action])==0:
                idleT=check_point
            else:
                #now node finally exe job
                if len(self.greedy_idleT_wait_queue_array[action])!=0:
                    node_fin_job=self.greedy_idleT_wait_queue_array[action][-1]
                else:
                    if len(self.greedy_idleT_exe_queue_array[action])!=0:
                        node_fin_job=self.greedy_idleT_exe_queue_array[action][-1]
                    else:
                        node_fin_job=job_id-1
                        print('error')
                idleT=self.greedy_idleT_events[1,node_fin_job]
                cur_node_cpu=self.greedy_idleT_Node_events[0,action]
                cur_node_mem=self.greedy_idleT_Node_events[1,action]
                fin_exeQ=dict()
                exeQ=np.array(self.greedy_idleT_exe_queue_array[action])
                for i in range(len(exeQ)):
                    if self.greedy_idleT_events[4,exeQ[i]]<=idleT:
                        cur_node_cpu+=self.cpu[exeQ[i]]
                        cur_node_mem+=self.mem[exeQ[i]]
                    else:
                        fin_exeQ.setdefault(exeQ[i],self.greedy_idleT_events[4,exeQ[i]])
                waitQ=np.array(self.greedy_idleT_wait_queue_array[action])
                for i in range(len(waitQ)):
                    if self.greedy_idleT_events[4,waitQ[i]]>idleT:
                        cur_node_cpu-=self.cpu[waitQ[i]]
                        cur_node_mem-=self.mem[waitQ[i]]
                        fin_exeQ.setdefault(waitQ[i],self.greedy_idleT_events[4,waitQ[i]])
                fin_exeQ=dict(sorted(fin_exeQ.items(),key = lambda kv:(kv[1], kv[0])))
                for j_id,j_leaveT in fin_exeQ.items():
                    if cur_node_cpu>=cpu and cur_node_mem>=mem:
                        break
                    cur_node_cpu+=self.cpu[j_id]
                    cur_node_mem+=self.mem[j_id]
                    idleT=j_leaveT
        elif policyID==5:
            if cpu<=self.greedy_resposeT_Node_events[0,action] and mem<=self.greedy_resposeT_Node_events[1,action] and len(self.greedy_resposeT_wait_queue_array[action])==0:
                idleT=check_point
            else:
                #now node finally exe job
                if len(self.greedy_resposeT_wait_queue_array[action])!=0:
                    node_fin_job=self.greedy_resposeT_wait_queue_array[action][-1]
                else:
                    if len(self.greedy_resposeT_exe_queue_array[action])!=0:
                        node_fin_job=self.greedy_resposeT_exe_queue_array[action][-1]
                    else:
                        node_fin_job=job_id-1
                        print('error')
                idleT=self.greedy_resposeT_events[1,node_fin_job]
                cur_node_cpu=self.greedy_resposeT_Node_events[0,action]
                cur_node_mem=self.greedy_resposeT_Node_events[1,action]
                fin_exeQ=dict()
                exeQ=np.array(self.greedy_resposeT_exe_queue_array[action])
                for i in range(len(exeQ)):
                    if self.greedy_resposeT_events[4,exeQ[i]]<=idleT:
                        cur_node_cpu+=self.cpu[exeQ[i]]
                        cur_node_mem+=self.mem[exeQ[i]]
                    else:
                        fin_exeQ.setdefault(exeQ[i],self.greedy_resposeT_events[4,exeQ[i]])
                waitQ=np.array(self.greedy_resposeT_wait_queue_array[action])
                for i in range(len(waitQ)):
                    if self.greedy_resposeT_events[4,waitQ[i]]>idleT:
                        cur_node_cpu-=self.cpu[waitQ[i]]
                        cur_node_mem-=self.mem[waitQ[i]]
                        fin_exeQ.setdefault(waitQ[i],self.greedy_resposeT_events[4,waitQ[i]])
                fin_exeQ=dict(sorted(fin_exeQ.items(),key = lambda kv:(kv[1], kv[0])))
                for j_id,j_leaveT in fin_exeQ.items():
                    if cur_node_cpu>=cpu and cur_node_mem>=mem:
                        break
                    cur_node_cpu+=self.cpu[j_id]
                    cur_node_mem+=self.mem[j_id]
                    idleT=j_leaveT
        return idleT
    def get_states(self,job_attrs,policyID = 7):
        arrival_time = job_attrs[1]
        type = job_attrs[2]
        cpu = job_attrs[3]
        mem = job_attrs[4]
        # state_job = [type,cpu/self.cpu_MI,mem/self.mem_MI]
        state_job = [type]
        idleTimes=np.zeros(self.Node_num)
        exeTimes=np.zeros(self.Node_num)
        resposeTimes=np.zeros(self.Node_num)
        for node in range(self.Node_num):
            idleTimes[node]=self.computing_idleT_simulate(job_attrs,node,policyID)
            resposeTimes[node]=idleTimes[node]+(self.store_exeT[int(type)]/ self.Node_performance[node])#(self.duration_MI[int(type)] / self.Node_performance[node])/1000
            # exeTimes[node] = (self.duration_MI[int(type)] / self.Node_performance[node])/1000
        if policyID==4:
            waitTimes = [t - arrival_time for t in idleTimes]
            waitTimes = np.maximum(waitTimes, 0)
            state = waitTimes
        else:
            durationTimes = [t - arrival_time for t in resposeTimes]
            durationTimes = np.maximum(durationTimes, 0)
            state = durationTimes
        # print(f'state:{state}')
        return state
    def get_store_state(self,job_attrs):
        arrival_time = job_attrs[1]
        type = job_attrs[2]
        cpu = job_attrs[3]
        mem = job_attrs[4]
        # state_job = [type,cpu/self.cpu_MI,mem/self.mem_MI]
        state_job = [type]
        idleTimes=np.zeros(self.Node_num)
        exeTimes=np.zeros(self.Node_num)
        resposeTimes=np.zeros(self.Node_num)
        for node in range(self.Node_num):
            idleTimes[node]=self.computing_idleT_simulate(job_attrs,node,4)
            resposeTimes[node]=idleTimes[node]+(self.duration_MI[int(type)] / self.Node_performance[node])/1000
        durationTimes = [t - arrival_time for t in resposeTimes]
        durationTimes = np.maximum(durationTimes, 0)
        state = durationTimes
        return state

    def feedback(self, job_attrs, action,policyID = 7):
        job_id = job_attrs[0]
        arrival_time = job_attrs[1]
        job_type = int(job_attrs[2])
        cpu = job_attrs[3]
        mem = job_attrs[4]
        # thoughout=self.success_num[policyID]/arrival_time
        # ddl =job_attrs[5]
        #computing real exeT
        duration=self.duration_MI[job_type]#np.DDQN.normal(self.duration_MI[job_type], self.duration_std[type])
        duration=int(duration)
        real_exeT=(duration/self.Node_performance[action])/1000
        #for computing start time and wait time must be computing idleT for now job
        #real idleT for now job computing
        idleT=self.computing_idleT(job_attrs,action,policyID)
        # print(f'job_id:{job_id},arrival_time:{arrival_time},duration:{duration/1000}ms,cpu:{cpu},mem:{mem},real_exeT:{real_exeT},idleT:{idleT}')
        if policyID == 7:
            if idleT<=arrival_time:
                self.myRL_Node_events[0,action]-=cpu
                self.myRL_Node_events[1,action]-=mem
                # print(f'myRL_Node_events,cpu:{self.myRL_Node_events[0,action]},mem:{self.myRL_Node_events[1,action]}')
                self.myRL_exe_queue_array[action].append(job_id)
                waitT = 0
                startT = arrival_time
            else:
                self.myRL_wait_queue_array[action].append(job_id)
                waitT = idleT - arrival_time
                startT = idleT
        elif policyID==1:
            if idleT<=arrival_time:
                self.DDQN_Node_events[0,action]-=cpu
                self.DDQN_Node_events[1,action]-=mem
                self.DDQN_exe_queue_array[action].append(job_id)
                waitT = 0
                startT = arrival_time
            else:
                self.DDQN_wait_queue_array[action].append(job_id)
                waitT = idleT - arrival_time
                startT = idleT
        elif policyID==2:
            if idleT<=arrival_time:
                self.DDuelDQN_Node_events[0,action]-=cpu
                self.DDuelDQN_Node_events[1,action]-=mem
                self.DDuelDQN_exe_queue_array[action].append(job_id)
                waitT = 0
                startT = arrival_time
            else:
                self.DDuelDQN_wait_queue_array[action].append(job_id)
                waitT = idleT - arrival_time
                startT = idleT
        elif policyID==3:
            if idleT<=arrival_time:
                self.DDuelDQN_pri_Node_events[0,action]-=cpu
                self.DDuelDQN_pri_Node_events[1,action]-=mem
                self.DDuelDQN_pri_exe_queue_array[action].append(job_id)
                waitT = 0
                startT = arrival_time
            else:
                self.DDuelDQN_pri_wait_queue_array[action].append(job_id)
                waitT = idleT - arrival_time
                startT = idleT
        elif policyID==4:
            if idleT<=arrival_time:
                self.greedy_idleT_Node_events[0,action]-=cpu
                self.greedy_idleT_Node_events[1,action]-=mem
                self.greedy_idleT_exe_queue_array[action].append(job_id)
                waitT = 0
                startT = arrival_time
            else:
                self.greedy_idleT_wait_queue_array[action].append(job_id)
                waitT = idleT - arrival_time
                startT = idleT
        elif policyID==5:
            if idleT<=arrival_time:
                self.greedy_resposeT_Node_events[0,action]-=cpu
                self.greedy_resposeT_Node_events[1,action]-=mem
                self.greedy_resposeT_exe_queue_array[action].append(job_id)
                waitT = 0
                startT = arrival_time
            else:
                self.greedy_resposeT_wait_queue_array[action].append(job_id)
                waitT = idleT - arrival_time
                startT = idleT
        durationT = waitT + real_exeT  # waitT+exeT
        leaveT = startT + real_exeT  # leave T
        success1 = 1 if waitT/durationT <= 0.25 else 0
        success2 = 1 if waitT/durationT <= 0.5 else 0
        reward = -durationT#-durationT / ddl
        # slow_down=durationT/real_exeT
        # print(f'job_id:{job_id},job_type:{job_type},waitT:{waitT},real_exeT:{real_exeT},reward:{reward}')
        if job_id!=0:
            if policyID == 7:
                reward=np.mean(self.greedy_resposeT_events[3, 0:job_id])-(np.sum(self.myRL_events[3, 0:job_id-1])+durationT)/job_id
            elif policyID==1:
                reward=np.mean(self.greedy_resposeT_events[3, 0:job_id])-(np.sum(self.DDQN_events[3, 0:job_id-1])+durationT)/job_id
            elif policyID==2:
                reward=np.mean(self.greedy_resposeT_events[3, 0:job_id])-(np.sum(self.DDuelDQN_events[3, 0:job_id-1])+durationT)/job_id
            elif policyID==3:
                reward=np.mean(self.greedy_resposeT_events[3, 0:job_id])-(np.sum(self.DDuelDQN_pri_events[3, 0:job_id-1])+durationT)/job_id
            elif policyID==4:
                reward=np.mean(self.greedy_resposeT_events[3, 0:job_id])-(np.sum(self.greedy_idleT_events[3, 0:job_id-1])+durationT)/job_id
                #(np.sum(self.greedy_idleT_events[3, 0:job_id-1]+durationT))/job_id-(np.sum(self.greedy_idleT_events[3, 0:job_id-1]+durationT))/job_id
            elif policyID==5:
                reward=(np.sum(self.greedy_resposeT_events[3, 0:job_id-1]+durationT))/job_id-(np.sum(self.greedy_resposeT_events[3, 0:job_id-1]+durationT))/job_id
        # if job_id!=0:
        #     if policyID == 7:
        #         reward=np.mean(self.greedy_idleT_events[3, 0:job_id])-(np.sum(self.myRL_events[3, 0:job_id-1])+durationT)/job_id
        #     elif policyID==1:
        #         reward=np.mean(self.greedy_idleT_events[3, 0:job_id])-(np.sum(self.DDQN_events[3, 0:job_id-1])+durationT)/job_id
        #     elif policyID==2:
        #         reward=np.mean(self.greedy_idleT_events[3, 0:job_id])-(np.sum(self.DDuelDQN_events[3, 0:job_id-1])+durationT)/job_id
        #     elif policyID==3:
        #         reward=np.mean(self.greedy_idleT_events[3, 0:job_id])-(np.sum(self.DDuelDQN_pri_events[3, 0:job_id-1])+durationT)/job_id
        #     elif policyID==4:
        #         reward=(np.sum(self.greedy_idleT_events[3, 0:job_id-1]+durationT))/job_id-(np.sum(self.greedy_idleT_events[3, 0:job_id-1]+durationT))/job_id#np.mean(self.greedy_idleT_events[3, 0:job_id])-(np.sum(self.greedy_idleT_events[3, 0:job_id-1])+durationT)/job_id
        #         #(np.sum(self.greedy_idleT_events[3, 0:job_id-1]+durationT))/job_id-(np.sum(self.greedy_idleT_events[3, 0:job_id-1]+durationT))/job_id
        #     elif policyID==5:
        #         reward=np.mean(self.greedy_idleT_events[3, 0:job_id])-(np.sum(self.greedy_resposeT_events[3, 0:job_id-1]+durationT))/job_id
        if reward<-2:
            reward=-2
        elif reward>2:
            reward=2
        #update event
        if policyID == 7:
            self.myRL_events[0, job_id] = action
            self.myRL_events[2, job_id] = waitT
            self.myRL_events[1, job_id] = startT
            self.myRL_events[3, job_id] = durationT
            self.myRL_events[4, job_id] = leaveT
            self.myRL_events[5, job_id] = reward
            self.myRL_events[6, job_id] = real_exeT
            self.myRL_events[7, job_id] = success1
            self.myRL_events[8, job_id] = success2
        elif policyID==1:
            self.DDQN_events[0, job_id] = action
            self.DDQN_events[2, job_id] = waitT
            self.DDQN_events[1, job_id] = startT
            self.DDQN_events[3, job_id] = durationT
            self.DDQN_events[4, job_id] = leaveT
            self.DDQN_events[5, job_id] = reward
            self.DDQN_events[6, job_id] = real_exeT
            self.DDQN_events[7, job_id] = success1
            self.DDQN_events[8, job_id] = success2
        elif policyID==2:
            self.DDuelDQN_events[0, job_id] = action
            self.DDuelDQN_events[2, job_id] = waitT
            self.DDuelDQN_events[1, job_id] = startT
            self.DDuelDQN_events[3, job_id] = durationT
            self.DDuelDQN_events[4, job_id] = leaveT
            self.DDuelDQN_events[5, job_id] = reward
            self.DDuelDQN_events[6, job_id] = real_exeT
            self.DDuelDQN_events[7, job_id] = success1
            self.DDuelDQN_events[8, job_id] = success2
        elif policyID==3:
            self.DDuelDQN_pri_events[0, job_id] = action
            self.DDuelDQN_pri_events[2, job_id] = waitT
            self.DDuelDQN_pri_events[1, job_id] = startT
            self.DDuelDQN_pri_events[3, job_id] = durationT
            self.DDuelDQN_pri_events[4, job_id] = leaveT
            self.DDuelDQN_pri_events[5, job_id] = reward
            self.DDuelDQN_pri_events[6, job_id] = real_exeT
            self.DDuelDQN_pri_events[7, job_id] = success1
            self.DDuelDQN_pri_events[8, job_id] = success2
        elif policyID==4:
            self.greedy_idleT_events[0, job_id] = action
            self.greedy_idleT_events[2, job_id] = waitT
            self.greedy_idleT_events[1, job_id] = startT
            self.greedy_idleT_events[3, job_id] = durationT
            self.greedy_idleT_events[4, job_id] = leaveT
            self.greedy_idleT_events[5, job_id] = reward
            self.greedy_idleT_events[6, job_id] = real_exeT
            self.greedy_idleT_events[7, job_id] = success1
            self.greedy_idleT_events[8, job_id] = success2
        elif policyID==5:
            self.greedy_resposeT_events[0, job_id] = action
            self.greedy_resposeT_events[2, job_id] = waitT
            self.greedy_resposeT_events[1, job_id] = startT
            self.greedy_resposeT_events[3, job_id] = durationT
            self.greedy_resposeT_events[4, job_id] = leaveT
            self.greedy_resposeT_events[5, job_id] = reward
            self.greedy_resposeT_events[6, job_id] = real_exeT
            self.greedy_resposeT_events[7, job_id] = success1
            self.greedy_resposeT_events[8, job_id] = success2
        return reward
    def get_accumulateRewards(self,start, end):
        rewards = np.zeros(self.policies)
        # print(f'get_accumulateRewards:self.myRL_events:{self.myRL_events[5, start:end]},self.greedy_events:{self.greedy_events[5, start:end]}')
        rewards[0] = sum(self.myRL_events[5, start:end])
        rewards[1] = sum(self.DDQN_events[5, start:end])
        rewards[2] = sum(self.DDuelDQN_events[5, start:end])
        rewards[3] = sum(self.DDuelDQN_pri_events[5, start:end])
        rewards[4] = sum(self.greedy_idleT_events[5, start:end])
        rewards[5] = sum(self.greedy_resposeT_events[5, start:end])
        return np.around(rewards, 2)
    def get_FinishTimes(self,start, end):
        finishT = np.zeros(self.policies)
        finishT[0] = max(self.myRL_events[4, start:end])
        finishT[1] = max(self.DDQN_events[4, start:end])
        finishT[2] = max(self.DDuelDQN_events[4, start:end])
        finishT[3] = max(self.DDuelDQN_pri_events[4, start:end])
        finishT[4] = max(self.greedy_idleT_events[4, start:end])
        finishT[5] = max(self.greedy_resposeT_events[4, start:end])
        return np.around(finishT, 2)
    def get_executeTs(self,start, end):
        executeTs = np.zeros(self.policies)
        executeTs[0] = np.mean(self.myRL_events[6, start:end])
        executeTs[1] = np.mean(self.DDQN_events[6, start:end])
        executeTs[2] = np.mean(self.DDuelDQN_events[6, start:end])
        executeTs[3] = np.mean(self.DDuelDQN_pri_events[6, start:end])
        executeTs[4] = np.mean(self.greedy_idleT_events[6, start:end])
        executeTs[5] = np.mean(self.greedy_resposeT_events[6, start:end])
        return np.around(executeTs, 3)
    def get_waitTs(self,start, end):
        waitTs = np.zeros(self.policies)
        waitTs[0] = np.mean(self.myRL_events[2, start:end])
        waitTs[1] = np.mean(self.DDQN_events[2, start:end])
        waitTs[2] = np.mean(self.DDuelDQN_events[2, start:end])
        waitTs[3] = np.mean(self.DDuelDQN_pri_events[2, start:end])
        waitTs[4] = np.mean(self.greedy_idleT_events[2, start:end])
        waitTs[5] = np.mean(self.greedy_resposeT_events[2, start:end])
        return np.around(waitTs, 3)
    def get_responseTs(self,start, end):
        respTs = np.zeros(self.policies)
        respTs[0] = np.mean(self.myRL_events[3, start:end])
        respTs[1] = np.mean(self.DDQN_events[3, start:end])
        respTs[2] = np.mean(self.DDuelDQN_events[3, start:end])
        respTs[3] = np.mean(self.DDuelDQN_pri_events[3, start:end])
        respTs[4] = np.mean(self.greedy_idleT_events[3, start:end])
        respTs[5] = np.mean(self.greedy_resposeT_events[3, start:end])
        return np.around(respTs, 3)
    # def get_avg_slowdown(self,start, end):
    #     avg_slowdown = np.zeros(self.policies)
    #     avg_slowdown[0] = np.mean(self.myRL_events[7,start:end])
    #     avg_slowdown[1] = np.mean(self.DDQN_events[7, start:end])
    #     avg_slowdown[2] = np.mean(self.DDuelDQN_events[7,start:end])
    #     avg_slowdown[3] = np.mean(self.DDuelDQN_pri_events[7,start:end])
    #     avg_slowdown[4] = np.mean(self.greedy_idleT_events[7,start:end])
    #     avg_slowdown[5] = np.mean(self.greedy_resposeT_events[7,start:end])
    #     avg_slowdown = np.around(avg_slowdown, 3)
    #     return avg_slowdown
    def get_successTimes1(self,start, end):
        successT = np.zeros(self.policies)
        successT[0] = sum(self.myRL_events[7, start:end])/(end - start)
        successT[1] = sum(self.DDQN_events[7, start:end])/(end - start)
        successT[2] = sum(self.DDuelDQN_events[7, start:end])/(end - start)
        successT[3] = sum(self.DDuelDQN_pri_events[7, start:end])/(end - start)
        successT[4] = sum(self.greedy_idleT_events[7, start:end])/(end - start)
        successT[5] = sum(self.greedy_resposeT_events[7, start:end])/(end - start)
        successT = np.around(successT, 3)
        return successT
    def get_successTimes2(self,start, end):
        successT = np.zeros(self.policies)
        successT[0] = sum(self.myRL_events[8, start:end])/(end - start)
        successT[1] = sum(self.DDQN_events[8, start:end])/(end - start)
        successT[2] = sum(self.DDuelDQN_events[8, start:end])/(end - start)
        successT[3] = sum(self.DDuelDQN_pri_events[8, start:end])/(end - start)
        successT[4] = sum(self.greedy_idleT_events[8, start:end])/(end - start)
        successT[5] = sum(self.greedy_resposeT_events[8, start:end])/(end - start)
        successT = np.around(successT, 3)
        return successT