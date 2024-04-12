import argparse
def parameter_parser():
    parser = argparse.ArgumentParser(description="myRL")
    #VM Settings
    parser.add_argument("--Node_performance",
                        type=list,
                        default=[1,2,4,3],#[1,2,4,3,2,1,3,1,2,4],
                        help="Node_performance")
    parser.add_argument("--Node_num",
                        type=int,
                        default=4,
                        help="The number of Node")
    parser.add_argument("--Node_cpu_capacity",
                        type=list,
                        default=[200,300,400,100],#[200,300,400,100,150,200,200,300,400,100],
                        help="Node_cpu_capacity")
    parser.add_argument("--Node_mem_capacity",
                        type=list,
                        default=[100,150,200,200],#[100,150,200,200,300,400,200,300,400,100],
                        help="Node_mem_capacity")
    #Job Settings
    parser.add_argument("--lamda",
                        type=int,
                        default=12,
                        help="The parameter used to control the length of each jobs.")
    parser.add_argument("--Job_Type",
                        type=list,
                        default=[0,1,2],
                        help="The parameter used to control the type of each jobs.")
    parser.add_argument("--Job_Type_prob",
                        type=list,
                        default=[0.3,0.3,0.4],
                        help="The parameter used to control the type of each jobs.")
    parser.add_argument("--cpu_mean",
                        type=list,
                        default=[60,30,45],
                        help="The parameter used to control the type of each jobs.")
    parser.add_argument("--cpu_std",
                        type=list,
                        default=[15,5,10],
                        help="The parameter used to control the type of each jobs.")
    parser.add_argument("--mem_mean",
                        type=list,
                        default=[30,60,45],
                        help="The parameter used to control the type of each jobs.")
    parser.add_argument("--mem_std",
                        type=list,
                        default=[5,15,10],
                        help="The parameter used to control the type of each jobs.")
    parser.add_argument("--duration_mean",
                        type=list,
                        default=[1000,4000,2000],
                        help="The parameter used to control the type of each jobs.")
    parser.add_argument("--duration_std",
                        type=list,
                        default=[200,800,400],
                        help="The parameter used to control the type of each jobs.")
    parser.add_argument("--Job_ddl",
                        type=float,
                        default=4,#s
                        help="Deadline time of each jobs")
    parser.add_argument("--Job_Num",
                        type=int,
                        default=500,
                        help="The number of jobs.")
    parser.add_argument("--alpha",
                        type=float,
                        default=0.4,
                        help="DQN alpha")
    parser.add_argument("--beta",
                        type=float,
                        default=0.6,
                        help="DQN beta")
    #myRL
    parser.add_argument("--Epoch",
                        type=int,
                        default=501,
                        help="Training Epochs")
    parser.add_argument("--Lr_DDQN",
                        type=float,
                        default=0.001,
                        help="Dueling DQN Lr")
    parser.add_argument("--myRL_start_learn",
                        type=int,
                        default=500,
                        help="Iteration start Learn for normal myRL")
    parser.add_argument("--myRL_learn_interval",
                        type=int,
                        default=8,
                        help="myRL's learning interval")
    parser.add_argument("--replace_target_iter",
                        type=int,
                        default=50,
                        help="replace_target_iter")
    parser.add_argument("--myRL_e_greedy",
                        type=float,
                        default=0.95,
                        help="myRL_e_greedy")
    parser.add_argument("--e_greedy_increment",
                        type=float,
                        default=0.002,
                        help="e_greedy_increment")
    #DQN
    parser.add_argument("--DQN_start_learn",
                        type=int,
                        default=500,
                        help="Iteration start Learn for normal myRL")
    parser.add_argument("--DQN_learn_interval",
                        type=int,
                        default=1,
                        help="DQN's learning interval")
    return parser.parse_args()