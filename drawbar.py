import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# data=pd.read_csv("./experiment4/result/node4_acc/success_differnamadalog.csv")
# data=np.array(data)
# x=np.array([10,11,12,13])
# RL_avg_responseT=np.zeros(len(x))
# RR_avg_responseT=np.zeros(len(x))
# Random_avg_responseT=np.zeros(len(x))
# greedy_wait_avg_responseT=np.zeros(len(x))
# greedy_response_avg_responseT=np.zeros(len(x))
# RL_finT=np.zeros(len(x))
# RR_finT=np.zeros(len(x))
# Random_finT=np.zeros(len(x))
# greedy_wait_finT=np.zeros(len(x))
# greedy_response_finT=np.zeros(len(x))
# RL_index=0
# RR_index=0
# Random_index=0
# greedy_wait_index=0
# greedy_response_index=0
# for i in range(len(data)):
#     if data[i,0]=='RL':
#         RL_avg_responseT[RL_index]=data[i,4]
#         RL_finT[RL_index]=data[i,6]
#         RL_index+=1
#     elif data[i,0]=='RR':
#         RR_avg_responseT[RR_index]=data[i,4]
#         RR_finT[RR_index]=data[i,6]
#         RR_index+=1
#     elif data[i,0]=='Random':
#         Random_avg_responseT[Random_index]=data[i,4]
#         Random_finT[Random_index]=data[i,6]
#         Random_index+=1
#     elif data[i,0]=='greedy_wait':
#         greedy_wait_avg_responseT[greedy_wait_index]=data[i,4]
#         greedy_wait_finT[greedy_wait_index]=data[i,6]
#         greedy_wait_index+=1
#     elif data[i,0]=='greedy_response':
#         greedy_response_avg_responseT[greedy_response_index]=data[i,4]
#         greedy_response_finT[greedy_response_index]=data[i,6]
#         greedy_response_index+=1















data=pd.read_csv("./experiment4/result/node4_acc/EvaulationMethodLog2.csv")
data=np.array(data)
x=np.array([10,11,12,13])
RL_avg_responseT=np.zeros(len(x))
RLavg_wait_responseT=np.zeros(len(x))
RLslow_down_responseT=np.zeros(len(x))
greedy_wait_avg_responseT=np.zeros(len(x))
greedy_response_avg_responseT=np.zeros(len(x))
RL_index=0
RR_index=0
Random_index=0
greedy_wait_index=0
greedy_response_index=0
for i in range(len(data)):
    if data[i,0]=='RL':
        RL_avg_responseT[RL_index]=data[i,2]
        RL_index+=1
    elif data[i,0]=='RLavg_wait_responseT':
        RLavg_wait_responseT[RR_index]=data[i,2]
        RR_index+=1
    elif data[i,0]=='RLslow_down_responseT':
        RLslow_down_responseT[Random_index]=data[i,2]
        Random_index+=1
    elif data[i,0]=='greedy_wait':
        greedy_wait_avg_responseT[greedy_wait_index]=data[i,2]
        greedy_wait_index+=1
    elif data[i,0]=='greedy_response':
        greedy_response_avg_responseT[greedy_response_index]=data[i,2]
        greedy_response_index+=1
# 创建分组柱状图，需要自己控制x轴坐标
xticks = np.arange(len(x))
fig, ax = plt.subplots(figsize=(10,7))
ax.bar(xticks, RL_avg_responseT, width=0.16, label="RL_avg_response", color='#8ECFC9')
ax.bar(xticks + 0.16, RLslow_down_responseT, width=0.16, label="RL_slow_down", color='#ADD8E6')
ax.bar(xticks + 0.32, RLavg_wait_responseT, width=0.16, label="RL_avg_wait", color='#90EE90')
ax.bar(xticks + 0.48, greedy_wait_avg_responseT, width=0.16, label="Greedy_wait", color='#FFBE7A')
ax.bar(xticks + 0.64, greedy_response_avg_responseT, width=0.16, label="Greedy_response", color='#FA7F6F')
ax.set_title("Average Response Times", fontsize=15)
ax.set_xlabel("Mean Arrival Rate",fontsize=12)
ax.set_ylabel("avg_responseT",fontsize=12)
ax.legend()
ax.set_xticks(xticks+0.32)
ax.set_xticklabels(x)
plt.savefig("./experiment4/resultshow/EvaulationMethodLog2_avg_responseTBar.svg", format='svg')
# print(f'RL_avg_responseT:{RL_avg_responseT},greedy_wait_avg_responseT:{greedy_wait_avg_responseT},greedy_response_avg_responseT:{greedy_response_avg_responseT}')
# ax.bar(xticks, 5000/RL_finT, width=0.25, label="RL")
# ax.bar(xticks + 0.16, RR_finT, width=0.16, label="RR", color="blue")
# ax.bar(xticks + 0.32, Random_finT, width=0.16, label="Random", color="orange")
# ax.bar(xticks + 0.25, 5000/greedy_wait_finT, width=0.25, label="greedy_wait")
# ax.bar(xticks + 0.5, 5000/greedy_response_finT, width=0.25, label="greedy_response")
# ax.set_title("Thoughout", fontsize=15)
# ax.set_xlabel("Mean Arrival Rate(Job/s)",fontsize=12)
# ax.set_ylabel("finT(s)",fontsize=12)
# ax.legend()
# ax.set_xticks(xticks+0.25)
# ax.set_xticklabels(x)
# plt.savefig("./experiment4/resultshow/noclip_node4_all_algorithm_finT")
# print(f'5000/RL_finT:{5000/RL_finT},5000/greedy_wait_finT:{5000/greedy_wait_finT},5000/greedy_response_finT:{5000/greedy_response_finT}')

# xticks = np.arange(len(shops))

# fig, ax = plt.subplots(figsize=(10, 7))
# # 所有门店第一种产品的销量，注意控制柱子的宽度，这里选择0.25
# ax.bar(xticks, sales_product_1, width=0.25, label="Product_1", color="red")
# # 所有门店第二种产品的销量，通过微调x轴坐标来调整新增柱子的位置
# ax.bar(xticks + 0.25, sales_product_2, width=0.25, label="Product_2", color="blue")
# # 所有门店第三种产品的销量，继续微调x轴坐标调整新增柱子的位置
# ax.bar(xticks + 0.5, sales_product_3, width=0.25, label="Product_3", color="green")

# ax.set_title("Grouped Bar plot", fontsize=15)
# ax.set_xlabel("Shops")
# ax.set_ylabel("Product Sales")
# ax.legend()

# # 最后调整x轴标签的位置
# ax.set_xticks(xticks + 0.25)
# ax.set_xticklabels(shops)
# fig.show()
