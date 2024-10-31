import numpy as np
#T_list:不同请求下的领接矩阵组合
#返回平均邻接矩阵
def E_T(T_list,len_M):
    ET=np.zeros(shape=(len_M,len_M))
    for r in range(len(T_list)):
        ET+=T_list[r]
    ET=ET/len(T_list)
    return ET


#根据平均邻居矩阵返et,返回的值是gamma的M维向量
def gamma_m_throw(et,len_M):
    gamma_m_throw=np.zeros(shape=(len_M))
    for i in range(len_M):
        gamma=0
        for j in range(len_M):
            gamma+=et[i][j]
        gamma_m_throw[i]=gamma
    return gamma_m_throw
