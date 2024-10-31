import numpy as np
import cvxpy as cp

class Eqution():
    def __init__(self,):
        pass


    #在mix-DCA中进行蒙特卡洛采样，以估计分布
    def sample_z(self,len_S,len_M):  #采样z -蒙特卡洛
        sample_z=np.zeros(shape=(len_S,len_M))
        for s in range(len_S):
            sample_z[s, :] = np.random.exponential(scale=1.0, size=len_M)
        return sample_z  #S*M

    #p_f()返回一个表达式列表，其中元素对应每一个f的CVaR(k*+1)
    def p_f(self,len_F,k_star,x,Z):     # 1*F的表达式
        function_p=[]
        for f in range(len_F):
            pq=Z[f]  #这是一个优先队列
            function_p_f =0
            for k in range(k_star+1):
                score,value=pq.queue[k]
                function_p_f+=x @ value
            function_p_f=function_p_f/(k_star+1)
            function_p.append(function_p_f)
        return function_p

    #同q_f返回一个表达式列表，其中元素对应每一个f的CVaR(k*)
    def q_f(self,len_F,k_star,x,Z):    # 1*F的表达式 ,x为1*M变量
        function_q=[]
        for f in range(len_F):
            pq=Z[f]  #这是一个优先队列
            function_q_f =0
            for k in range(k_star):
                score, value = pq.queue[k]
                function_q_f += x @ value
            function_q_f=function_q_f/(k_star)
            function_q.append(function_q_f)
        return function_q



