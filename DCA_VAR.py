import time

import cvxpy as cp
import numpy as np

import Function
import Microservice
import Util as ut
import queue
import itertools
from itertools import product
from Eqution import Eqution

len_M = 15
len_F = 3
slo=50
class DCA:
    def __init__(self,F,M):
        self.M=M
        self.F=F
        self.len_M=len(M)
        self.len_F=len(F)
        self.i_max=10000  #最大迭代数
        self.len_S=1000   #场景数
        self.alpha=0.95
        self.k_star=(int) (self.len_S*self.alpha-1)
        self.k_star_plus= self.k_star + 1
        self.N_max=10
        self.tao_star=0.001    #控制参数
    def forword(self,a_r,b_r,c_r,g,k,lamda,o):
        #g:          1*M  资源成本
        #a_r:a'_m    1*M
        #b_r:b'_m    1*M
        #c_r:c'_m    1*M
        #k           F*M  端到端分解权重
        #lamda_m     1*M  到达率
        #o           1*F  函数占比率

        left_bound=a_r/(self.N_max+b_r)
        right_bound=a_r/(1+b_r)
        eq=Eqution()
        tao=o*self.tao_star
        z_sample=eq.sample_z(self.len_S,self.len_M)    #S*M
        varsigma=np.zeros(shape=[self.len_F])
        Z = [queue.PriorityQueue(maxsize=self.len_S + 1) for _ in range(self.len_F)]
        for f in range(self.len_F):
            varsigma[f]=ut.get_slo(f,self.F)-(c_r*k[f]).sum()   #slo-Σkc

        y=cp.Variable(shape=[self.len_M])
        x=cp.Variable(shape=[self.len_M])
        x_cur=cp.Parameter(shape=[self.len_M])
        y_cur=cp.Parameter(shape=[self.len_M])
        a_multi_g = cp.Parameter(shape=[self.len_M], value=g*a_r)
        i=0                                         #迭代index
        while(i<self.i_max):
            for f in range(self.len_F):
                Z[f].queue.clear()
                for s in range(self.len_S):
                    l_wave_s=(x_cur*z_sample[s]*k[f]).sum()
                    Z[f].put(l_wave_s,z_sample[s])
            h_sub_grad=self.sub_grad_collection(self.len_F,Z)
            if(len(h_sub_grad)==1):
                y_cur=h_sub_grad[0]
            # y_cur=self.grad_h(Z,tao)
            g_y_exps=cp.multiply(cp.inv_pos(y),a_multi_g)
            p_f_vals = eq.p_f(self.len_F, self.k_star, y, Z)
            q_f_vals = eq.q_f(self.len_F, self.k_star, y, Z)
                      # cp.max(-eq.p_f(self.len_F,self.k_star,y,Z),-eq.q_f(self.len_F,self.k_star,y,Z)-)
            for f in range(self.len_F):
                g_y_exps += tao[f] * cp.max(-p_f_vals[f], -q_f_vals[f] - varsigma[f])
            g_y_constraint=[left_bound<=x,x<=right_bound]


            if(False) : #TODO 跳出逻辑
                break

    def sub_grad_collection(self,len_F,Z,tao):
        l_at_k_starp_list=[]  #1*F
        grad=[[] for _ in range(self.len_F)]
        for f in range(len_F):
            i = self.k_star_plus
            smin = i
            smax = i
            l_at_k_starp, z_s = L[f][i]
            l_at_k_starp_list.append(l_at_k_starp)
            while (True):
                i = i - 1
                if (i < 0):
                    break
                l_at_i, z_s = Z[f].queue[i]
                if(l_at_i!=l_at_k_starp_list[f]):
                    break
                smin=i
            i = self.k_star_plus
            while(True):
                i=i+1
                if(i>=self.len_F):
                    break
                l_at_i, z_s = Z[f].queue[i]
                if(l_at_i!=l_at_k_starp_list[f]):
                    break
                smax = i
            s_range_min_to_max = [i for i in range(smin, smax + 1)]
            s_range_min_to_kstarp_sublist = list(itertools.combinations(s_range_min_to_max, self.k_star + 2 - smin))

            grad_f=[]
            print(f"f:{f}的子集数量为",len(s_range_min_to_kstarp_sublist))
            if len(s_range_min_to_kstarp_sublist)==1:
                grad_f.append(self.k_sum(Z=Z,f=f,start=0,end=self.k_star_plus,tao=tao))
                continue

            #子集的列表
            left_grad_f=self.k_sum(Z,0,smin-1)
            for sublist in s_range_min_to_kstarp_sublist:
                grad_f.append(left_grad_f+self.k_sum(Z=Z,f=f,sublist=sublist,tao=tao))
            grad.append(grad_f)
        sub_grad=[sum(product_Cartesian) for product_Cartesian in product(*grad)]

    def k_sum_range(self, L, f, start, end, tao):
        # 提取每个元组的第二个值 z_s，并将其转换为 numpy 数组
        z_s_values = np.array([Z[f].queue[s][1] for s in range(start, end + 1)])
        # 计算累减值并乘以 tao[f]，然后累加到 grad_f 中
        grad_f = -np.sum(z_s_values, axis=0) * tao[f]
        return grad_f

    def k_sum(self, L, f, sublist, tao):
        z_s_values = np.array([L[f][s][1] for s in sublist])
        # 计算累减值并乘以 tao[f]，然后累加到 grad_f 中
        grad_f = -np.sum(z_s_values, axis=1) * tao[f]
        return grad_f

    def grad_h(self, L, tao):
        grad_h = np.zeros(shape=(self.len_M))
        for f in range(self.len_F):
            for s in range(self.k_star_plus):
                l_wave, z_s = L[f][s]
                grad_h -= z_s * tao[f]
        return grad_h







np.random.seed(100)
dca = DCA(M, F)
k = np.random.uniform(size=(len_F, len_M), low=1, high=2)
a_r = np.random.uniform(size=(len_M), low=0, high=5)
b_r = -np.random.uniform(size=(len_M), low=0, high=5)
c_r = np.random.uniform(size=(len_M), low=1, high=3)
g = np.ones(shape=(len_M))
o = np.random.uniform(size=(len_F), low=0.1, high=1)
o = o / o.sum()
dca.forward(a_r, b_r, c_r, g, k, o)
print("end--------------------")
