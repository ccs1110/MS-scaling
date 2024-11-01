import time

import cvxpy as cp
import numpy as np

import Function
import Microservice
import Util as ut
import queue
import itertools
from itertools import product

slo=50
len_M = 15
len_F = 1
class DCA:
    def __init__(self, M, F):
        self.M = M
        self.F = F
        self.len_M = len(M)
        self.len_F = len(F)
        self.i_max = 20  # 最大迭代数
        self.len_S = 1000  # 场景数
        self.alpha = 0.95  # 场景采样概率
        self.k_star = (int)(self.len_S * self.alpha - 1)
        self.k_star_plus = self.k_star + 1
        self.N_max = 100
        self.tao_star = 0.1  # 控制参数

    def forward(self, a_r, b_r, c_r, g, k, o):
        # g:          1*M  资源成本
        # a_r:a'_m    1*M
        # b_r:b'_m    1*M
        # c_r:c'_m    1*M
        # k           F*M  端到端分解权重
        # o           1*F  函数占比率

        left_bound = a_r / (self.N_max - b_r)
        right_bound = a_r / (1 - b_r)
        tao = o * self.tao_star
        z_sample = self.sample_z(self.len_S, self.len_M)  # S*M
        varsigma = np.zeros(shape=[self.len_F])
        Z = [queue.PriorityQueue(maxsize=self.len_S + 1) for _ in range(self.len_F)]
        for f in range(self.len_F):
            varsigma[f] = ut.get_slo(f, self.F) - (c_r * k[f]).sum()  # slo-Σkc
        x = cp.Variable(shape=[self.len_M])
        v = cp.Variable(shape=[self.len_F])
        x_cur = cp.Parameter(shape=(self.len_M))  # TODO x需要具有初始值

        x_cur.value = left_bound
        print("xbegin value")
        print(x_cur.value)
        print(left_bound)
        print(right_bound)
        y_cur = cp.Parameter(shape=(self.len_M))
        a_multi_g = cp.Parameter(shape=(self.len_M), nonneg=True, value=g * a_r)
        i = 0  # 迭代index
        g_x_constraint = [left_bound <= x, x <= right_bound]  # 上下界约束
        x_process_value = []
        y_process_value = []
        g_process_value = []
        h_process_value = []
        fenshu = []
        fen_x = cp.sum((cp.inv_pos(x) @ a_multi_g))
        while (i < self.i_max):
            for f in range(self.len_F):
                Z[f].queue.clear()
                for s in range(self.len_S):
                    l_wave_s = (x_cur.value * z_sample[s] * k[f]).sum()
                    Z[f].put((l_wave_s, z_sample[s]))
            L = []
            for f in range(self.len_F):
                l = []
                for s in range(self.len_S):
                    l_wave_s, z_sample_f = Z[f].get()
                    l.append((l_wave_s, z_sample_f))
                L.append(l)
            # 获取h的次梯度
            h_sub_grad_list = self.sub_grad_collection(L, tao)
            # if (len(h_sub_grad) == 1):
            #     y_cur = h_sub_grad[0]
            g_x = cp.sum(cp.multiply(cp.inv_pos(x), a_multi_g))
            h_x = cp.Variable()
            h_x.value = 0
            p_f_expr = self.p_f(x, L)
            q_f_expr = self.q_f(x, L)
            for f in range(self.len_F):
                g_x += tao[f] * cp.maximum(-p_f_expr[f], -q_f_expr[f] - varsigma[f])
                h_x -= (self.k_star_plus) * tao[f] * p_f_expr[f]
            y_cur.value = h_sub_grad_list[0]
            min_value = 100000000000000000000000000000000000000000000000000000000
            if (len(h_sub_grad_list) > 1):  # 当次梯度集大于1时，进行计算
                for h_sub_grad in h_sub_grad_list:  # 枚举求最小值
                    # y_cur.value=h_sub_grad
                    cur_value = self.get_g_y_conj_value(g_x, h_sub_grad, x, g_x_constraint)
                    if (cur_value <= min_value):
                        min_value = cur_value
                        y_cur.value = h_sub_grad
            y_process_value.append(y_cur.value)
            # y_best即为当前的y_t
            # 接下来求x_t+1
            P4_C = []
            P4_C1 = []
            P4_C2 = []
            P4_C3 = g_x_constraint
            for f in range(self.len_F):
                P4_C1.append(-q_f_expr[f] - varsigma[f] <= v[f])
                P4_C2.append(-p_f_expr[f] <= v[f])
            P4_C.extend(P4_C1)
            P4_C.extend(P4_C2)
            P4_C.extend(P4_C3)
            P4_Obj = cp.Minimize(
                cp.sum((cp.inv_pos(x) @ a_multi_g)) - cp.sum(cp.multiply(x, y_cur)) + cp.sum(cp.multiply(v, tao)))
            P4 = cp.Problem(P4_Obj, P4_C)
            P4.solve()
            x_cur.value = x.value
            x_process_value.append(x_cur.value)
            print(g_x.value - h_x.value)
            g_process_value.append(g_x.value)
            h_process_value.append(h_x.value)
            fenshu.append(fen_x.value)
            i = i + 1
            print("var值")
            print([L[f][self.k_star + 1][0] for f in range(self.len_F)])
        print("x过程值")
        print(x_process_value)
        print("y过程值")
        print(y_process_value)
        print("fenshu_x")
        print(fenshu)
        print("varsigma值：")
        print(varsigma)
        return x_process_value

    def sub_grad_collection(self, L, tao):
        l_at_k_starp_list = []  # 1*F
        grad = []
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
                l_at_i, z_s = L[f][i]
                if (l_at_i < 0.999 * l_at_k_starp_list[f]):
                    break
                smin = i

            i = self.k_star_plus
            while (True):
                i = i + 1
                if (i >= self.len_S):
                    break
                l_at_i, z_s = L[f][i]
                if (l_at_i >= 1.001 * l_at_k_starp_list[f]):
                    break
                smax = i
            s_range_min_to_max = [i for i in range(smin, smax + 1)]
            s_range_min_to_kstarp_sublist = list(itertools.combinations(s_range_min_to_max, self.k_star + 2 - smin))

            grad_f = []
            if len(s_range_min_to_kstarp_sublist) == 1:
                grad_f.append(self.k_sum_range(L=L, f=f, start=0, end=self.k_star_plus, tao=tao))
                grad.append(grad_f)
                continue

            # 子集的列表
            left_grad_f = self.k_sum_range(L, f, 0, smin - 1, tao)
            for sublist in s_range_min_to_kstarp_sublist:
                grad_f.append(left_grad_f + self.k_sum(L=L, f=f, sublist=sublist, tao=tao))
            grad.append(grad_f)  # F*S_sub
        sub_grad = [sum(product_Cartesian) for product_Cartesian in product(*grad)]
        return sub_grad

    def k_sum_range(self, L, f, start, end, tao):
        # 提取每个元组的第二个值 z_s，并将其转换为 numpy 数组
        z_s_values = np.array([L[f][s][1] for s in range(start, end)])
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

    def get_g_y_conj_value(self, g_x, y_val, x, constraint):
        objective = cp.Maximize(y_val @ x - g_x)
        g_x_constraint = constraint
        problem = cp.Problem(objective, g_x_constraint)
        problem.solve()
        return problem.value

    # 在mix-DCA中进行蒙特卡洛采样，以估计分布
    def sample_z(self, len_S, len_M):  # 采样z -蒙特卡洛
        sample_z = np.zeros(shape=(len_S, len_M))
        for s in range(len_S):
            sample_z[s, :] = np.random.exponential(scale=1.0, size=len_M)
        return sample_z  # S*M

    # p_f()返回一个表达式列表，其中元素对应每一个f的CVaR(k*+1)
    def p_f(self, x, L):  # 1*F的表达式
        function_p = []
        for f in range(self.len_F):
            function_p_f_value = 0
            for k in range(self.k_star + 1):
                score, value = L[f][k]
                function_p_f_value += value
            function_p_f_value = function_p_f_value / (self.k_star + 1)
            function_p_f = cp.scalar_product(x, function_p_f_value)
            function_p.append(function_p_f)
        return function_p

    # 同q_f返回一个表达式列表，其中元素对应每一个f的CVaR(k*)
    def q_f(self, x, L):  # 1*F的表达式 ,x为1*M变量
        function_q = []
        for f in range(self.len_F):
            function_q_f_value = 0
            for k in range(self.k_star):
                score, value = L[f][k]
                function_q_f_value += value
            function_q_f_value = function_q_f_value / (self.k_star)
            function_q_f = cp.scalar_product(x, function_q_f_value)
            function_q.append(function_q_f)
        return function_q



F = []
M = []
for i in range(len_F):
    f = Function.Function(i, slo)
    F.append(f)
for i in range(len_M):
    m = Microservice.Microservice()
    M.append(m)

np.random.seed(100)
dca = DCA(M, F)
k = np.random.uniform(size=(len_F, len_M), low=1, high=1)
a_r = np.random.uniform(size=(len_M), low=0, high=5)
b_r = -np.random.uniform(size=(len_M), low=0, high=5)
c_r = np.random.uniform(size=(len_M), low=1, high=3)
g = np.ones(shape=(len_M))
o = np.random.uniform(size=(len_F), low=0.1, high=1)
o = o / o.sum()
dca.forward(a_r, b_r, c_r, g, k, o)
print("end--------------------")
