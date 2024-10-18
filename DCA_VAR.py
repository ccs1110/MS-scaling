
import cvxpy as cp
import numpy as np
import Util as ut
import queue
class DCA:
    def __init__(self,F,M):
        self.M=M
        self.F=F
        self.len_M=len(M)
        self.len_F=len(F)
        self.i_max=10000
        self.len_S=1000   #场景数
        self.alpha=0.95
        self.k_star=(int) (self.len_S*self.alpha)-1
        self.k_star_plus= self.k_star + 1
        self.N_max=10
        self.tao=0.001    #控制参数

    def forword(self,a_r,b_r,c_r,k,lamda):
        #a_r:a'_m    1*M
        #b_r:b'_m    1*M
        #c_r:c'_m    1*M
        #k           F*M
        #lamda_m     1*M

        left_bound=a_r/(self.N_max+b_r)
        right_bound=a_r/(1+b_r)

        z_sample=self.sample_z()    #S*M
        varsigma=np.zeros(shpe=(self.len_F))
        for f in range(self.len_F):
            varsigma[f]=ut.get_slo(f,self.F)-c_r.sum()

        y=cp.Variable(shape=[self.len_M])
        x=cp.Variable(shape=[self.len_M])
        x_val=cp.Parameter(shape=[self.len_M])
        y_val=cp.Parameter(shape=[self.len_M])
        i=0                                         #迭代index
        R = queue.PriorityQueue(maxsize=self.len_S+1)
        while(i<self.i_max):
            R.queue.clear()
            for s in range(self.len_S):
                r_s=cp.scalar_product(x_val,z_sample[s])   #内积值作为优先队列的分数
                R.put(r_s,z_sample[s])





            if(False) : #TODO 跳出逻辑
                break
        N_r=x
        np.array()






    def sample_z(self):  #采样z -蒙特卡洛
        sample_z=np.zeros(shape=(self.len_S,self.len_M))
        for s in range(self.len_S):
            sample_z[s, :] = np.random.exponential(scale=1.0, size=self.len_M)
        return sample_z




