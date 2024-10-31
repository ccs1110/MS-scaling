import cvxpy as cp
import numpy as np

# 创建两个标量优化变量。
x = cp.Variable(shape=(3,))
y = cp.Parameter(shape=(3,),value=np.array([2,3,4]))
c=np.array([1,3,4])
# 创建两个约束条件。
constraints = [x>=c]

# 构建目标函数。
obj = cp.Minimize(cp.sum(cp.multiply(cp.inv_pos(x),y)))

P=cp.Problem(obj,constraints)
print(P.is_dcp())
P.solve()