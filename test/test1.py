import cvxpy as cp

# 创建两个标量优化变量。
x = cp.Variable()
y = cp.Variable()

# 创建两个约束条件。
constraints = [x + y == 1,
               x - y >= 1]

# 构建目标函数。
obj = cp.Minimize((x - y)**2)

# 构建并求解问题。
prob = cp.Problem(obj, constraints)
prob.solve()  # 返回最优值。
print("status:", prob.status)
print("optimal value", prob.value)
print("optimal var", x.value, y.value)