import math
import random
import numpy as np
from scipy import interpolate
from math import cos
import copy
import matplotlib.pyplot as plt
import pylab as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']

# 地图模型类
class Model():
    def __init__(self, start, target, bound, obstacle, n, vpr = 0.1):
        [self.xs, self.ys] = start
        [self.xt, self.yt] = target
        [[self.xmin, self.xmax], [self.ymin, self.ymax]] = bound
        self.vxmax = vpr * (self.xmax - self.xmin)
        self.vxmin = - self.vxmax
        self.vymax = vpr * (self.ymax - self.ymin)
        self.vymin = - self.vymax
        self.nobs = len(obstacle)
        self.xobs = [obs[0] for obs in obstacle]
        self.yobs = [obs[1] for obs in obstacle]
        self.robs = [obs[2] for obs in obstacle]
        self.n = n

    #起点到终点直线路径中间点
    def Straight_path(self):
        xn = np.linspace(self.xs, self.xt, self.n + 2)[1:-1]
        yn = np.linspace(self.ys, self.yt, self.n + 2)[1:-1]
        return [xn, yn]

    #起点到终点随机路径中间点
    def Random_path(self):
        xn = np.random.uniform(self.xmin, self.xmax, self.n)
        yn = np.random.uniform(self.ymin, self.ymax, self.n)
        return [xn, yn]

    # 随机速度值
    def Random_velocity(self):
        vxn = np.random.uniform(self.vxmin, self.vxmax, self.n)
        vyn = np.random.uniform(self.vymin, self.vymax, self.n)
        return [vxn, vyn]

# 位置类
class Position():
    def __init__(self):
        self.x = []
        self.y = []
    def display(self):
        n = len(self.x)
        for i in range(n):
            print('(%f,%f) '%(self.x[i],self.y[i]),end='')
        print('\n')

# 速度类
class Velocity():
    def __init__(self):
        self.x = []
        self.y = []

# 路径类
class Path():
    def __init__(self):
        self.xx = []
        self.yy = []
        self.L = []
        self.violation = np.inf
        self.isfeasible = False
        self.cost = np.inf

    def plan(self,position,model):
        #路径上的决策点
        XS = np.insert(np.array([model.xs, model.xt]), 1, position.x)
        YS = np.insert(np.array([model.ys, model.yt]), 1, position.y)
        TS = np.linspace(0, 1, model.n + 2)
        #插值序列
        tt = np.linspace(0, 1, 100)
        #三次样条插值
        f1 = interpolate.UnivariateSpline(TS, XS, s=0)
        xx = f1(tt)
        f2 = interpolate.UnivariateSpline(TS, YS, s=0)
        yy = f2(tt)
        #差分计算
        dx = np.diff(xx)
        dy = np.diff(yy)
        #路径大小
        L = np.sum(np.sqrt(dx**2 + dy**2))
        #碰撞检测
        violation = 0
        for i in range(model.nobs):
            d = np.sqrt((xx - model.xobs[i])**2 + (yy - model.yobs[i])**2)
            v = np.maximum(1 - np.array(d)/model.robs[i], 0)
            violation = violation + np.mean(v)
        self.xx = xx
        self.yy = yy
        self.L = L
        self.violation = violation
        self.isfeasible = (violation == 0)
        self.cost = L * (1 + violation * 100)

# 最优结果类
class Best():
    def __init__(self):
        self.position = Position()
        self.path = Path()
    pass

# 粒子类
class Particle():
    def __init__(self):
        self.position = Position()
        self.velocity = Velocity()
        self.path = Path()
        self.best = Best()

# 星鸦类
class Nutcracker():
    def __init__(self):
        self.position = Position()
        self.velocity = Velocity()
        self.path = Path()
        self.best = Best()

# 画图函数
def drawPath(model,GBest):
    plt.rcParams['axes.unicode_minus'] = False
    plt.title('路径规划')
    plt.scatter(model.xs, model.ys, label='起点', marker='o', linewidths = 3, color='red')
    plt.scatter(model.xt, model.yt, label='终点', marker='*', linewidths = 3, color='green')
    theta = np.linspace(0, 2 * np.pi, 100)
    for i in range(model.nobs):
        plt.fill(model.xobs[i] + model.robs[i] * np.cos(theta), model.yobs[i] + model.robs[i] * np.sin(theta),'gray')
    plt.scatter(GBest.position.x, GBest.position.y, label='决策点', marker='x', linewidths = 1.5, color='blue')
    plt.plot(GBest.path.xx, GBest.path.yy, label='路径')
    plt.legend()
    plt.grid()
    plt.axis('equal')
    plt.show()

# 画代价曲线
def drawCost(BestCost):
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure()
    plt.title('代价曲线')
    plt.xlabel('代数')
    plt.ylabel('最优代价')
    n = len(BestCost)
    plt.plot(range(n),BestCost)
    plt.grid()
    plt.xlim((0, n + 10))
    # plt.ylim(ymin=0)
    plt.show()
    
def levy_flight(step_size=0.01, size=2):
    # 生成随机步长
    sigma = (np.sqrt(np.pi * 2)) * np.power((np.random.gamma(1.5, 1)), 1 / 3) / np.power(0.625, 1 / 3)
    step = step_size * (sigma / np.abs(np.random.normal(0, 1, size)))

    # 生成随机方向
    direction = np.random.randn(size)

    # 计算新位置
    new_position = step * direction

    return new_position
def find_mu(tau1, tau2, tau3):
    mu1 = None
    mu2 = None
    r = []
    mu = [mu1, mu2]
    for i, value in enumerate(mu):
        while value is None:
            r1 = np.random.rand()
            r2 = np.random.rand()
            r3 = np.random.rand()
            if r1 < r2:
                value = tau1[i]
            if r2 < r3:
                value = tau2[i]
            if r1 < r3:
                value = tau3[i]
        mu[i] = value
        r.append(r1)
    return mu, r
def init_NOA(U, L, n=3):
    x = np.array([])
    y = np.array([])
    for _ in range(n):
        x = np.append(x, (U[0]-L[0]) * np.random.rand() + L[0])
        y = np.append(y, (U[1]-L[1]) * np.random.rand() + L[1])
    return [x, y]
def create_rps(row, column):
    matrix = np.zeros((row, column), dtype=object)
    matrix.fill(None)
    for i in range(row):
        for j in range(column):
            matrix[i, j] = Nutcracker()
    return matrix
            
def NOA(T, size, U, L, model, delta=0.05, Pa1=0.2, Pa2=0.4, Prp=0.2):
    # 初始化种群
    plt.ion()
    plt.figure(1)
    GBest = Best()
    BestCost = []
    Swarm = []
    for i in range(size):
        p = Nutcracker()
        if i: [p.position.x, p.position.y] = init_NOA(U, L)
        else: [p.position.x, p.position.y] = model.Straight_path()
        p.path.plan(p.position, model)   #根据路径点和模型规划具体路径
        #更新局部最优和全局最优
        p.best.position = copy.deepcopy(p.position)
        p.best.path = copy.deepcopy(p.path)
        if p.best.path.isfeasible and (p.best.path.cost < GBest.path.cost):
            GBest = copy.deepcopy(p.best)
        Swarm.append(p)
    BestCost.append(GBest.path.cost)
    L = np.linspace(1, 0, num=T)
    for t in range(T):
        sigma = np.random.rand()
        sigma1 = np.random.rand()
        if sigma < sigma1:
            x_sum = 0
            y_sum = 0
            xm = 0
            ym = 0
            sum = 0
            l = L[t]
            for i in range(size):
                if sum != 0:
                    xm = x_sum / sum
                    ym = y_sum / sum
                phi = np.random.rand()
                A = random.randint(0, len(Swarm)-1)
                B = random.randint(0, len(Swarm)-1)
                C = random.randint(0, len(Swarm)-1)
                p = Swarm[i]
                tau1 = [np.random.rand(), np.random.rand()]
                tau2 = [np.random.rand(), np.random.rand()]
                tau3 = [np.random.rand(), np.random.rand()]
                tau4 = [np.random.normal(0, 1), np.random.normal(0, 1)]
                tau5 = levy_flight(size=2)
                mu, r1 = find_mu(tau3, tau4, tau5)
                Lambda = levy_flight(size=2)
                r = [np.random.rand(), np.random.rand()]
                if phi > Pa1:
                    if tau1 >= tau2:
                        gama = levy_flight(size=2)
                        if t <= T/2:
                            p.position.x = xm  + gama[0] * (Swarm[A].position.x - Swarm[B].position.x) + mu[0] * (pow(r[0], 2) * U[0] - L[0])     
                            p.position.y = ym + gama[1] * (Swarm[A].position.y - Swarm[B].position.y) + mu[1] * (pow(r[1], 2) * U[1] - L[1]) 
                        else:
                            p.position.x = Swarm[C].position.x + mu[0] * (Swarm[A].position.x - Swarm[B].position.x) + mu[0] * (pow(r[0], 2) * U[0] - L[0])   
                            p.position.y = Swarm[C].position.y + mu[1] * (Swarm[A].position.y - Swarm[B].position.y) + mu[1] * (pow(r[1], 2) * U[1] - L[1]) * (r1[1] < delta)      
                else:
                    if tau1[0] < tau2[0]:
                        p.position.x = mu[0] * (p.best.position.x - p.position.x) * abs(Lambda[0]) + r1[0] * (Swarm[A].position.x - Swarm[B].position.x)
                    if tau1[0] < tau3[0]:
                        p.position.x = p.best.position.x + mu[0] * (Swarm[A].position.x - Swarm[B].position.x)
                    else:
                        p.position.x = p.best.position.x * l
                    if tau1[1] < tau2[1]:
                        p.position.x = mu[0] * (p.best.position.x - p.position.x) * abs(Lambda[0]) + r1[1] * (Swarm[A].position.x - Swarm[B].position.x)
                    if tau1[1] < tau3[1]:
                        p.position.x = p.best.position.x + mu[0] * (Swarm[A].position.x - Swarm[B].position.x)
                    else:
                        p.position.x = p.best.position.x * l
                p.path.plan(p.position, model)
                Swarm[i].path.plan(Swarm[i].position, model)
                if p.path.cost < Swarm[i].path.cost:
                    Swarm[i] = p
                if p.path.cost < p.best.path.cost:
                    p.best.position = copy.deepcopy(p.position)
                    p.best.path = copy.deepcopy(p.path)
                    if p.best.path.isfeasible and (p.best.path.cost < GBest.path.cost):
                        GBest = copy.deepcopy(p.best)
                x_sum += p.position.x
                y_sum += p.position.y
                sum += 1
        else:
            RPs = create_rps(size, 2)
            for i in range(size): 
                PHI = np.random.rand()
                theta = [random.uniform(0, math.pi), random.uniform(0, math.pi)]
                A = random.randint(0, len(Swarm)-1)
                B = random.randint(0, len(Swarm)-1)
                C = random.randint(0, len(Swarm)-1)
                RP = random.randint(0, len(Swarm)-1)
                p = Swarm[i]
                tau3 = [np.random.rand(), np.random.rand()]
                tau4 = [np.random.rand(), np.random.rand()]
                tau7 = [np.random.rand(), np.random.rand()]
                tau8 = [np.random.rand(), np.random.rand()]
                r1 = [np.random.rand(), np.random.rand()]
                r2 = [np.random.rand(), np.random.rand()]
                if r2[0] < Prp:
                    U2 = 1
                else:
                    U2 = 0
                if r1[0] > r2[0]:
                    alpha = pow((1 - t/T), (2 * t / T))
                else:
                    alpha = pow((t / T), (2 / (t + 1e-8)))
                if theta[0] == math.pi / 2:
                    RPs[i, 0].position.x = p.position.x + alpha * cos(theta[0]) * (Swarm[A].position.x - Swarm[B].position.x) + Swarm[RP].position.x
                    RPs[i, 1].position.x = p.position.x + (alpha * cos(theta[0]) * ((U[0] - L[0]) * tau3[0] + L[0]) + alpha * Swarm[RP].position.x) * U2
                else:
                    RPs[i, 0].position.x = p.position.x + alpha * cos(theta[0]) * (Swarm[A].position.x - Swarm[B].position.x)
                    RPs[i, 1].position.x = p.position.x + (alpha * cos(theta[0]) * (U[0] - L[0]) * tau3[0] + L[0]) * U2
                if r2[1] < Prp:
                    U2 = 1
                else:
                    U2 = 0
                if r1[1] > r2[0]:
                    alpha = pow((1 - t/T), (2 * t / T))
                else:
                    alpha = pow((t / T), (2 / (t + 1e-8)))
                if theta[1] == math.pi / 2:
                    RPs[i, 0].position.y = p.position.y + alpha * cos(theta[1]) * (Swarm[A].position.y - Swarm[B].position.y) + Swarm[RP].position.y
                    RPs[i, 1].position.y = p.position.y + (alpha * cos(theta[1]) * ((U[1] - L[1]) * tau3[1] + L[1]) + alpha * Swarm[RP].position.y) * U2
                else:
                    RPs[i, 0].position.y = p.position.y + alpha * cos(theta[1]) * (Swarm[A].position.y - Swarm[B].position.y)
                    RPs[i, 1].position.y = p.position.y + (alpha * cos(theta[1]) * (U[1] - L[1]) * tau3[1] + L[1]) * U2
                if PHI > Pa2:
                    if tau7[0] < tau8[0]:
                        if tau3[0] >= tau4[0]:
                            p.position.x = p.position.x + r1[0] * (p.best.position.x - p.position.x) + r2[0] * (RPs[i, 0].position.x - Swarm[C].position.x)
                    if tau7[1] < tau8[1]:
                        if tau3[1] >= tau4[1]:
                            p.position.y = p.position.y + r1[1] * (p.best.position.y - p.position.y) + r2[1] * (RPs[i, 0].position.y - Swarm[C].position.y)
                else:
                    p.path.plan(p.position, model)
                    RPs[i, 0].path.plan(RPs[i, 0].position, model)
                    RPs[i, 1].path.plan(RPs[i, 1].position, model)
                    if p.path.cost < RPs[i, 0].path.cost:
                        p12 = p
                    else:
                        p12 = RPs[i, 0]
                    if p.path.cost < RPs[i, 1].path.cost:
                        p14 = p
                    else:
                        p14 = RPs[i, 1]
                    p12.path.plan(p12.position, model)
                    p14.path.plan(p14.position, model)
                    if p12.path.cost < p14.path.cost:
                        p = p12
                    else:
                        p = p14
                p.path.plan(p.position, model)
                Swarm[i].path.plan(Swarm[i].position, model)
                if p.path.cost < Swarm[i].path.cost:
                    Swarm[i] = p
                p.path.plan(p.position, model)
                if p.path.cost < p.best.path.cost:
                    p.best.position = copy.deepcopy(p.position)
                    p.best.path = copy.deepcopy(p.path)
                    if p.best.path.isfeasible and (p.best.path.cost < GBest.path.cost):
                        GBest = copy.deepcopy(p.best)
        #展示信息
        print('第%d代:cost=%f,决策点为'%(t+1,GBest.path.cost),end='')
        GBest.position.display()
        BestCost.append(GBest.path.cost)   
        plt.cla()
        drawPath(model,GBest)
        plt.pause(0.01)
    plt.ioff()
    drawCost(BestCost)                
                        
# PSO过程
def PSO(T,size,wmax,wmin,c1,c2,model):
    # 初始化种群
    plt.ion()
    plt.figure(1)
    GBest = Best()
    BestCost = []
    Swarm = []
    for i in range(size):
        p = Particle()
        #第一个粒子路径为起点到终点的直线，其他粒子随机生成路径点，初始速度随机生成
        if i: [p.position.x, p.position.y] = model.Random_path()
        else: [p.position.x, p.position.y] = model.Straight_path()
        [p.velocity.x, p.velocity.y] = model.Random_velocity()
        p.path.plan(p.position, model)   #根据路径点和模型规划具体路径
        #更新局部最优和全局最优
        p.best.position = copy.deepcopy(p.position)
        p.best.path = copy.deepcopy(p.path)
        if p.best.path.isfeasible and (p.best.path.cost < GBest.path.cost):
            GBest = copy.deepcopy(p.best)
        Swarm.append(p)
    BestCost.append(GBest.path.cost)
    #开始迭代
    w = wmax
    for t in range(T):
        for i in range(size):
            p = Swarm[i]
            ##x部分
            #更新速度
            p.velocity.x = w * p.velocity.x + \
                           c1 * np.random.rand() * (p.best.position.x - p.position.x) \
                           + c2 * np.random.rand() * (GBest.position.x - p.position.x)
            #边界约束
            p.velocity.x = np.minimum(model.vxmax, p.velocity.x)
            p.velocity.x = np.maximum(model.vxmin, p.velocity.x)
            #更新x
            p.position.x = p.position.x + p.velocity.x
            #边界约束
            outofrange = (p.position.x < model.xmin) | (p.position.x > model.xmax)
            p.velocity.x[outofrange] = -p.velocity.x[outofrange]
            p.position.x = np.minimum(model.xmax, p.position.x)
            p.position.x = np.maximum(model.xmin, p.position.x)
            ##y部分
            # 更新速度
            p.velocity.y = w * p.velocity.y + \
                           c1 * np.random.rand() * (p.best.position.y - p.position.y) \
                           + c2 * np.random.rand() * (GBest.position.y - p.position.y)
            # 边界约束
            p.velocity.y = np.minimum(model.vymax, p.velocity.y)
            p.velocity.y = np.maximum(model.vymin, p.velocity.y)
            # 更新y
            p.position.y = p.position.y + p.velocity.y
            # 边界约束
            outofrange = (p.position.y < model.ymin) | (p.position.y > model.ymax)
            p.velocity.y[outofrange] = -p.velocity.y[outofrange]
            p.position.y = np.minimum(model.ymax, p.position.y)
            p.position.y = np.maximum(model.ymin, p.position.y)
            ## 重新规划路径
            p.path.plan(p.position, model)
            if p.path.cost < p.best.path.cost:
                p.best.position = copy.deepcopy(p.position)
                p.best.path = copy.deepcopy(p.path)
                if p.best.path.isfeasible and (p.best.path.cost < GBest.path.cost):
                    GBest = copy.deepcopy(p.best)
        #展示信息
        print('第%d代:cost=%f,决策点为'%(t+1,GBest.path.cost),end='')
        GBest.position.display()
        BestCost.append(GBest.path.cost)
        w = w - (wmax - wmin)/T      #动态更新w
        plt.cla()
        drawPath(model,GBest)
        plt.pause(0.01)
    plt.ioff()
    drawCost(BestCost)

if __name__ == '__main__':
    ##创建模型
    start = [0.0, 0.0]  # 起点
    target = [4.0, 6.0]  # 终点
    bound = [[-10.0, 10.0],[-10.0, 10.0]]          #x,y的边界
    obstacle = [[1.5,4.5,1.3],[4.0,3.0,1.0],[1.2,1.5,0.8]]  #障碍圆(x,y,r)
    n = 3   #变量数，即用于确定路径的变量点数
    model = Model(start,target,bound,obstacle,n,0.4)
    ##粒子群参数
    T = 200
    size = 100
    wmax = 0.9
    wmin = 0.4
    c1 = 1.5
    c2 = 1.5
    NOA(T, size, [10, 10], [-10, -10], model, delta=0.1, Pa1=0.3, Pa2=0.5, Prp=0.3)
    PSO(T, size, wmax, wmin, c1, c2, model)

