#obstacle, collision check(is_in_obstacle)
import numpy as np
import random

class Node:
    def __init__(self, config, time):
        self.config = np.array(config)
        self.time = time
        self.parent = None
        self.children = []
        self.cost = 0 #float('inf')

class Tree:
    def __init__(self):
        self.nodes = []

    def add_node(self, node):
        self.nodes.append(node)

    def get_nearest(self, config, time):
        nearest_node = None
        min_distance = float('inf')
        for node in self.nodes:
            dist = d(node, Node(config, time))
            if dist < min_distance:
                min_distance = dist
                nearest_node = node
        return nearest_node
    
    def get_nearest_g(self, config, time):
        nearest_node = None
        min_distance = float('inf')
        for node in self.nodes:
            dist = d(Node(config, time), node)
            if dist < min_distance:
                min_distance = dist
                nearest_node = node
        return nearest_node

    def get_nearest_k(self, config, time, k): #k는 radius, 거리범위가 k 이내인 노드들 조사
        distances = []
        for node in self.nodes:
            dist = d(node, Node(config, time)) #node가 골 트리의 노드, Node가 x_new
            if dist  != float('inf'): distances.append((dist, node))
            #print(len(distances))
        distances.sort(key=lambda x: x[0])
        return [node for _, node in distances]

class Obstacle:
    def __init__(self, space_bounds, time_bounds):
        self.space_bounds = space_bounds  # [x_min, x_max]
        self.time_bounds = time_bounds    # [t_min, t_max]


def interpolate(node1, node2, step_size=0.1):
    config1 = node1.config
    config2 = node2.config
    time1 = node1.time
    time2 = node2.time
    # 두 노드 사이의 disance fuction에 따른 거리 계산 (공간+시간 차원 포함)
    #distance = np.linalg.norm(np.append(config2 - config1, time2 - time1))
    dt = abs(node2.time - node1.time)
    dq = node2.config - node1.config
    lam = 0.5
    #distance = np.linalg.norm(np.append(config2 - config1, time2 - time1))
    distance = lam * np.linalg.norm(dq) + (1-lam) * dt

    # 거리와 step_size에 따른 step 개수 계산
    num_steps = max(1, int(distance / step_size))

    interpolated_points = []

    for i in range(num_steps + 1):
        t = i / num_steps
        interpolated_config = config1 * (1 - t) + config2 * t
        interpolated_time = time1 * (1 - t) + time2 * t
        interpolated_points.append(Node(interpolated_config, interpolated_time))

    return interpolated_points

def is_in_obstacle(node, obstacle):  # True면 collision 발생
    config = node.config
    time = node.time

    # 장애물의 공간과 시간 범위 내에 있는지 확인 (1차원)
    within_space = obstacle.space_bounds[0][0] <= config[0] <= obstacle.space_bounds[0][1]
    within_time = obstacle.time_bounds[0] <= time <= obstacle.time_bounds[1]
    if within_space and within_time:
        return True
    return False

def st_rrt_star(X, x_start, X_goal, PTC, t_max, p_goal, P, obstacles):
    T_a = Tree()
    T_b = Tree()
    T_goal = T_b
    start_node = Node(x_start[:-1], x_start[-1])
    start_node.cost = 0 #시작 노드의 비용은 0으로 설정

    T_a.add_node(start_node)
    B = initialize_bound_variables(P)
    solution = None
    num = 0
    # switch가 0 이면 Ta가 시작트리, Tb가 골트리. switch가 1이면 Ta가 골트리, Tb가 시작트리
    iteration = 0

    while iteration < PTC:
        #print(iteration)
        switch = num % 2
        B = update_goal_region(B, P, t_max)
        
        if random.random() < p_goal:
            B = sample_goal(x_start, X_goal, T_goal, t_max, B)  # T_goal로 고정이 맞는 것 같음
        
        x_rand = sample_conditionally(x_start, X, B)
        ext_result, x_new = extend(T_a, x_rand, switch, obstacles)

        if ext_result == "Advanced":  # extend성공한 x_rand가 x_new가 되면서 트리에 추가되어야
            B['samplesInBatch'] += 1
            B['totalSamples'] += 1
            if switch == 1: rewire_tree(T_goal, x_new, obstacles)  #골트리에서만 진행해야돼

            connect_result, x_new_copy = connect(T_b, x_new, switch, obstacles)
            if connect_result == True:  # T_a에 연결한 x_rand가 T_b와도 연결되나 확인   
                new_solution = update_solution(x_new, x_new_copy, T_a, T_b)
                #new solution이 시간이 더 적게 걸리면 교체함.
                if solution is None or new_solution[-1].time < solution[-1].time:
                    solution = new_solution
                    #debugging위해서 프루닝 안 하고 솔루션 바로 냄
                    print('Total iteration until first solution :', iteration)
                    break


                    ##솔루션들의 시간 어떻게 나오나 확인
                    t_max = solution[-1].time
                    print('t_max : ', t_max)
                    B['batchProbability'] = 1
                    prune_trees(t_max, T_a, T_b)
      
        #goal tree 업데이트.
        if switch == 0:
            T_goal = T_b
        else:
            T_goal = T_a
        num += 1
        iteration += 1
        #Swap
        T_a, T_b = T_b, T_a

    return solution, T_a, T_b

def initialize_bound_variables(P):
    return {
        'timeRange': P['rangeFactor'],
        'newTimeRange': P['rangeFactor'],
        'batchSize': P['initialBatchSize'],
        'samplesInBatch': 0,
        'totalSamples': 0,
        'batchProbability': 1,
        'goals': [],
        'newGoals': []
    }

def update_goal_region(B, P, t_max):
    if t_max == float('inf') and B['samplesInBatch'] == B['batchSize']:
        B['timeRange'] = B['newTimeRange']
        B['newTimeRange'] *= P['rangeFactor']
        B['batchSize'] = int((P['rangeFactor'] - 1) * B['totalSamples'] / P['sampleRatio'])
        B['batchProbability'] = (1 - P['sampleRatio']) / P['rangeFactor']
        B['goals'].extend(B['newGoals'])
        B['newGoals'] = []
        B['samplesInBatch'] = 0
    return B

def sample_goal(x_start, X_goal, T_goal, t_max, B):
    q = sample_uniform(X_goal)
    t_min = lower_bound_arrival_time(x_start[:-1], q)
    
    sample_old_batch = random.random() <= B['batchProbability']
    
    if t_max != float('inf'):
        t_lb, t_ub = t_min, t_max
    elif sample_old_batch:
        t_lb, t_ub = t_min, t_min * B['timeRange']
    else:
        t_lb, t_ub = t_min * B['timeRange'], t_min * B['newTimeRange']
    
    if t_ub > t_lb:
        t = random.uniform(t_lb, t_ub)
        goal_node = Node(q, t)
        T_goal.add_node(goal_node)
        if sample_old_batch:
            B['goals'].append(goal_node)
        else:
            B['newGoals'].append(goal_node)
    
    return B

def sample_conditionally(x_start, X, B): #본 함수 PTC만큼 실행됨
    while True:
        q = sample_uniform(X[:-1])  # Sample configuration space
        t_min = x_start[-1] + lower_bound_arrival_time(x_start[:-1], q)
        if random.random() < B['batchProbability']:  # old region 샘플
            t_lb = t_min
            t_ub = max_valid_time(q, B['goals'])  
        else:  # new region 샘플
            t_star_min = max_valid_time(q, B['goals'])
            t_lb = max(t_min, t_star_min)
            t_ub = max_valid_time(q, B['newGoals'])
        #print('db. t_lb, t_ub :', t_lb, t_ub)
        if t_lb < t_ub:
            t = random.uniform(t_lb, t_ub)
            x_rand = Node(q,t)
            # 충돌 검사: 장애물과 충돌하면 다시 샘플링
            if not any(is_in_obstacle(x_rand, obs) for obs in obstacles):
                return x_rand

def extend(T, x_rand, switch, obstacles):
    if switch == 0: 
        x_nearest = T.get_nearest(x_rand.config, x_rand.time)
    else: 
        x_nearest = T.get_nearest_g(x_rand.config, x_rand.time)

    if x_nearest is not None:
        if switch == 0:  # 시작 트리
            if d(x_nearest, x_rand) < float('inf'):
                x_new = Node(x_rand.config, x_rand.time)
                x_new.parent = x_nearest
                x_nearest.children.append(x_new)

                # 보간된 점들에 대한 충돌 검사
                interpolated_points = interpolate(x_nearest, x_new)
                for point in interpolated_points:
                    if any(is_in_obstacle(point, obs) for obs in obstacles):
                        return "Trapped", None  # 충돌 시 연결 중단

                T.add_node(x_new)
                x_new.cost = x_nearest.cost + d(x_nearest, x_new)
                return "Advanced", x_new
        else:  # 골 트리
            if d(x_rand, x_nearest) < float('inf'):
                x_new = Node(x_rand.config, x_rand.time)
                x_new.parent = x_nearest
                x_nearest.children.append(x_new)

                # 보간된 점들에 대한 충돌 검사
                interpolated_points = interpolate(x_nearest, x_new)
                for point in interpolated_points:
                    if any(is_in_obstacle(point, obs) for obs in obstacles):
                        return "Trapped", None  # 충돌 시 연결 중단

                T.add_node(x_new)
                x_new.cost = x_nearest.cost - d(x_new, x_nearest)
                return "Advanced", x_new

    return "Trapped", None

def connect(T, x_new, switch, obstacles):
    if x_new.parent is None:
        print("Error: x_new is not connected to T_a yet.")
        return False

    if switch == 0: 
        x_nearest = T.get_nearest_g(x_new.config, x_new.time)
    else: 
        x_nearest = T.get_nearest(x_new.config, x_new.time)

    if x_nearest is not None:
        x_new_copy = Node(config=x_new.config, time=x_new.time)

        # 보간된 점들에 대한 충돌 검사
        interpolated_points = interpolate(x_nearest, x_new_copy)
        for point in interpolated_points:
            if any(is_in_obstacle(point, obs) for obs in obstacles):
                return False, None  # 충돌 시 연결 중단
        if switch == 0:
            x_new_copy.parent = x_nearest
            x_nearest.children.append(x_new_copy)
            T.add_node(x_new_copy)
            return True, x_new_copy
        elif switch == 1:
            x_new_copy.parent = x_nearest
            x_nearest.children.append(x_new_copy)
            T.add_node(x_new_copy)
            return True, x_new_copy
    return False, None


def rewire_tree(T_goal, x_new, obstacles):
    radius = compute_rewire_radius(len(T_goal.nodes))
    nearby_nodes = T_goal.get_nearest_k(x_new.config, x_new.time, k=int(radius))

    for x_near in nearby_nodes:
        cost = x_new.cost - d(x_near, x_new)
        
        # 보간된 점들에 대한 충돌 검사
        interpolated_points = interpolate(x_new, x_near)
        for point in interpolated_points:
            if any(is_in_obstacle(point, obs) for obs in obstacles):
                continue  # 충돌 시 재연결 중단

        if cost > x_near.cost:
            x_near.parent.children.remove(x_near)
            x_near.parent = x_new
            x_new.children.append(x_near)
            x_near.cost = cost
            update_children_cost(x_near)

def update_children_cost(node):
    for child in node.children:
        child.cost = node.cost - d(child, node)
        update_children_cost(child)

def update_solution(x_new, x_new_copy, T_a, T_b):
    path_a = get_path_to_root(x_new, T_a)
    path_b = get_path_to_root(x_new_copy, T_b)
    # 중간에 겹치는 노드 제외하고 전체 경로 생성
    total_path = path_a[::-1] + path_b[1:]
    # time 속성에 대해 오름차순으로 정렬
    total_path_sorted = sorted(total_path, key=lambda node: node.time)
    
    return total_path_sorted

def get_path_to_root(node, T):
    path = [node]
    while node.parent is not None:
        node = node.parent
        path.append(node)
    return path

def prune_trees(t_max, T_a, T_b):
    #print(t_max)
    T_a.nodes = [n for n in T_a.nodes if n.time <= t_max]
    T_b.nodes = [n for n in T_b.nodes if n.time <= t_max]
    print('Pruning!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

# Helper 함수들
def sample_uniform(X):
    return np.random.uniform(X[:, 0], X[:, 1])

# def lower_bound_arrival_time(q_start, q_goal):
#     return np.linalg.norm(q_goal - q_start) / np.max(V_MAX)
def lower_bound_arrival_time(q_start, q_goal):
    # q_goal - q_start의 각 원소를 대응되는 V_MAX로 나눔
    time_per_dimension = np.abs(q_goal - q_start) / V_MAX
    # 그 중 가장 큰 값을 반환
    return np.max(time_per_dimension)
def max_valid_time(q_rnd, goals):
    if not goals:  # goals가 비어 있을 경우 처리
        return float('inf')  # 기본적으로 매우 큰 값을 반환 (유효한 시간이 없음을 나타냄)
    return max(g.time - min(abs(q_rnd[i] - g.config[i]) / V_MAX[i] for i in range(len(q_rnd))) for g in goals)

def d(x1, x2):
    dt = x2.time - x1.time
    if dt <= 0:
        return float('inf')
    
    dq = x2.config - x1.config
    v = dq / dt
    lam = 0.2
    if np.all(np.abs(v) <= V_MAX):
        return lam * np.linalg.norm(dq) + (1-lam) * dt
    else:
        return float('inf')

def compute_rewire_radius(n):
    if n == 0:
        return 20.0
    return min(20.0, 1.1 * (np.log(n) / n) ** (1/2))

# 1D 시각화 함수
def plot_trees_with_obstacles_1d(T_a, T_b, obstacles):
    print('plot!')
    fig, ax = plt.subplots()

    # Plot T_a (Start Tree)
    for node in T_a.nodes:
        if node.parent is not None:
            parent = node.parent
            ax.plot([node.config[0], parent.config[0]],
                    [node.time, parent.time], color='b')  # 파란색으로 시작 트리 표시
        #ax.text(node.config[0], node.time, f'{node.cost:.2f}', color='blue')

    # Plot T_b (Goal Tree)
    for node in T_b.nodes:
        if node.parent is not None:
            parent = node.parent
            ax.plot([node.config[0], parent.config[0]],
                    [node.time, parent.time], color='r')  # 빨간색으로 목표 트리 표시
        #ax.text(node.config[0], node.time, f'{node.cost:.2f}', color='red')

    # 장애물 추가 (장애물은 1D에서 구간으로 시각화)
    for obstacle in obstacles:
        x_min, x_max = obstacle.space_bounds[0]
        t_min, t_max = obstacle.time_bounds

        # Plotting the obstacle as a shaded area
        ax.fill_betweenx([t_min, t_max], x_min, x_max, color='gray', alpha=0.8)

    # Set labels and title
    ax.set_xlabel('Configuration (1D)')
    ax.set_ylabel('Time')
    ax.set_title('Start Tree (T_a) and Goal Tree (T_b) in 1D Space with Obstacles')

    plt.legend(['T_a (Start Tree)', 'T_b (Goal Tree)', 'Obstacles'])
    plt.show()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants
V_MAX = np.array([0.1])  # Maximum velocities for 1-DoF manipulator

# Main 실행 부분
if __name__ == "__main__":
    X = np.array([[-np.pi / 2, np.pi / 2]] + [[0, float('inf')]])  # Configuration space + time
    x_start = np.array([0, 0])  # Start configuration + time
    X_goal = np.array([[0.25, 0.25]])  # Goal configuration + time

    t_max = float('inf')
    p_goal = 0.1
    P = {
        'rangeFactor': 2.0,
        'initialBatchSize': 25,
        'sampleRatio': 0.5
    }
    
    # 장애물 생성 (1차원 구성 공간에 맞게 수정)
    obstacles = [
        Obstacle([[0.1, 0.2]], [0, 20]),
        Obstacle([[0.1, 0.2]], [25,60])
    ]
    
    solution, T_a, T_b = st_rrt_star(X, x_start, X_goal, 1000, t_max, p_goal, P, obstacles)
    
    if solution:
        print("Solution found!")
        for node in solution:
            print(f"Configuration: {node.config}, Time: {node.time}")
        plot_trees_with_obstacles_1d(T_a, T_b, obstacles)  # 1D 시각화 함수 사용
    else:
        print("No solution found, visualizing trees...")
        plot_trees_with_obstacles_1d(T_a, T_b, obstacles)


