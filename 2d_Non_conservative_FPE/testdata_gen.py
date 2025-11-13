import numpy as np
import matplotlib.pyplot as plt
from fplanck import fokker_planck, boundary, gaussian_pdf
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 基本参数设置
nm = 1e-9
viscosity = 8e-4
radius = 50 * nm
drag = 6 * np.pi * viscosity * radius  # 固定的 drag 值
temperature = 300
extent = [800 * nm, 800 * nm]
resolution = 10 * nm
boundary_condition = boundary.reflecting

# 定义 F 函数，默认参数作为占位
def F(x, y, L=100*nm, a=1, b=1, c=1, d=1):
    rad = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    Fphi = 1e-12 * rad / L * np.exp(-rad / L * b) * a
    Frad = 1e-12 * (1 - rad / L) * np.exp(-rad / L * d) * c
    Fx = -np.sin(phi) * Fphi + np.cos(phi) * Frad
    Fy =  np.cos(phi) * Fphi + np.sin(phi) * Frad
    return np.array([Fx, Fy])

# 模拟参数
M = 400            # 模拟次数
N_THREADS = 8       # 线程数

# 单次模拟的函数
def run_simulation(sim_index):
    # 随机采样 F 的参数
    L_rand = np.random.uniform(50 * nm, 150 * nm)
    a_rand = np.random.uniform(0.5, 2)
    b_rand = np.random.uniform(0.5, 2)
    c_rand = np.random.uniform(0.5, 2)
    d_rand = np.random.uniform(0.5, 2)
    
    # 固定参数传入 force 参数（注意：通过 lambda 固定参数）
    force_func = lambda x, y: F(x, y, L=L_rand, a=a_rand, b=b_rand, c=c_rand, d=d_rand)
    
    # 创建模拟对象，使用固定的 drag
    sim = fokker_planck(temperature=temperature,
                        drag=drag,
                        extent=extent,
                        resolution=resolution,
                        boundary=boundary_condition,
                        force=force_func)
    
    # 初始 PDF（这里采用与例子中相同的参数）
    pdf = gaussian_pdf(center=(-150 * nm, -150 * nm), width=30 * nm)
    p0 = pdf(*sim.grid)
    
    # 使用 1000 个时间步长进行传播，时间间隔取 10e-3（与原 F 模拟代码保持一致）
    Nsteps = 500
    dt = 10e-3
    time, Pt = sim.propagate_interval(pdf, dt, Nsteps=Nsteps)
    
    # 随机抽取 100 个时刻的 Pt 作为 y_selected
    selected_indices = np.sort(np.random.choice(len(time), size=100, replace=False))
    t_selected = time[selected_indices]
    y_selected = Pt[selected_indices, :, :]
    
    # 计算网格上 F 的值
    F_field = force_func(*sim.grid)
    grid = sim.grid
    
    print(f"Simulation {sim_index + 1} completed.")
    return t_selected, grid, y_selected, F_field

# 存储所有模拟结果的列表
time_array = []         # 每个模拟中抽取的时间数组（形状：(100,)）
grid_array = []         # 每个模拟的网格（形状：(2, Nx, Ny)）
trajectories_array = [] # 每个模拟抽取的 y_selected（形状：(100, Nx, Ny)）
F_array = []            # 每个模拟计算得到的 F（形状：(2, Nx, Ny)）

# 多线程运行模拟
with ThreadPoolExecutor(max_workers=N_THREADS) as executor:
    futures = [executor.submit(run_simulation, i) for i in range(M)]
    for future in tqdm(as_completed(futures), total=M):
        t_selected, grid, y_selected, F_field = future.result()
        time_array.append(t_selected)
        grid_array.append(grid)
        trajectories_array.append(y_selected)
        F_array.append(F_field)

# 转换为 numpy 数组（注意：grid 在每次模拟中相同，可以只记录一次）
time_array = np.array(time_array)            # 形状: (M, 100)
grid_array = np.array(grid_array)            # 形状: (M, 2, Nx, Ny)
trajectories_array = np.array(trajectories_array)  # 形状: (M, 100, Nx, Ny)
F_array = np.array(F_array)                    # 形状: (M, 2, Nx, Ny)

# 保存数据集到 .npz 文件
np.savez(
    "test_dataset_2D_drift.npz",
    time=time_array,
    grid=grid_array,
    trajectories=trajectories_array,
    F=F_array
)
