import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# -----------------------------
# 已有的初值与参数函数
# -----------------------------
def get_initial_condition(ic, x):
    """
    根据 ic 选择初始条件：
      ic = 1: ψ₁(x,0) = exp(-x²/10)
      ic = 2: ψ₂(x,0) = 2*sin(x)/(exp(x)+exp(-x)) = sin(x)/cosh(x)
      ic = 3: ψ₃(x,0) = 2*cos(x)/(exp(x)+exp(-x)) = cos(x)/cosh(x)
    """
    if ic == 1:
        return np.exp(-x**2/10)
    elif ic == 2:
        return 2 * np.sin(x) / (np.exp(x) + np.exp(-x))
    elif ic == 3:
        return 2 * np.cos(x) / (np.exp(x) + np.exp(-x))
    else:
        raise ValueError("初值索引必须为 1, 2 或 3")

def sech(x):
    return 1/np.cosh(x)

# -----------------------------
# 伪谱方法与时间分裂步骤
# -----------------------------
def step_linear(psi, dt, k):
    """
    对动能部分 T = -0.5 ψ_xx 进行时间推进（傅里叶伪谱计算）
    """
    psi_hat = np.fft.fft(psi)
    psi_hat = np.exp(-1j * dt * 0.5 * (k**2)) * psi_hat
    return np.fft.ifft(psi_hat)

def step_nonlinear(psi, dt, V, g, kappa):
    """
    对非线性部分 N = V(x) + g|ψ|² + κ|ψ|⁴ 进行推进
    """
    phase = np.exp(-1j * dt * (V + g * np.abs(psi)**2 + kappa * np.abs(psi)**4))
    return phase * psi

def step_strang(psi, dt, k, V, g, kappa):
    """
    二阶Strang分裂：ψ(t+dt) = exp(-i dt/2 N) exp(-i dt T) exp(-i dt/2 N) ψ(t)
    """
    psi = step_nonlinear(psi, dt/2, V, g, kappa)
    psi = step_linear(psi, dt, k)
    psi = step_nonlinear(psi, dt/2, V, g, kappa)
    return psi

def step_fourth_order(psi, dt, k, V, g, kappa):
    """
    四阶分裂方案（采用 Yoshida 系数）：
    定义:
      a₁ = a₃ = 1/(2-2^(1/3))
      a₂ = -2^(1/3)/(2-2^(1/3))
      b₁ = b₃ = 1/(2-2^(1/3))
      b₂ = -2^(1/3)/(2-2^(1/3))
    更新：
      ψ = exp(-i b₁ dt N) exp(-i a₁ dt T) exp(-i b₂ dt N) exp(-i a₂ dt T)
          exp(-i b₁ dt N) exp(-i a₁ dt T) exp(-i b₂ dt N) exp(-i a₂ dt T)
          exp(-i b₁ dt N) ψ.
    """
    c = 2 - 2**(1/3)
    a1 = 1.0 / c
    a2 = - 2**(1/3) / c
    b1 = 1.0 / c
    b2 = - 2**(1/3) / c
    psi = step_nonlinear(psi, b1*dt, V, g, kappa)
    psi = step_linear(psi, a1*dt, k)
    psi = step_nonlinear(psi, b2*dt, V, g, kappa)
    psi = step_linear(psi, a2*dt, k)
    psi = step_nonlinear(psi, b1*dt, V, g, kappa)  # 对称性：b₃ = b₁
    psi = step_linear(psi, a2*dt, k)
    psi = step_nonlinear(psi, b2*dt, V, g, kappa)
    psi = step_linear(psi, a1*dt, k)
    psi = step_nonlinear(psi, b1*dt, V, g, kappa)
    return psi

# -----------------------------
# 求解器（自定义参数版）
# -----------------------------
def solve_GPE_custom(init_func, x, dt, t_final, order, g, kappa, V):
    """
    求解GPE：x ∈ [-10,10]，t ∈ [0,t_final]
    参数：
      init_func: 用于计算初值 ψ₀(x) 的函数
      dt: 时间步长
      t_final: 终止时间
      order: 分裂方法阶数（支持2或4）
      g, kappa, V: 分别为常数和势函数（在空间 x 上的数组），用于非线性部分
    返回：
      t: 时间数组
      psi_record: 数值解，形状为 (Nt, len(x))
    """
    Nx = len(x)
    dx = x[1] - x[0]
    k = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    Nt = int(t_final/dt) + 1
    t = np.linspace(0, t_final, Nt)
    psi0 = init_func(x)
    psi = psi0.astype(complex)
    psi_record = np.zeros((Nt, Nx), dtype=complex)
    psi_record[0, :] = psi
    for n in range(1, Nt):
        if order == 2:
            psi = step_strang(psi, dt, k, V, g, kappa)
        elif order == 4:
            psi = step_fourth_order(psi, dt, k, V, g, kappa)
        else:
            raise ValueError("目前仅支持二阶和四阶分裂方法（order=2 或 4）")
        psi_record[n, :] = psi
    return t, psi_record

# -----------------------------
# 数据生成及保存（拼接为 batch 数组）
# -----------------------------
def generate_and_save_training_data(num_orbits=6000, Nx=128, dt=0.005, t_final=5.0, order=2, num_time_samples=100, save_path="training_data.npy"):
    """
    生成 num_orbits 条轨道数据，求解区间为 [-10,10]。
    每条轨道：
      - 初值条件固定使用2号初值（sin(x)/cosh(x)）
      - g 和 κ 均从 [1,2.5] 均匀采样
      - 潜在势 V(x)= a*(x-x₀)² + b*cos(c*(x-x₀))²，其中
            a ∈ [0.1,0.3],  b ∈ [0.5,2], c ∈ [0.5,2],  x₀ ∈ [-3,3]（均匀采样）
      - 使用 Nx=128, dt=0.005, t_final=5.0 进行模拟
      - 从完整轨道中随机采样 num_time_samples 个时间点（排序后）作为样本
    最后，将所有轨道的 y（shape: (num_orbits, num_time_samples, Nx)），
    g（(num_orbits,)）， κ（(num_orbits,)）和 V（(num_orbits, Nx)）分别拼接，
    同时也保存 t（(num_orbits, num_time_samples)）和势参数 potential_params，
    并保存为一个 npy 文件。
    """
    all_y = []           # 保存 |ψ|，shape: (num_time_samples, Nx)
    all_t = []           # 保存采样的 t 数组，shape: (num_time_samples,)
    all_g = []           # 保存 g，shape: (num_orbits,)
    all_kappa = []       # 保存 κ，shape: (num_orbits,)
    all_V = []           # 保存 V，shape: (Nx,)
    all_pot_params = []  # 保存势函数参数字典

    x = np.linspace(-10, 10, Nx)
    init_func = lambda x: get_initial_condition(2, x)  # 固定使用2号初值
    Nt = int(t_final/dt) + 1

    for orbit in tqdm(range(num_orbits)):
        # 随机采样参数
        g_val = 2
        kappa_val = 2
        a = np.random.uniform(0.1, 0.3)
        b = np.random.uniform(0.5, 2)
        c = np.random.uniform(0.5, 2)
        x0 = np.random.uniform(-3, 3)
        V = a * (x - x0)**2 + b * (np.cos(c * (x - x0)))**2 

        # 保存势函数参数
    

        # 求解轨道
        t_arr, psi_record = solve_GPE_custom(init_func, x, dt, t_final, order, g_val, kappa_val, V)
        psi_abs = np.abs(psi_record)  # shape: (Nt, Nx)
        # 随机采样 num_time_samples 个时间点
        selected_indices = np.sort(np.random.choice(np.arange(Nt), size=num_time_samples, replace=False))
        t_selected = t_arr[selected_indices]
        y_selected = psi_abs[::10, :]  # shape: (num_time_samples, Nx)

        all_y.append(y_selected)
        all_g.append(g_val)
        all_kappa.append(kappa_val)
        all_V.append(V)

        if (orbit+1) % 100 == 0:
            print(f"Completed {orbit+1}/{num_orbits} orbits.")

    # 拼接成数组
    all_y = np.stack(all_y, axis=0)         # shape: (num_orbits, num_time_samples, Nx)
    all_g = np.array(all_g)                 # shape: (num_orbits,)
    all_kappa = np.array(all_kappa)         # shape: (num_orbits,)
    all_V = np.stack(all_V, axis=0)           # shape: (num_orbits, Nx)
    # 注意 potential_params 为列表

    # 保存为一个字典
    data_dict = {
        'y': all_y,
        'g': all_g,
        'kappa': all_kappa,
        'V': all_V
    }
    np.save(save_path, data_dict, allow_pickle=True)
    print(f"数据生成完毕，并保存在 {save_path} 中。")
    return data_dict

# -----------------------------
# 可视化单条轨道数据
# -----------------------------
def visualize_sample(sample):
    """
    可视化一条轨道数据中势函数 V(x) 与 |ψ(x,t)| (用 imshow) 的图像
    """
    V_val = sample['V']         # shape: (Nx,)
    g_val = sample['g']
    kappa_val = sample['kappa']
    y_selected = sample['y']      # shape: (num_time_samples, Nx)
    Nx = V_val.shape[0]
    x = np.linspace(-10, 10, Nx)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, V_val, 'b-', lw=2)
    plt.xlabel("x")
    plt.ylabel("V(x)")
    #plt.title(f"Potential V(x)\n g={g_val:.3f}, κ={kappa_val:.3f}\n params: a={pot_params['a']:.3f}, b={pot_params['b']:.3f}, c={pot_params['c']:.3f}, x0={pot_params['x0']:.3f}")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    im = plt.imshow(y_selected,  aspect='auto', cmap='viridis')
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title("|ψ(x,t)| (selected time samples)")
    plt.colorbar(im, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

# -----------------------------
# 主程序入口
# -----------------------------
if __name__ == '__main__':
    np.random.seed(42)  # 固定随机种子，确保可复现
    # 生成数据并保存
    data_dict = generate_and_save_training_data(num_orbits=6000, Nx=128, dt=0.005, t_final=5.0, order=2, num_time_samples=100, save_path="training_data_Schrodinger.npy")
    
