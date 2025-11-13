import numpy as np
import matplotlib.pyplot as plt
from fplanck import fokker_planck, boundary, gaussian_pdf, combine, gaussian_potential
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
from tqdm import tqdm
nm = 1e-9
viscosity = 8e-4
radius = 50 * nm
drag = 6 * np.pi * viscosity * radius
temperature = 300
extent = 800*nm
resolution = 10 * nm
boundary_condition = boundary.reflecting
M = 100  # Number of simulations

# Function to generate random Gaussian potential parameters
def random_gaussian_params():
    while True:
        # Random centers in the range [-100*nm, 100*nm]
        centers = np.random.uniform(-150 * nm, 150 * nm, size=(3))
        
        # Ensure distances between all pairs are at least 30*nm
        if all(np.abs(centers[i] - centers[j]) > 80 * nm for i in range(3) for j in range(i+1, 3)):
            break

    # Random widths in the range [20*nm, 80*nm]
    widths = np.random.uniform(20 * nm, 80 * nm, size=3)
    As = np.random.uniform(1e-20,2e-20,size = 3)
    viscosity_fact =  np.random.uniform(1,2,size = 1)
    return centers, widths, As, viscosity_fact


# Storage for the dataset
time_array = []
grid_array = []
trajectories_array = []
potential_array = []
drag_array = []
for i in tqdm(range(M)):
    # Randomly sample Gaussian potential parameters
    centers, widths, As, viscosity_fact = random_gaussian_params()

    # Create the combined potential
    U = combine(
        gaussian_potential(center=(centers[0]), width=widths[0], amplitude=As[0]*viscosity_fact[0]),
        gaussian_potential(center=(centers[1]), width=widths[1], amplitude=As[1]*viscosity_fact[0]),
        gaussian_potential(center=(centers[2]), width=widths[2], amplitude=As[2]*viscosity_fact[0])
    )
    drag_temp = drag*viscosity_fact[0]
    # Initialize the Fokker-Planck simulation
    sim = fokker_planck(
        temperature=temperature,
        drag=drag_temp,
        extent=extent,
        resolution=resolution,
        boundary=boundary_condition,
        potential=U
    )
    
    # Initial PDF
    pdf = gaussian_pdf(center=(0 * nm), width=50 * nm)
    p0 = pdf(*sim.grid)

    # Time evolution
    Nsteps = 400
    time, Pt = sim.propagate_interval(pdf, 2e-3, Nsteps=Nsteps)

    # Calculate the potential values on the grid
    U_grid = U(*sim.grid)

    # Randomly select 100 time points
    selected_indices = np.sort(np.random.choice(range(len(time)), size=100, replace=False))
    t_selected = time[selected_indices]
    y_selected = Pt[selected_indices, :]  # Shape: (100, Nx, Ny)

    # Store the processed data
    time_array.append(t_selected)
    grid_array.append(sim.grid)
    trajectories_array.append(y_selected)
    potential_array.append(U_grid)
    drag_array.append(drag_temp)
    print(f"Simulation {i + 1}/{M} completed.")

time_array = np.array(time_array)  # Shape: (M, 100)
grid_array = np.array(grid_array)  # Shape: (M, 2, Nx, Ny)
trajectories_array = np.array(trajectories_array)  # Shape: (M, 100, Nx, Ny)
potential_array = np.array(potential_array)  # Shape: (M, Nx, Ny)
drag_array = np.array(drag_array)
# # Save the dataset
np.savez(
    "dataset_1D_drift_diffusion_8000.npz",
    time=time_array,
    grid=grid_array,
    trajectories=trajectories_array,
    potential=potential_array,
    drag = drag_array
)
print("Dataset saved to 'dataset_1D_drift_diffusion.npz'.")



data = np.load("dataset_1D_drift_diffusion.npz")

# # Extract the arrays
time_array = data["time"]  # Shape: (M, 100)
grid_array = data["grid"]  # Shape: (M, 2, Nx, Ny)
trajectories_array = data["trajectories"]  # Shape: (M, 100, Nx, Ny)
potential_array = data["potential"]  # Shape: (M, Nx, Ny)
drag_array = data['drag']
print(time_array.shape)
print(grid_array.shape)
print(trajectories_array.shape)
print(potential_array.shape)
print(drag_array)