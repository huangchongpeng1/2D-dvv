import numpy as np
import pandas as pd
from scipy.sparse.linalg import  lsmr
from geopy.distance import geodesic
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from joblib import Parallel, delayed
import os



########################################################
#                                                      # 
#            Code for 2D dv/v solving                  # 
#            Author: Chongpeng Huang                   # 
#            Email: hchongpeng@gmail.com               # 
#                                                      #
########################################################



# Set current working directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Initialize parameters
year = 2020
start_date = datetime(year, 1, 1)
c = 2  # Constant velocity (km/s)

# Grid generation
lat_start, lon_start = 30.8, -104.0
lat_end, lon_end = 31.7, -102.8
resolution = 2  # km

lat_range_km = geodesic((lat_end, lon_start), (lat_start, lon_start)).meters / 1000 
lon_range_km = geodesic((lat_start, lon_end), (lat_start, lon_start)).meters / 1000

n_lat = int(np.ceil(lat_range_km / resolution)) + 1
n_lon = int(np.ceil(lon_range_km / resolution)) + 1
n_points = n_lat * n_lon

adjusted_lat_length = (n_lat - 1) * resolution
adjusted_lon_length = (n_lon - 1) * resolution

lats = np.linspace(0, adjusted_lat_length, n_lat)
lons = np.linspace(0, adjusted_lon_length, n_lon)
lon_grid, lat_grid = np.meshgrid(lons, lats)

# Restore latitude and longitude axes
lon_degree_km = geodesic((lat_start, lon_start), (lat_start, lon_start + 1)).meters / 1000
lat_degree_km = geodesic((lat_start, lon_start), (lat_start + 1, lon_start)).meters / 1000

adjusted_lon_length = (n_lon - 1) * resolution
adjusted_lat_length = (n_lat - 1) * resolution

lon_min = lon_start
lon_max = lon_start + adjusted_lon_length / lon_degree_km
lat_min = lat_start
lat_max = lat_start + adjusted_lat_length / lat_degree_km

# Parameter settings
station_number = 25

corr_len = 15  # Correlation length
std_model = 0.01  # Model standard deviation
scaling_factor = resolution
freq_max, freq_min = 4.0, 0.1 # Frequency range

x_grid_re = lon_grid.ravel()
y_grid_re = lat_grid.ravel()
n_points = len(x_grid_re)

# Calculate distance matrix
y_diff = y_grid_re[:, np.newaxis] - y_grid_re
x_diff = x_grid_re[:, np.newaxis] - x_grid_re
dist = np.sqrt(y_diff**2 + x_diff**2)

# Calculate model covariance matrix
lambda0 = resolution  
p1 = (std_model * (lambda0 / corr_len)) ** 2
smooth = np.exp(-dist / corr_len)
C_M = p1 * smooth  # Model covariance matrix

# Read station data
station_file_path = os.path.join(current_dir, "station.csv")
file_station = pd.read_csv(
    station_file_path,
    dtype=str
)

file_station.rename(columns={
    "station": "Station_ID",
    "longitude": "Longitude",
    "latitude": "Latitude"
}, inplace=True)

file_station["Longitude"] = pd.to_numeric(file_station["Longitude"], errors="coerce")
file_station["Latitude"] = pd.to_numeric(file_station["Latitude"], errors="coerce")
file_station = file_station.dropna(subset=["Longitude", "Latitude"])
coords = file_station.set_index("Station_ID").to_dict(orient="index")

def iterative_tarantola_valette_solution(G, d, C_D, C_M, m0=None, max_iter=30, tol=1e-6):
    """
    Iterative version of Tarantola and Valette (1982) Bayesian inversion
    m = m0 + Cm G^T (G Cm G^T + Cd)^(-1) (d - G m0)
    
    Parameters:
    - G: Sensitivity kernel matrix
    - d: Observation data
    - C_D: Data covariance matrix
    - C_M: Model covariance matrix
    - m0: Prior model (default is zero vector, updated with iterations)
    - max_iter: Maximum number of iterations
    - tol: Convergence tolerance
    
    Returns:
    - m_est: Final estimated model
    - history: Dictionary containing results of each iteration
    """
    n_obs, n_params = G.shape
    
    # Set prior model as zero vector
    if m0 is None:
        m0 = np.zeros(n_params)
    
    # Initialize current model
    m_current = m0.copy()
    
    # Store iteration history
    history = {
        'iterations': [],
        'models': [],
        'residuals': [],
        'residual_norms': [],
        'model_changes': [],
        'rms_values': []
    }
    
    print(f"Starting iterative Tarantola-Valette inversion, maximum iterations: {max_iter}")
    print("=" * 60)
    
    # Handle diagonal C_D
    if C_D.ndim == 1:
        C_D_matrix = np.diag(C_D)
    else:
        C_D_matrix = C_D
    
    for iter in range(max_iter):
        # Calculate residual: d - G m_current
        residual = d - G @ m_current
        residual_norm = np.linalg.norm(residual)
        rms = np.sqrt(np.mean(residual**2))
        
        # Store current iteration information
        history['iterations'].append(iter)
        history['models'].append(m_current.copy())
        history['residuals'].append(residual.copy())
        history['residual_norms'].append(residual_norm)
        history['rms_values'].append(rms)
        
        # Output current iteration information
        print(f"Iteration {iter+1}/{max_iter}: Residual norm = {residual_norm:.6e}, RMS = {rms:.6e}")
        
        # Check convergence condition
        if iter > 0:
            model_change = np.linalg.norm(history['models'][-1] - history['models'][-2])
            history['model_changes'].append(model_change)
            
            if model_change < tol:
                print(f"Converged at iteration {iter+1}, model change = {model_change:.2e} < tolerance {tol}")
                break
        else:
            history['model_changes'].append(np.inf)  # No model change in first iteration
        
        # Calculate A = G Cm G^T + Cd
        G_Cm_GT = G @ C_M @ G.T
        A = G_Cm_GT + C_D_matrix
        

        # Use linear solver to avoid direct inversion
        result = lsmr(A, residual)
        x = result[0]
        # Calculate model update
        update_term = C_M @ G.T @ x
        m_new = m_current + update_term

        
        # Update current model
        m_current = m_new
    
    print(f"Iteration completed, final iteration count: {len(history['iterations'])}")
    print(f"Final residual norm: {history['residual_norms'][-1]:.6e}")
    print(f"Final RMS: {history['rms_values'][-1]:.6e}")
    
    return m_current, history

def calculate_data_std(coherence, freq_c, t1, t2):
    """
    Calculate data standard deviation
    """
    # Calculate angular frequency
    omega_c = 2 * np.pi * freq_c
    
    # Calculate T (inverse of center frequency)
    T = 1.0 / freq_c
    
    # Calculate numerator and denominator
    numerator = 6 * np.sqrt(np.pi / 2) * T
    denominator = (omega_c ** 2) * (t2**3 - t1**3)
    
    # Calculate standard deviation
    std_d = (np.sqrt(1 - coherence**2) / (2 * coherence)) * np.sqrt(numerator / denominator)
    
    return std_d

def process_day_iterative_tarantola_valette(day0, max_iter=30):
    """
    Process single day data using iterative Tarantola-Valette method
    """
    day1 = start_date + timedelta(days=day0)
    day_str = day1.strftime("%Y-%m-%d")
    
    # Read data
    data_dir = os.path.join(current_dir, "input")
    csv_file_path = os.path.join(data_dir, f"dt_{day_str}.csv")
    
    try:
        # Read CSV file with header
        text_station = pd.read_csv(csv_file_path)
        
        # Extract column data using column names
        station_lag_time = text_station['time']
        station_delay = text_station['delay']
        station_coherence = text_station['coherence']
        station0 = text_station['station1']
        station1 = text_station['station2']
        
    except FileNotFoundError:
        print(f"File not found: dt_{day_str}.csv")
        return None, "File not found", None
    except KeyError as e:
        print(f"Missing column in CSV file {day_str}.csv: {e}")
        return None, f"Missing column: {e}", None
    
    # Filter data based on coherence coefficient (>= 0.7)
    coherence_mask = station_coherence >= 0.7
    station_lag_time = station_lag_time[coherence_mask]
    station_delay = station_delay[coherence_mask]
    station_coherence = station_coherence[coherence_mask]
    station0 = station0[coherence_mask]
    station1 = station1[coherence_mask]
    
    # Output filtering information
    original_count = len(text_station)
    filtered_count = len(station_lag_time)
    print(f"Date {day_str}: Original data {original_count} rows, filtered {filtered_count} rows (coherence >= 0.7)")
    
    if filtered_count == 0:
        print(f"Warning: Date {day_str} has no qualified data after filtering, skipping")
        return None, "No data after coherence filtering", None
    
    # Calculate delay time ratio
    station_delay_ratio = station_delay / station_lag_time
    
    text_station_length = len(station_lag_time)
    
    # Initialize result arrays
    k_all = np.zeros((text_station_length, n_points))
    cd_all = np.zeros(text_station_length)
    
    # Calculate center frequency and time window parameters
    freq_c = (freq_max + freq_min) / 2
    coda_range = 10  # moving time window (s)
    
    # Main calculation loop
    for idx in range(text_station_length):
        t = station_lag_time.iloc[idx] if hasattr(station_lag_time, 'iloc') else station_lag_time[idx]
        st1 = station0.iloc[idx] if hasattr(station0, 'iloc') else station0[idx]
        st2 = station1.iloc[idx] if hasattr(station1, 'iloc') else station1[idx]
        coherence = station_coherence.iloc[idx] if hasattr(station_coherence, 'iloc') else station_coherence[idx]
        
        # Extract longitude and latitude from dictionary
        lon1 = coords[st1]["Longitude"] 
        lat1 = coords[st1]["Latitude"]   
        lon2 = coords[st2]["Longitude"]  
        lat2 = coords[st2]["Latitude"]   
        
        # Calculate geographical distance
        x1 = geodesic((lat_start, lon1), (lat_start, lon_start)).meters / 1000
        y1 = geodesic((lat1, lon_start), (lat_start, lon_start)).meters / 1000
        x2 = geodesic((lat_start, lon2), (lat_start, lon_start)).meters / 1000
        y2 = geodesic((lat2, lon_start), (lat_start, lon_start)).meters / 1000
        
        d = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
       
        # Calculate data standard deviation
        t1 = t - coda_range/2  # Time window start
        t2 = t + coda_range/2  # Time window end
        std_d = calculate_data_std(coherence, freq_c, t1, t2)
        
        # Data variance (diagonal elements of Cd)
        cd_all[idx] = std_d ** 2
        
        # Calculate sensitivity kernel function
        a = np.sqrt(c**2 * t**2 - d**2)
        
        d1 = np.sqrt((lon_grid - x1)**2 + (lat_grid - y1)**2)
        d2 = np.sqrt((lon_grid - x2)**2 + (lat_grid - y2)**2)
        
        d12 = (x1 - x2) * (x1 - lon_grid) + (y1 - y2) * (y1 - lat_grid)
        d02 = (lon_grid - x2) * (x1 - x2) + (lat_grid - y2) * (y1 - y2)
        
        denom_left = d1 * (1 - d12 / (c * t * d1)) 
        denom_right = d2 * (1 - d02 / (c * t * d2))       
        
        kk = (a) * (1 / denom_left + 1 / denom_right)
        
        k1 = kk.ravel() 
        k_all[idx] = k1
    
    # Solve using iterative Tarantola-Valette method
    m0 = np.zeros(n_points)

    m, history = iterative_tarantola_valette_solution(
        k_all, station_delay_ratio.values, cd_all, C_M, m0, max_iter
    )
    method_used = f'Iterative Tarantola-Valette ({max_iter} iterations)'

    # Reshape result
    xx = m.reshape(n_lat, n_lon)
    
    # Plot iteration convergence
    if history is not None:
        plot_iteration_convergence(history, day_str)
    
    # Plot final result
    plot_daily_result(xx, day_str, method_used)
    
    return xx, method_used, history

def plot_iteration_convergence(history, day_str):
    """
    Plot iteration convergence curves
    """
    iterations = history['iterations']
    residual_norms = history['residual_norms']
    rms_values = history['rms_values']
    
    if len(iterations) > 1:
        # Create output directory
        output_dir = os.path.join(current_dir, "plot_convergence")
        os.makedirs(output_dir, exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Residual norm convergence curve
        ax1.semilogy(iterations, residual_norms, 'bo-', linewidth=2, markersize=6)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Residual Norm (log scale)')
        ax1.set_title(f'Residual Norm Convergence Curve - {day_str}')
        ax1.grid(True, alpha=0.3)
        
        # RMS convergence curve
        ax2.semilogy(iterations, rms_values, 'ro-', linewidth=2, markersize=6)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('RMS Error (log scale)')
        ax2.set_title(f'RMS Error Convergence Curve - {day_str}')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"convergence_{day_str}.png"), dpi=300)
        plt.close(fig)
        
        print(f"Convergence curve saved: {os.path.join(output_dir, f'convergence_{day_str}.png')}")
        
        # Output convergence statistics
        print("Convergence statistics:")
        print(f"  Initial residual norm: {residual_norms[0]:.6e}")
        print(f"  Final residual norm: {residual_norms[-1]:.6e}")
        print(f"  Residual reduction ratio: {(residual_norms[0] - residual_norms[-1]) / residual_norms[0] * 100:.2f}%")
        print(f"  Initial RMS: {rms_values[0]:.6e}")
        print(f"  Final RMS: {rms_values[-1]:.6e}")
        print(f"  RMS reduction ratio: {(rms_values[0] - rms_values[-1]) / rms_values[0] * 100:.2f}%")

def plot_daily_result(xx, day_str, method_used):
    """
    Plot and save daily inversion result
    """
    # Create directory
    output_dir = os.path.join(current_dir, "plot_day")
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot daily result
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(
        xx, 
        origin='lower', 
        cmap='seismic', 
        vmin=-np.abs(xx).max(), 
        vmax=np.abs(xx).max(),
        extent=[lon_min, lon_max, lat_min, lat_max]
    )
    
    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label('dv/v')
    plt.title(f'dv/v {day_str} ({method_used})')
    plt.xlabel('Longitude (°)')
    plt.ylabel('Latitude (°)')
    
    lon_ticks = np.arange(lon_min, lon_max, 0.2)
    lat_ticks = np.arange(lat_min, lat_max, 0.2)
    ax.set_xticks(lon_ticks)
    ax.set_yticks(lat_ticks)
    ax.tick_params(axis='x', rotation=45)
    
    plt.savefig(os.path.join(output_dir, f"dvv{day_str}.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Daily plot saved: {os.path.join(output_dir, f'dvv{day_str}.png')}")


def calculate_final_sum(all_results):
    """
    Calculate the sum of all daily results
    """
    # Extract valid results (skip None values)
    valid_results = [result for result in all_results if result[0] is not None]
    
    if not valid_results:
        print("No valid results to sum")
        return None, 0
    
    # Extract the model arrays
    model_arrays = [result[0] for result in valid_results]
    
    # Calculate the sum
    final_sum = np.sum(model_arrays, axis=0)
    
    return final_sum, len(valid_results)

def plot_final_sum(final_sum, num_days):
    """
    Plot and save the final sum of all daily results
    """
    # Create directory
    output_dir = os.path.join(current_dir, "plot_final")
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot final sum
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(
        final_sum, 
        origin='lower', 
        cmap='seismic', 
        vmin=-np.abs(final_sum).max(), 
        vmax=np.abs(final_sum).max(),
        extent=[lon_min, lon_max, lat_min, lat_max]
    )
    
    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label('dv/v')
    plt.title(f'Sum of dv/v for {num_days} days (Iterative Tarantola-Valette method)')
    plt.xlabel('Longitude (°)')
    plt.ylabel('Latitude (°)')
    
    lon_ticks = np.arange(lon_min, lon_max, 0.2)
    lat_ticks = np.arange(lat_min, lat_max, 0.2)
    ax.set_xticks(lon_ticks)
    ax.set_yticks(lat_ticks)
    ax.tick_params(axis='x', rotation=45)
    
    plt.savefig(os.path.join(output_dir, "final_sum.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Final sum plot saved: {os.path.join(output_dir, 'final_sum.png')}")

def save_results(final_sum, all_results):
    """
    Save the final results and daily data
    """
    # Create directory
    data_dir = os.path.join(current_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Save final sum
    if final_sum is not None:
        final_sum_path = os.path.join(data_dir, "final_sum_iterative_tarantola_valette.npy")
        np.save(final_sum_path, final_sum)
        print(f"Final sum saved: {final_sum_path}")
    
    # Save daily results
    daily_results = []
    for i, (xx, method_used, history) in enumerate(all_results):
        if xx is not None:
            # Save daily model
            model_path = os.path.join(data_dir, f"model_day{i}.npy")
            np.save(model_path, xx)
            daily_results.append({
                'day': i,
                'method': method_used,
                'model_shape': xx.shape,
                'model_path': model_path
            })
            
            # Save iteration history if available
            if history is not None:
                history_path = os.path.join(data_dir, f"history_day{i}.npy")
                np.save(history_path, history, allow_pickle=True)
                daily_results[-1]['history_path'] = history_path
            else:
                daily_results[-1]['history_path'] = None
    
    # Save summary
    summary_path = os.path.join(data_dir, "processing_summary_iterative.csv")
    summary_df = pd.DataFrame(daily_results)
    summary_df.to_csv(summary_path, index=False)
    print(f"Processing summary saved: {summary_path}")


def main_iterative_tarantola_valette():
    """
    Main function using iterative Tarantola-Valette method
    """
    dday = 731 # Set number of days
    max_iter = 30  # Set number of iterations
    
    print("Starting iterative Tarantola-Valette inversion")
    print("=" * 60)
    print(f"Using parameters: corr_len = {corr_len} km, std_model = {std_model}")
    print(f"Reference length lambda0 = {lambda0} km")
    print(f"Maximum iterations: {max_iter}")
    print(f"Inversion formula: m = m0 + Cm G^T (G Cm G^T + Cd)^(-1) (d - G m0)")
    print("=" * 60)
    
    # Process all days using parallel processing
    print(f"Processing {dday} days of data...")
    
    # Use parallel processing for efficiency
    results = Parallel(n_jobs=-1)(
        delayed(process_day_iterative_tarantola_valette)(day0, max_iter) for day0 in range(dday)
    )
    
    # Calculate final sum
    print("Calculating final sum of all daily results...")
    final_sum, num_valid_days = calculate_final_sum(results)
    
    # Plot final sum
    if final_sum is not None:
        plot_final_sum(final_sum, num_valid_days)
    
    # Save results
    save_results(final_sum, results)
    
    # Print summary
    valid_results = [r for r in results if r[0] is not None]
    failed_results = [r for r in results if r[0] is None]
    
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total days processed: {min(10, dday)}")
    print(f"Successful inversions: {len(valid_results)}")
    print(f"Failed inversions: {len(failed_results)}")
    print(f"Success rate: {len(valid_results)/min(10, dday)*100:.2f}%")
    
    # Print methods used
    methods_used = [result[1] for result in valid_results]
    method_counts = {method: methods_used.count(method) for method in set(methods_used)}
    print("\nMethods used:")
    for method, count in method_counts.items():
        print(f"  {method}: {count} days")
    
    return results, final_sum



if __name__ == "__main__":
    # Run iterative Tarantola-Valette inversion
    results, final_sum = main_iterative_tarantola_valette()