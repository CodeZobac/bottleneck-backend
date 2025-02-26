import pandas as pd
import logging

logger = logging.getLogger(__name__)

def get_unique_components(cpu_path, gpu_path):
    """
    Extract unique brand+model combinations from CPU and GPU datasets
    
    Parameters:
    -----------
    cpu_path : str
        Path to the CPU CSV file
    gpu_path : str
        Path to the GPU CSV file
        
    Returns:
    --------
    dict
        Dictionary containing lists of unique CPU and GPU components
    """
    try:
        # Load CSV files
        cpu_df = pd.read_csv(cpu_path)
        gpu_df = pd.read_csv(gpu_path)
        
        # Extract unique CPU components
        cpu_components = set()
        for _, row in cpu_df.iterrows():
            component_name = f"{row['Brand']} {row['Model']}"
            cpu_components.add(component_name.strip())
            
        # Extract unique GPU components
        gpu_components = set()
        for _, row in gpu_df.iterrows():
            component_name = f"{row['Brand']} {row['Model']}"
            gpu_components.add(component_name.strip())
        
        # Convert to sorted lists
        unique_cpus = sorted(list(cpu_components))
        unique_gpus = sorted(list(gpu_components))
        
        return {
            "cpus": unique_cpus,
            "gpus": unique_gpus
        }
    except Exception as e:
        logger.error(f"Error getting component data: {str(e)}", exc_info=True)
        raise e

# For testing purposes
if __name__ == "__main__":
    test_cpu_path = "./data/CPU_UserBenchmarks.csv"
    test_gpu_path = "./data/GPU_UserBenchmarks.csv"
    
    components = get_unique_components(test_cpu_path, test_gpu_path)
    print(f"Found {len(components['cpus'])} unique CPUs")
    print(f"Found {len(components['gpus'])} unique GPUs")
    
    # Print first 5 of each for verification
    print("\nSample CPUs:")
    for cpu in components['cpus'][:5]:
        print(f"  - {cpu}")
        
    print("\nSample GPUs:")
    for gpu in components['gpus'][:5]:
        print(f"  - {gpu}")
