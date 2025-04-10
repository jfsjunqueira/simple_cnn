import torch
import gc
import platform
import os

def get_gpu_memory_usage():
    """
    Get current GPU memory usage.
    
    Returns:
        dict: Dictionary with memory usage stats
    """
    if not torch.cuda.is_available():
        return {"gpu_available": False}
    
    gpu_stats = {}
    gpu_stats["gpu_available"] = True
    gpu_stats["device_name"] = torch.cuda.get_device_name(0)
    
    # Memory allocated
    memory_allocated = torch.cuda.memory_allocated(0)
    memory_allocated_gb = memory_allocated / 1024**3
    gpu_stats["memory_allocated"] = f"{memory_allocated_gb:.2f} GB"
    
    # Memory reserved by cache
    memory_reserved = torch.cuda.memory_reserved(0)
    memory_reserved_gb = memory_reserved / 1024**3
    gpu_stats["memory_reserved"] = f"{memory_reserved_gb:.2f} GB"
    
    # Total memory
    total_memory = torch.cuda.get_device_properties(0).total_memory
    total_memory_gb = total_memory / 1024**3
    gpu_stats["total_memory"] = f"{total_memory_gb:.2f} GB"
    
    # Usage percentage
    percentage_used = (memory_allocated / total_memory) * 100
    gpu_stats["percentage_used"] = f"{percentage_used:.2f}%"
    
    return gpu_stats

def print_gpu_memory_usage():
    """Print GPU memory usage in a formatted way"""
    gpu_stats = get_gpu_memory_usage()
    
    if not gpu_stats["gpu_available"]:
        print("No GPU available")
        return
    
    print("\nGPU Memory Usage:")
    print(f"Device:          {gpu_stats['device_name']}")
    print(f"Allocated:       {gpu_stats['memory_allocated']}")
    print(f"Reserved:        {gpu_stats['memory_reserved']}")
    print(f"Total:           {gpu_stats['total_memory']}")
    print(f"Utilization:     {gpu_stats['percentage_used']}")

def clean_gpu_memory():
    """
    Clean up GPU memory and move it to idle state.
    Can be run as a standalone script or imported as a function.
    """
    # Empty the PyTorch cache
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Clear MPS cache for Apple Silicon
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Currently there's no direct way to clear MPS cache like cuda.empty_cache()
        # But we can try to force garbage collection
        pass
        
    # Clear any tensors still in memory
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                if obj.is_cuda:
                    del obj
                elif hasattr(obj, 'is_mps') and obj.is_mps:
                    del obj
            elif hasattr(obj, 'data') and torch.is_tensor(obj.data):
                if obj.data.is_cuda:
                    del obj
                elif hasattr(obj.data, 'is_mps') and obj.data.is_mps:
                    del obj
        except Exception:
            pass
                
    # Force garbage collection
    gc.collect()
    
    # Print memory stats
    print("\nAccelerator Memory Status:")
    
    if torch.cuda.is_available():
        # Reset the GPU stats
        torch.cuda.reset_peak_memory_stats()
        
        # Print CUDA memory stats
        print(f"GPU Device:  CUDA")
        print_gpu_memory_usage()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Print Apple Silicon MPS info
        print(f"GPU Device:  Apple MPS ({platform.processor()})")
        print(f"Note: Detailed MPS memory statistics not available")
    else:
        print("No GPU acceleration available.")

if __name__ == "__main__":
    print_gpu_memory_usage()
    clean_gpu_memory()