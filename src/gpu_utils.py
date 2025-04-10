import torch
import gc
import platform

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
        print(f"Allocated:   {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Cached:      {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Print Apple Silicon MPS info
        print(f"GPU Device:  Apple MPS ({platform.processor()})")
        print(f"Note: Detailed MPS memory statistics not available")
    else:
        print("No GPU acceleration available.")

if __name__ == "__main__":
    clean_gpu_memory()