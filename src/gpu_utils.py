import torch
import gc

def clean_gpu_memory():
    """
    Clean up GPU memory and move it to idle state.
    Can be run as a standalone script or imported as a function.
    """
    if torch.cuda.is_available():
        # Empty the cache
        torch.cuda.empty_cache()
        
        # Clear any tensors still in memory
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj):
                    if obj.is_cuda:
                        del obj
                elif hasattr(obj, 'data') and torch.is_tensor(obj.data):
                    if obj.data.is_cuda:
                        del obj
            except Exception:
                pass
                
        # Force garbage collection
        gc.collect()
        
        # Reset the GPU state
        torch.cuda.reset_peak_memory_stats()
        
        # Print memory stats
        print("\nGPU Memory Status:")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Cached:    {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    else:
        print("No GPU available.")

if __name__ == "__main__":
    clean_gpu_memory()