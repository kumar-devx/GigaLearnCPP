// GPU Memory Leak Test Utility
// This file demonstrates how to monitor GPU memory usage during GigaLearnCPP operations

#include <GigaLearnCPP/Util/InferUnit.h>
#include <iostream>
#include <thread>
#include <chrono>

#ifdef RG_CUDA_SUPPORT
#include <torch/cuda.h>
#include <c10/cuda/CUDACachingAllocator.h>
#endif

void PrintGPUMemoryStats() {
#ifdef RG_CUDA_SUPPORT
    if (torch::cuda::is_available()) {
        size_t free_mem = 0;
        size_t total_mem = 0;
        
        // Get CUDA memory info
        cudaMemGetInfo(&free_mem, &total_mem);
        
        size_t used_mem = total_mem - free_mem;
        
        std::cout << "GPU Memory Status:" << std::endl;
        std::cout << "  Total: " << (total_mem / (1024 * 1024)) << " MB" << std::endl;
        std::cout << "  Used:  " << (used_mem / (1024 * 1024)) << " MB" << std::endl;
        std::cout << "  Free:  " << (free_mem / (1024 * 1024)) << " MB" << std::endl;
        
        // PyTorch's internal memory stats
        auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
        std::cout << "  PyTorch Allocated: " << (stats.allocated_bytes[0].current / (1024 * 1024)) << " MB" << std::endl;
        std::cout << "  PyTorch Reserved:  " << (stats.reserved_bytes[0].current / (1024 * 1024)) << " MB" << std::endl;
    } else {
        std::cout << "CUDA not available" << std::endl;
    }
#else
    std::cout << "CUDA support not compiled in" << std::endl;
#endif
}

void TestInferUnitMemoryLeak() {
    std::cout << "\n=== Testing InferUnit Memory Management ===" << std::endl;
    
    std::cout << "\nInitial GPU Memory:" << std::endl;
    PrintGPUMemoryStats();
    
    // Create and destroy InferUnit multiple times to test for leaks
    for (int iteration = 0; iteration < 5; iteration++) {
        std::cout << "\n--- Iteration " << (iteration + 1) << " ---" << std::endl;
        
        {
            // This scope will test if InferUnit properly frees memory
            // Note: You'll need to provide actual obs builder, action parser, and model config
            // This is just a template - adjust to your actual setup
            
            std::cout << "Creating InferUnit..." << std::endl;
            // GGL::InferUnit* inferUnit = new GGL::InferUnit(...);
            
            std::cout << "GPU Memory after creation:" << std::endl;
            PrintGPUMemoryStats();
            
            // Simulate inference
            std::cout << "Running inference..." << std::endl;
            // inferUnit->InferAction(...);
            
            std::cout << "GPU Memory after inference:" << std::endl;
            PrintGPUMemoryStats();
            
            // Delete InferUnit - this should free all GPU memory
            std::cout << "Deleting InferUnit..." << std::endl;
            // delete inferUnit;
            
            // Force PyTorch to release cached memory
#ifdef RG_CUDA_SUPPORT
            if (torch::cuda::is_available()) {
                torch::cuda::synchronize();
                c10::cuda::CUDACachingAllocator::emptyCache();
            }
#endif
        }
        
        std::cout << "GPU Memory after deletion:" << std::endl;
        PrintGPUMemoryStats();
        
        // Brief pause between iterations
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    
    std::cout << "\n=== Test Complete ===" << std::endl;
    std::cout << "If memory usage returns to near-initial levels, the leak is fixed!" << std::endl;
}

int main() {
    std::cout << "GPU Memory Leak Test Utility" << std::endl;
    std::cout << "============================" << std::endl;
    
    TestInferUnitMemoryLeak();
    
    return 0;
}
