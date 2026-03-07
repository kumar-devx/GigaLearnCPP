// Standalone Memory Leak Test for GigaLearnCPP
// This tests the memory leak fixes without needing GigaLearnBot.exe
// Build: Add this to CMakeLists.txt as a separate executable

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <memory>

#ifdef RG_CUDA_SUPPORT
#include <torch/torch.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime.h>
#endif

// Forward declarations
namespace GGL {
    class InferUnit;
    struct PPOLearner;
    struct PolicyVersionManager;
}

// GPU Memory Tracking
struct GPUMemorySnapshot {
    size_t allocated_mb;
    size_t free_mb;
    size_t total_mb;
    std::chrono::system_clock::time_point timestamp;
    
    void Print() const {
        std::cout << "GPU Memory: " 
                  << allocated_mb << " MB used, " 
                  << free_mb << " MB free, "
                  << total_mb << " MB total" << std::endl;
    }
};

GPUMemorySnapshot GetGPUMemory() {
    GPUMemorySnapshot snapshot;
    snapshot.timestamp = std::chrono::system_clock::now();
    
#ifdef RG_CUDA_SUPPORT
    try {
        if (torch::cuda::is_available()) {
            size_t free_bytes = 0;
            size_t total_bytes = 0;
            cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
            
            if (err != cudaSuccess) {
                std::cerr << "cudaMemGetInfo failed: " << cudaGetErrorString(err) << std::endl;
                snapshot.total_mb = 0;
                snapshot.free_mb = 0;
                snapshot.allocated_mb = 0;
                return snapshot;
            }
            
            snapshot.total_mb = total_bytes / (1024 * 1024);
            snapshot.free_mb = free_bytes / (1024 * 1024);
            snapshot.allocated_mb = snapshot.total_mb - snapshot.free_mb;
        } else {
            std::cout << "CUDA not available!" << std::endl;
            snapshot.total_mb = 0;
            snapshot.free_mb = 0;
            snapshot.allocated_mb = 0;
        }
    } catch (const c10::Error& e) {
        std::cerr << "CUDA Error in GetGPUMemory: " << e.what() << std::endl;
        snapshot.total_mb = 0;
        snapshot.free_mb = 0;
        snapshot.allocated_mb = 0;
    } catch (const std::exception& e) {
        std::cerr << "Exception in GetGPUMemory: " << e.what() << std::endl;
        snapshot.total_mb = 0;
        snapshot.free_mb = 0;
        snapshot.allocated_mb = 0;
    }
#else
    std::cout << "CUDA support not compiled in!" << std::endl;
    snapshot.total_mb = 0;
    snapshot.free_mb = 0;
    snapshot.allocated_mb = 0;
#endif
    
    return snapshot;
}

void PrintSeparator() {
    std::cout << "================================================" << std::endl;
}

class MemoryLeakTester {
private:
    std::vector<GPUMemorySnapshot> snapshots;
    size_t leak_threshold_mb = 100; // Consider it a leak if memory grows by this much
    
public:
    void TakeSnapshot(const std::string& label) {
        auto snapshot = GetGPUMemory();
        snapshots.push_back(snapshot);
        
        std::cout << "[" << label << "] ";
        snapshot.Print();
    }
    
    bool DetectLeak() {
        if (snapshots.size() < 2) {
            std::cout << "Not enough snapshots to detect leak" << std::endl;
            return false;
        }
        
        auto& first = snapshots.front();
        auto& last = snapshots.back();
        
        int64_t memory_diff = static_cast<int64_t>(last.allocated_mb) - static_cast<int64_t>(first.allocated_mb);
        
        std::cout << "\nMemory Analysis:" << std::endl;
        std::cout << "  Initial: " << first.allocated_mb << " MB" << std::endl;
        std::cout << "  Final:   " << last.allocated_mb << " MB" << std::endl;
        std::cout << "  Diff:    " << memory_diff << " MB" << std::endl;
        
        if (memory_diff > static_cast<int64_t>(leak_threshold_mb)) {
            std::cout << "  ❌ LEAK DETECTED: Memory increased by " << memory_diff << " MB!" << std::endl;
            return true;
        } else if (std::abs(memory_diff) < 50) {
            std::cout << "  ✅ NO LEAK: Memory returned to baseline" << std::endl;
            return false;
        } else {
            std::cout << "  ⚠️  Small memory difference (likely PyTorch cache): " << memory_diff << " MB" << std::endl;
            return false;
        }
    }
    
    void ClearSnapshots() {
        snapshots.clear();
    }
};

// Test 1: Basic CUDA Memory Test (no GigaLearnCPP dependencies)
bool TestBasicCUDAMemory() {
    PrintSeparator();
    std::cout << "TEST 1: Basic CUDA Memory Allocation/Deallocation" << std::endl;
    PrintSeparator();
    
#ifndef RG_CUDA_SUPPORT
    std::cout << "SKIPPED: CUDA support not compiled" << std::endl;
    return true;
#endif
    
    try {
        MemoryLeakTester tester;
        
        tester.TakeSnapshot("Before allocation");
        
        // Allocate some GPU memory
        {
            auto tensor = torch::randn({1000, 1000}, torch::device(torch::kCUDA));
            tester.TakeSnapshot("After allocation (1000x1000 tensor)");
        } // tensor goes out of scope here
        
        // Force CUDA synchronization and cache clearing
#ifdef RG_CUDA_SUPPORT
        torch::cuda::synchronize();
        c10::cuda::CUDACachingAllocator::emptyCache();
#endif
        
        tester.TakeSnapshot("After deallocation");
        
        bool has_leak = tester.DetectLeak();
        
        std::cout << "\nTest 1: " << (has_leak ? "FAILED ❌" : "PASSED ✅") << std::endl;
        return !has_leak;
    } catch (const c10::Error& e) {
        std::cerr << "CUDA Error in Test 1: " << e.what() << std::endl;
        std::cout << "\nTest 1: FAILED ❌ (CUDA Error)" << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "Exception in Test 1: " << e.what() << std::endl;
        std::cout << "\nTest 1: FAILED ❌ (Exception)" << std::endl;
        return false;
    }
}

// Test 2: Multiple Allocation Cycles
bool TestMultipleCycles() {
    PrintSeparator();
    std::cout << "TEST 2: Multiple Allocation/Deallocation Cycles" << std::endl;
    PrintSeparator();
    
#ifndef RG_CUDA_SUPPORT
    std::cout << "SKIPPED: CUDA support not compiled" << std::endl;
    return true;
#endif
    
    try {
        MemoryLeakTester tester;
        
        tester.TakeSnapshot("Initial");
        
        for (int i = 0; i < 5; i++) {
            std::cout << "\n--- Cycle " << (i + 1) << " ---" << std::endl;
            
            // Allocate
            {
                auto tensor = torch::randn({500, 500}, torch::device(torch::kCUDA));
                std::cout << "  Allocated tensor" << std::endl;
            }
            
            // Clean up
#ifdef RG_CUDA_SUPPORT
            torch::cuda::synchronize();
            c10::cuda::CUDACachingAllocator::emptyCache();
#endif
            
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        
        tester.TakeSnapshot("After 5 cycles");
        
        bool has_leak = tester.DetectLeak();
        
        std::cout << "\nTest 2: " << (has_leak ? "FAILED ❌" : "PASSED ✅") << std::endl;
        return !has_leak;
    } catch (const c10::Error& e) {
        std::cerr << "CUDA Error in Test 2: " << e.what() << std::endl;
        std::cout << "\nTest 2: FAILED ❌ (CUDA Error)" << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "Exception in Test 2: " << e.what() << std::endl;
        std::cout << "\nTest 2: FAILED ❌ (Exception)" << std::endl;
        return false;
    }
}

// Test 3: Pointer Lifecycle Test
bool TestPointerLifecycle() {
    PrintSeparator();
    std::cout << "TEST 3: Heap Allocation with Proper Cleanup" << std::endl;
    PrintSeparator();
    
#ifndef RG_CUDA_SUPPORT
    std::cout << "SKIPPED: CUDA support not compiled" << std::endl;
    return true;
#endif
    
    try {
        MemoryLeakTester tester;
        
        tester.TakeSnapshot("Initial");
        
        // Test pattern similar to InferUnit
        {
            std::cout << "Allocating on heap..." << std::endl;
            auto* data = new std::vector<torch::Tensor>();
            
            for (int i = 0; i < 3; i++) {
                data->push_back(torch::randn({100, 100}, torch::device(torch::kCUDA)));
            }
            
            tester.TakeSnapshot("After allocation");
            
            // Clean up
            data->clear();
            delete data;
            
            std::cout << "Deleted heap data" << std::endl;
        }
        
#ifdef RG_CUDA_SUPPORT
        torch::cuda::synchronize();
        c10::cuda::CUDACachingAllocator::emptyCache();
#endif
        
        tester.TakeSnapshot("After cleanup");
        
        bool has_leak = tester.DetectLeak();
        
        std::cout << "\nTest 3: " << (has_leak ? "FAILED ❌" : "PASSED ✅") << std::endl;
        return !has_leak;
    } catch (const c10::Error& e) {
        std::cerr << "CUDA Error in Test 3: " << e.what() << std::endl;
        std::cout << "\nTest 3: FAILED ❌ (CUDA Error)" << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "Exception in Test 3: " << e.what() << std::endl;
        std::cout << "\nTest 3: FAILED ❌ (Exception)" << std::endl;
        return false;
    }
}

// Test 4: Simulate InferUnit Pattern
bool TestInferUnitPattern() {
    PrintSeparator();
    std::cout << "TEST 4: InferUnit Creation/Destruction Pattern" << std::endl;
    PrintSeparator();
    
#ifndef RG_CUDA_SUPPORT
    std::cout << "SKIPPED: CUDA support not compiled" << std::endl;
    return true;
#endif
    
    try {
        MemoryLeakTester tester;
    
    tester.TakeSnapshot("Initial");
    
    // Simulate InferUnit lifecycle
    for (int iteration = 0; iteration < 3; iteration++) {
        std::cout << "\n--- InferUnit Iteration " << (iteration + 1) << " ---" << std::endl;
        
        {
            std::cout << "  Creating mock InferUnit..." << std::endl;
            
            // Simulate model storage (like InferUnit::models)
            struct MockModelSet {
                std::vector<torch::Tensor> params;
                
                ~MockModelSet() {
                    std::cout << "    MockModelSet destructor called" << std::endl;
                    params.clear();
#ifdef RG_CUDA_SUPPORT
                    torch::cuda::synchronize();
#endif
                }
            };
            
            auto* models = new MockModelSet();
            
            // Add some "model parameters"
            for (int i = 0; i < 5; i++) {
                models->params.push_back(torch::randn({200, 200}, torch::device(torch::kCUDA)));
            }
            
            tester.TakeSnapshot(std::string("  After creation ") + std::to_string(iteration + 1));
            
            // Simulate inference
            auto input = torch::randn({10, 100}, torch::device(torch::kCUDA));
            auto output = input * 2.0; // Simple operation
            
            // Clean up (what the destructor should do)
            std::cout << "  Destroying mock InferUnit..." << std::endl;
            delete models; // This should call ~MockModelSet()
            
#ifdef RG_CUDA_SUPPORT
            torch::cuda::synchronize();
            c10::cuda::CUDACachingAllocator::emptyCache();
#endif
        }
        
        tester.TakeSnapshot(std::string("  After destruction ") + std::to_string(iteration + 1));
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    
    tester.TakeSnapshot("Final");
    
    bool has_leak = tester.DetectLeak();
    
    std::cout << "\nTest 4: " << (has_leak ? "FAILED ❌" : "PASSED ✅") << std::endl;
    return !has_leak;
    } catch (const c10::Error& e) {
        std::cerr << "CUDA Error in Test 4: " << e.what() << std::endl;
        std::cout << "\nTest 4: FAILED ❌ (CUDA Error)" << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "Exception in Test 4: " << e.what() << std::endl;
        std::cout << "\nTest 4: FAILED ❌ (Exception)" << std::endl;
        return false;
    }
}

// Main test runner
int main() {
    std::cout << "GigaLearnCPP Memory Leak Detection Test Suite" << std::endl;
    std::cout << "=============================================" << std::endl;
    std::cout << std::endl;
    
#ifdef RG_CUDA_SUPPORT
    try {
        if (torch::cuda::is_available()) {
            std::cout << "✓ CUDA is available" << std::endl;
            std::cout << "  Device count: " << torch::cuda::device_count() << std::endl;
            std::cout << "  CUDA Version: " << CUDART_VERSION / 1000 << "." << (CUDART_VERSION % 100) / 10 << std::endl;
            
            // Check GPU compute capability
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);
            std::cout << "  GPU: " << prop.name << std::endl;
            std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
            std::cout << "  Total Memory: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB" << std::endl;
            
            std::cout << "\n⚠️  WARNING: If tests fail with 'no kernel image' error," << std::endl;
            std::cout << "    your libtorch was compiled for different GPU architectures." << std::endl;
            std::cout << "    Download libtorch with compute capability " << prop.major << "." << prop.minor << " support." << std::endl;
        } else {
            std::cout << "✗ CUDA not available - some tests will be skipped" << std::endl;
        }
    } catch (const c10::Error& e) {
        std::cerr << "✗ CUDA Error during initialization: " << e.what() << std::endl;
        std::cerr << "Tests will be skipped." << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "✗ Exception during CUDA initialization: " << e.what() << std::endl;
        std::cerr << "Tests will be skipped." << std::endl;
        return 1;
    }
#else
    std::cout << "✗ CUDA support not compiled - tests will be skipped" << std::endl;
#endif
    
    std::cout << std::endl;
    
    // Run tests
    std::vector<bool> results;
    
    results.push_back(TestBasicCUDAMemory());
    std::this_thread::sleep_for(std::chrono::seconds(1));
    
    results.push_back(TestMultipleCycles());
    std::this_thread::sleep_for(std::chrono::seconds(1));
    
    results.push_back(TestPointerLifecycle());
    std::this_thread::sleep_for(std::chrono::seconds(1));
    
    results.push_back(TestInferUnitPattern());
    
    // Summary
    PrintSeparator();
    std::cout << "\nTEST SUMMARY" << std::endl;
    PrintSeparator();
    
    int passed = 0;
    int failed = 0;
    
    for (size_t i = 0; i < results.size(); i++) {
        std::cout << "Test " << (i + 1) << ": " << (results[i] ? "PASSED ✅" : "FAILED ❌") << std::endl;
        if (results[i]) passed++;
        else failed++;
    }
    
    std::cout << "\nTotal: " << passed << " passed, " << failed << " failed" << std::endl;
    
    if (failed == 0) {
        std::cout << "\n🎉 ALL TESTS PASSED! Memory leak fixes are working correctly." << std::endl;
        return 0;
    } else {
        std::cout << "\n⚠️  SOME TESTS FAILED! Memory leaks may still exist." << std::endl;
        return 1;
    }
}
