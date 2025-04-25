/*
Multi environment batch collision checker.
*/

#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <numeric>

#include "src/collision/environment.hh"
#include "src/collision/factory.hh"
#include "src/Planners.hh"
#include "src/pRRTC_settings.hh"
#include "src/utils.cuh"
#include "batch_cc.hh"


#include <float.h>

namespace batch_cc {

    template<typename Robot>
    struct HaltonState {
        float b[Robot::dimension];   // bases
        float n[Robot::dimension];   // numerators
        float d[Robot::dimension];   // denominators
    };

    void __device__ shuffle_array(float *array, int n, curandState &state) {
        for (int i = n - 1; i > 0; i--) {
            int j = curand(&state) % (i + 1);
            float temp = array[i];
            array[i] = array[j];
            array[j] = temp;
        }
    }

    template <typename Robot>
    __global__ void
    __launch_bounds__(32, 4)
    batch_cc_kernel(ppln::collision::Environment<float>** envs, float* edges[2][Robot::dimension], int num_envs, int num_edges, bool *cc_result, int resolution)
    {
        constexpr auto dim = Robot::dimension;
        const int tid = threadIdx.x;
        const int bid = blockIdx.x;
        // each block handles one (edge, environment) pair
        const int env_idx = bid % num_envs;
        const int edge_idx = bid / num_envs;
        if (env_idx >= num_envs || edge_idx >= num_edges) {
            return;
        }
        // if (tid == 0) printf("bid: %d, env_idx: %d, edge_idx: %d\n", bid, env_idx, edge_idx);
        ppln::collision::Environment<float>* env = envs[env_idx];
        // if (tid == 0 && bid == 0) {
        //     if (env == nullptr) {
        //         printf("Environment is null!\n");
        //     } else {
        //         printf("Environment is valid.\n");
        //         printf("Number of spheres: %d\n", env->num_spheres);
        //         printf("Number of capsules: %d\n", env->num_capsules);
        //         printf("Number of cuboids: %d\n", env->num_cuboids);
        //     }
        //     for (int i = 0; i < num_edges; ++i) {
        //         printf("Edge %d: ", i);
        //         for (int j = 0; j < dim; ++j) {
        //             printf("%f ", edges[0][j][i]);
        //         }
        //         printf(" to ");
        //         for (int j = 0; j < dim; ++j) {
        //             printf("%f ", edges[1][j][i]);
        //         }
        //         printf("\n");
        //     }
        // }
        // __syncthreads();
        __shared__ float edge_start[dim];
        __shared__ float edge_end[dim];
        __shared__ float delta[dim];
        __shared__ unsigned int local_cc_result;
        __shared__ int n;
        float config[dim];
        if (tid < dim) {
            edge_start[tid] = edges[0][tid][edge_idx];
            edge_end[tid] = edges[1][tid][edge_idx];
        }
        __syncthreads();
        if (tid == 0) {
            // if (bid == 0) {
            //     printf("Edge start: ");
            //     for (int i = 0; i < dim; ++i) {
            //         printf("%f ", edge_start[i]);
            //     }
            //     printf("\nEdge end: ");
            //     for (int i = 0; i < dim; ++i) {
            //         printf("%f ", edge_end[i]);
            //     }
            //     printf("\n");
            // }
            float dist = sqrt(device_utils::sq_l2_dist(edge_start, edge_end, dim));
            n = max(ceil((dist / (float) blockDim.x) * resolution), 1.0f);
            local_cc_result = 0;
        }
        __syncthreads();
        if (tid < dim) {
            delta[tid] = (edge_end[tid] - edge_start[tid]) / (float) (blockDim.x * n);
        }
        __syncthreads();
        # pragma unroll
        for (int j = 0; j < dim; j++) {
            config[j] = edge_start[j] + delta[j] * (tid * n);
        }
        __syncthreads();
        for (int i = 0; i < n; i++) {
            bool config_in_collision = not ppln::collision::fkcc<Robot>(config, env, tid);
            // atomicOr(&local_cc_result, config_in_collision ? 1u : 0u);
            // __syncthreads();
            // if (local_cc_result) break;
            local_cc_result = __any_sync(0xffffffff, config_in_collision);
            if (local_cc_result) break;
            # pragma unroll
            for (int j = 0; j < dim; j++) {
                config[j] += delta[j];
            }
        }
        if (tid == 0) {
            cc_result[edge_idx * num_envs + env_idx] = local_cc_result ? true : false;
        }
    }

    inline void setup_environment_on_device(ppln::collision::Environment<float> *&d_env, 
                                        const ppln::collision::Environment<float> &h_env) {
        // First allocate the environment struct
        cudaMalloc(&d_env, sizeof(ppln::collision::Environment<float>));
        
        // Initialize struct to zeros first
        cudaMemset(d_env, 0, sizeof(ppln::collision::Environment<float>));

        // Handle each primitive type separately
        if (h_env.num_spheres > 0) {
            // Allocate and copy spheres array
            ppln::collision::Sphere<float> *d_spheres;
            cudaMalloc(&d_spheres, sizeof(ppln::collision::Sphere<float>) * h_env.num_spheres);
            cudaMemcpy(d_spheres, h_env.spheres, 
                    sizeof(ppln::collision::Sphere<float>) * h_env.num_spheres, 
                    cudaMemcpyHostToDevice);
            
            // Update the struct fields directly
            cudaMemcpy(&(d_env->spheres), &d_spheres, sizeof(ppln::collision::Sphere<float>*), 
                    cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_env->num_spheres), &h_env.num_spheres, sizeof(unsigned int), 
                    cudaMemcpyHostToDevice);
        }

        if (h_env.num_capsules > 0) {
            ppln::collision::Capsule<float> *d_capsules;
            cudaMalloc(&d_capsules, sizeof(ppln::collision::Capsule<float>) * h_env.num_capsules);
            cudaMemcpy(d_capsules, h_env.capsules,
                    sizeof(ppln::collision::Capsule<float>) * h_env.num_capsules,
                    cudaMemcpyHostToDevice);
            
            cudaMemcpy(&(d_env->capsules), &d_capsules, sizeof(ppln::collision::Capsule<float>*),
                    cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_env->num_capsules), &h_env.num_capsules, sizeof(unsigned int),
                    cudaMemcpyHostToDevice);
        }

        // Repeat for each primitive type...
        if (h_env.num_z_aligned_capsules > 0) {
            ppln::collision::Capsule<float> *d_z_capsules;
            cudaMalloc(&d_z_capsules, sizeof(ppln::collision::Capsule<float>) * h_env.num_z_aligned_capsules);
            cudaMemcpy(d_z_capsules, h_env.z_aligned_capsules,
                    sizeof(ppln::collision::Capsule<float>) * h_env.num_z_aligned_capsules,
                    cudaMemcpyHostToDevice);
            
            cudaMemcpy(&(d_env->z_aligned_capsules), &d_z_capsules, sizeof(ppln::collision::Capsule<float>*),
                    cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_env->num_z_aligned_capsules), &h_env.num_z_aligned_capsules, sizeof(unsigned int),
                    cudaMemcpyHostToDevice);
        }

        if (h_env.num_cylinders > 0) {
            ppln::collision::Cylinder<float> *d_cylinders;
            cudaMalloc(&d_cylinders, sizeof(ppln::collision::Cylinder<float>) * h_env.num_cylinders);
            cudaMemcpy(d_cylinders, h_env.cylinders,
                    sizeof(ppln::collision::Cylinder<float>) * h_env.num_cylinders,
                    cudaMemcpyHostToDevice);
            
            cudaMemcpy(&(d_env->cylinders), &d_cylinders, sizeof(ppln::collision::Cylinder<float>*),
                    cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_env->num_cylinders), &h_env.num_cylinders, sizeof(unsigned int),
                    cudaMemcpyHostToDevice);
        }

        if (h_env.num_cuboids > 0) {
            ppln::collision::Cuboid<float> *d_cuboids;
            cudaMalloc(&d_cuboids, sizeof(ppln::collision::Cuboid<float>) * h_env.num_cuboids);
            cudaMemcpy(d_cuboids, h_env.cuboids,
                    sizeof(ppln::collision::Cuboid<float>) * h_env.num_cuboids,
                    cudaMemcpyHostToDevice);
            
            cudaMemcpy(&(d_env->cuboids), &d_cuboids, sizeof(ppln::collision::Cuboid<float>*),
                    cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_env->num_cuboids), &h_env.num_cuboids, sizeof(unsigned int),
                    cudaMemcpyHostToDevice);
        }

        if (h_env.num_z_aligned_cuboids > 0) {
            ppln::collision::Cuboid<float> *d_z_cuboids;
            cudaMalloc(&d_z_cuboids, sizeof(ppln::collision::Cuboid<float>) * h_env.num_z_aligned_cuboids);
            cudaMemcpy(d_z_cuboids, h_env.z_aligned_cuboids,
                    sizeof(ppln::collision::Cuboid<float>) * h_env.num_z_aligned_cuboids,
                    cudaMemcpyHostToDevice);
            
            cudaMemcpy(&(d_env->z_aligned_cuboids), &d_z_cuboids, sizeof(ppln::collision::Cuboid<float>*),
                    cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_env->num_z_aligned_cuboids), &h_env.num_z_aligned_cuboids, sizeof(unsigned int),
                    cudaMemcpyHostToDevice);
        }
    }


    inline void cleanup_environment_on_device(ppln::collision::Environment<float> *d_env, 
                                        const ppln::collision::Environment<float> &h_env) {
        // Get the pointers from device struct before freeing
        ppln::collision::Sphere<float> *d_spheres = nullptr;
        ppln::collision::Capsule<float> *d_capsules = nullptr;
        ppln::collision::Capsule<float> *d_z_capsules = nullptr;
        ppln::collision::Cylinder<float> *d_cylinders = nullptr;
        ppln::collision::Cuboid<float> *d_cuboids = nullptr;
        ppln::collision::Cuboid<float> *d_z_cuboids = nullptr;

        // Copy each pointer from device memory
        if (h_env.num_spheres > 0) {
            cudaMemcpy(&d_spheres, &(d_env->spheres), sizeof(ppln::collision::Sphere<float>*), cudaMemcpyDeviceToHost);
            cudaFree(d_spheres);
        }
        
        if (h_env.num_capsules > 0) {
            cudaMemcpy(&d_capsules, &(d_env->capsules), sizeof(ppln::collision::Capsule<float>*), cudaMemcpyDeviceToHost);
            cudaFree(d_capsules);
        }
        
        if (h_env.num_z_aligned_capsules > 0) {
            cudaMemcpy(&d_z_capsules, &(d_env->z_aligned_capsules), sizeof(ppln::collision::Capsule<float>*), cudaMemcpyDeviceToHost);
            cudaFree(d_z_capsules);
        }
        
        if (h_env.num_cylinders > 0) {
            cudaMemcpy(&d_cylinders, &(d_env->cylinders), sizeof(ppln::collision::Cylinder<float>*), cudaMemcpyDeviceToHost);
            cudaFree(d_cylinders);
        }
        
        if (h_env.num_cuboids > 0) {
            cudaMemcpy(&d_cuboids, &(d_env->cuboids), sizeof(ppln::collision::Cuboid<float>*), cudaMemcpyDeviceToHost);
            cudaFree(d_cuboids);
        }
        
        if (h_env.num_z_aligned_cuboids > 0) {
            cudaMemcpy(&d_z_cuboids, &(d_env->z_aligned_cuboids), sizeof(ppln::collision::Cuboid<float>*), cudaMemcpyDeviceToHost);
            cudaFree(d_z_cuboids);
        }

        // Finally free the environment struct itself
        cudaFree(d_env);
    }



    template <typename Robot>
    void batch_cc(std::vector<ppln::collision::Environment<float>>& h_envs, std::vector<std::array<typename Robot::Configuration, 2>>& edges, int resolution, std::vector<bool>& results) {
        // verify pointers in h_envs are valid
        // for (const auto env : h_envs) {
        //     std::cout << "checking environment" << std::endl;
        //     std::cout << env->num_spheres << " spheres, "
        //               << env->num_capsules << " capsules, "
        //               << env->num_z_aligned_capsules << " z-aligned capsules, "
        //               << env->num_cylinders << " cylinders, "
        //               << env->num_cuboids << " cuboids, "
        //               << env->num_z_aligned_cuboids << " z-aligned cuboids."
        //               << env->spheres << " spheres pointer, "
        //             << env->capsules << " capsules pointer, "
        //             << env->z_aligned_capsules << " z-aligned capsules pointer, "
        //             << env->cylinders << " cylinders pointer, "
        //             << env->cuboids << " cuboids pointer, "
        //             << env->z_aligned_cuboids << " z-aligned cuboids pointer."
        //               << std::endl;
        // }
        // std::cout << "here0.5" << std::endl;
        // std::cout << "Number of environments: " << h_envs.size() << std::endl;
        std::vector<ppln::collision::Environment<float>*> d_envs(h_envs.size());
        // std::cout << "here0" << std::endl;
        for (size_t i = 0; i < h_envs.size(); ++i) {
            // std::cout << "Setting up environment " << i << std::endl;
            setup_environment_on_device(d_envs[i], h_envs[i]);
        }
        int num_envs = h_envs.size();
        int num_edges = edges.size();
        int num_blocks = num_envs * num_edges;
        int num_threads = 32;
        // std::cout << "here1" << std::endl;
        ppln::collision::Environment<float>** d_envs_ptr;
        cudaMalloc(&d_envs_ptr, sizeof(ppln::collision::Environment<float>*) * num_envs);
        cudaMemcpy(d_envs_ptr, d_envs.data(), sizeof(ppln::collision::Environment<float>*) * num_envs, cudaMemcpyHostToDevice);

        bool *d_cc_result;
        cudaMalloc(&d_cc_result, sizeof(bool) * num_envs * num_edges);
        cudaMemset(d_cc_result, 0, sizeof(bool) * num_envs * num_edges);
        // std::cout << "here2" << std::endl;
        float* d_edges[2][Robot::dimension];        
        for (int i = 0; i < Robot::dimension; ++i) {
            cudaMalloc(&d_edges[0][i], sizeof(float) * num_edges);
            cudaMalloc(&d_edges[1][i], sizeof(float) * num_edges);
            // Copy the edges to device memory
            for (int j = 0; j < num_edges; ++j) {
                float start = edges[j][0][i];
                float end = edges[j][1][i];
                cudaMemcpy(d_edges[0][i] + j, &start, sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(d_edges[1][i] + j, &end, sizeof(float), cudaMemcpyHostToDevice);
            }
        }

        // Allocate memory for the device array of pointers
        float* (*d_edges_ptr)[Robot::dimension];
        cudaMalloc(&d_edges_ptr, sizeof(float*) * 2 * Robot::dimension);

        // Copy the host array of pointers to the device
        cudaMemcpy(d_edges_ptr, d_edges, sizeof(float*) * 2 * Robot::dimension, cudaMemcpyHostToDevice);

        // std::cout << "here3" << std::endl;
        cudaCheckError(cudaGetLastError());
        // std::cout << "num_envs: " << num_envs << ", num_edges: " << num_edges << "resolution: " << resolution << std::endl;
        auto start_time = std::chrono::steady_clock::now();
        batch_cc_kernel<Robot><<<num_blocks, num_threads>>>(d_envs_ptr, d_edges_ptr, num_envs, num_edges, d_cc_result, resolution);
        cudaDeviceSynchronize();
        auto nanoseconds = get_elapsed_nanoseconds(start_time);

        std::cout << "Time taken: " << nanoseconds << " ns" << std::endl;
        int edges_checked = num_envs * num_edges;
        std::cout << "Edges checked: " << edges_checked << std::endl;
        double throughput = edges_checked / (nanoseconds / 1e9);
        std::cout << "Throughput: " << throughput << " edges/s" << std::endl;

        // Create a temporary buffer for the results
        bool* h_cc_result = new bool[num_envs * num_edges];
        cudaMemcpy(h_cc_result, d_cc_result, sizeof(bool) * num_envs * num_edges, cudaMemcpyDeviceToHost);
        
        // Copy from temporary buffer to vector<bool>
        for (int i = 0; i < num_envs * num_edges; ++i) {
            results[i] = h_cc_result[i];
        }
        delete[] h_cc_result;

        cudaCheckError(cudaGetLastError());
        // std::cout << "here4" << std::endl;
        for (int i = 0; i < h_envs.size(); ++i) {
            cleanup_environment_on_device(d_envs[i], h_envs[i]);
        }
        cudaFree(d_cc_result);
        for (int i = 0; i < Robot::dimension; ++i) {
            cudaFree(d_edges[0][i]);
            cudaFree(d_edges[1][i]);
        }
        cudaFree(d_envs_ptr);
    }

    template void batch_cc<typename ppln::robots::Panda>(std::vector<ppln::collision::Environment<float>>& h_envs, std::vector<std::array<typename ppln::robots::Panda::Configuration, 2>>& edges, int resolution, std::vector<bool>& results);
    template void batch_cc<typename ppln::robots::Fetch>(std::vector<ppln::collision::Environment<float>>& h_envs, std::vector<std::array<typename ppln::robots::Fetch::Configuration, 2>>& edges, int resolution, std::vector<bool>& results);
    template void batch_cc<typename ppln::robots::Baxter>(std::vector<ppln::collision::Environment<float>>& h_envs, std::vector<std::array<typename ppln::robots::Baxter::Configuration, 2>>& edges, int resolution, std::vector<bool>& results);
} // namespace batch_cc