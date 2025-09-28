#pragma once

#include <vector>
#include <optional>
#include "shapes.hh"

/* Adapted from https://github.com/KavrakiLab/vamp/blob/main/src/impl/vamp/collision/environment.hh */

namespace ppln::collision
{
    template <typename DataT>
    struct Environment
    {
        Sphere<DataT> *spheres;
        unsigned int num_spheres;

        Capsule<DataT> *capsules;
        unsigned int num_capsules;

        Capsule<DataT> *z_aligned_capsules;
        unsigned int num_z_aligned_capsules;

        Cylinder<DataT> *cylinders;
        unsigned int num_cylinders;

        Cuboid<DataT> *cuboids;
        unsigned int num_cuboids;

        Cuboid<DataT> *z_aligned_cuboids;
        unsigned int num_z_aligned_cuboids;

        // HeightField<DataT> *heightfields;
        // unsigned int num_heightfields;

        bool owns_memory; // Flag to indicate if this object owns the memory

        Environment() : owns_memory(true) {}

        // move constructor
        Environment(Environment &&other) noexcept
            : spheres(other.spheres),
              num_spheres(other.num_spheres),
              capsules(other.capsules),
              num_capsules(other.num_capsules),
              z_aligned_capsules(other.z_aligned_capsules),
              num_z_aligned_capsules(other.num_z_aligned_capsules),
              cylinders(other.cylinders),
              num_cylinders(other.num_cylinders),
              cuboids(other.cuboids),
              num_cuboids(other.num_cuboids),
              z_aligned_cuboids(other.z_aligned_cuboids),
              num_z_aligned_cuboids(other.num_z_aligned_cuboids),
              owns_memory(other.owns_memory)
        {
            other.spheres = nullptr;
            other.capsules = nullptr;
            other.z_aligned_capsules = nullptr;
            other.cylinders = nullptr;
            other.cuboids = nullptr;
            other.z_aligned_cuboids = nullptr;
            other.owns_memory = false; // Transfer ownership
        }


        ~Environment() {
            if (owns_memory) {
                delete[] spheres;
                delete[] capsules;
                delete[] cuboids;
                delete[] z_aligned_capsules;
                delete[] cylinders;
                delete[] z_aligned_cuboids;
            }
        }
    };
}  // namespace ppln::collision