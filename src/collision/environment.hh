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

        Environment() = default;

        // copy constructor (deep copy)
        // Environment(const Environment &other)
        //     : num_spheres(other.num_spheres),
        //       num_capsules(other.num_capsules),
        //       num_z_aligned_capsules(other.num_z_aligned_capsules),
        //       num_cylinders(other.num_cylinders),
        //       num_cuboids(other.num_cuboids),
        //       num_z_aligned_cuboids(other.num_z_aligned_cuboids)
        // {
        //     spheres = new Sphere<DataT>[num_spheres];
        //     for (unsigned int i = 0; i < num_spheres; ++i)
        //     spheres[i] = other.spheres[i];

        //     capsules = new Capsule<DataT>[num_capsules];
        //     for (unsigned int i = 0; i < num_capsules; ++i)
        //     capsules[i] = other.capsules[i];

        //     z_aligned_capsules = new Capsule<DataT>[num_z_aligned_capsules];
        //     for (unsigned int i = 0; i < num_z_aligned_capsules; ++i)
        //     z_aligned_capsules[i] = other.z_aligned_capsules[i];

        //     cylinders = new Cylinder<DataT>[num_cylinders];
        //     for (unsigned int i = 0; i < num_cylinders; ++i)
        //     cylinders[i] = other.cylinders[i];

        //     cuboids = new Cuboid<DataT>[num_cuboids];
        //     for (unsigned int i = 0; i < num_cuboids; ++i)
        //     cuboids[i] = other.cuboids[i];

        //     z_aligned_cuboids = new Cuboid<DataT>[num_z_aligned_cuboids];
        //     for (unsigned int i = 0; i < num_z_aligned_cuboids; ++i)
        //     z_aligned_cuboids[i] = other.z_aligned_cuboids[i];
        // }

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
              num_z_aligned_cuboids(other.num_z_aligned_cuboids)
        {
            other.spheres = nullptr;
            other.capsules = nullptr;
            other.z_aligned_capsules = nullptr;
            other.cylinders = nullptr;
            other.cuboids = nullptr;
            other.z_aligned_cuboids = nullptr;
        }


        ~Environment() {
            std::cout << "Destroying environment" << std::endl;
            delete[] spheres;
            delete[] capsules;
            delete[] cuboids;
            delete[] z_aligned_capsules;
            delete[] cylinders;
            delete[] z_aligned_cuboids;
        }
    };
}  // namespace ppln::collision