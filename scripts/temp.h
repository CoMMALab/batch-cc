#include <vector>
#include <array>
#include <utility>
#include <iostream>
#include <cmath>
#include <algorithm>

#include <vamp/collision/factory.hh>
#include <vamp/planning/validate.hh>
#include <vamp/robots/panda.hh>
#include <ompl/base/MotionValidator.h>
#include <ompl/base/ProblemDefinition.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/StateValidityChecker.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/base/objectives/PathLengthOptimizationObjective.h>
#include <ompl/geometric/PathSimplifier.h>
#include <ompl/geometric/planners/informedtrees/BITstar.h>
#include <ompl/util/Exception.h>

#include "MyPRMOptVamp.h"

namespace ob = ompl::base;
namespace og = ompl::geometric;

using Robot = vamp::robots::Panda;
static constexpr std::size_t dimension = Robot::dimension;
using Configuration = Robot::Configuration;
static constexpr const std::size_t rake = vamp::FloatVectorWidth;
using EnvironmentInput = vamp::collision::Environment<float>;
using EnvironmentVector = vamp::collision::Environment<vamp::FloatVector<rake>>;

#define BOX (1)
#define SPHERE (2)
#define CYLINDER (3)
#define CONE (4)
#define PRISM (5)

struct CuboidData {
    const std::array<float, 3> center;
    const std::array<float, 3> euler_angles;
    const std::array<float, 3> half_dimens;

    CuboidData(float cx, float cy, float cz, float r, float p, float y, float dimX, float dimY, float dimZ)
        : center{cx, cy, cz},
          euler_angles{r, p, y},
          half_dimens{dimX / 2, dimY / 2, dimZ / 2} {}
};

struct SphereData {
    std::array<float, 3> center;
    float radius;

    SphereData(float cx, float cy, float cz, float r)
        : center{cx, cy, cz},
          radius(r) {}
};

struct CylinderData {
    std::array<float, 3> center;
    std::array<float, 3> euler_angles;
    float radius;
    float length;

    CylinderData(float cx, float cy, float cz, float r, float p, float y, float rad, float len)
        : center{cx, cy, cz},
          euler_angles{r, p, y},
          radius(rad), length(len) {}
};

const std::vector<std::array<float, 3>> problem_ = {
    {0.55, 0, 0.25},
    {0.35, 0.35, 0.25},
    {0, 0.55, 0.25},
    {-0.55, 0, 0.25},
    {-0.35, -0.35, 0.25},
    {0, -0.55, 0.25},
    {0.35, -0.35, 0.25},
    {0.35, 0.35, 0.8},
    {0, 0.55, 0.8},
    {-0.35, 0.35, 0.8},
    {-0.55, 0, 0.8},
    {-0.35, -0.35, 0.8},
    {0, -0.55, 0.8},
    {0.35, -0.35, 0.8},
    };

static std::array<EnvironmentInput, MAX_WORLD_SAMPLES_EVAL> environments;
static std::array<EnvironmentVector, MAX_WORLD_SAMPLES_EVAL> env_vectors;
static EnvironmentInput env_static;
static EnvironmentVector env_static_vec;

ob::StateValidityCheckerPtr getVampStateValidityCheck();
Configuration ompl_to_vamp(const ob::State *state);
//inline void setup_vamp_environments(std::string filename);

inline std::array<float, 3> quaternion_to_euler(float x, float y, float z, float w) {
    std::array<float, 3> angles; // angles[0]: roll, angles[1]: pitch, angles[2]: yaw

    // Roll (x-axis rotation)
    float t0 = 2.0f * (w * x + y * z);
    float t1 = 1.0f - 2.0f * (x * x + y * y);
    angles[0] = std::atan2(t0, t1);

    // Pitch (y-axis rotation)
    float t2 = 2.0f * (w * y - z * x);
    // Clamp t2 to be in the range [-1, 1] to avoid errors due to floating-point inaccuracies.
    t2 = std::max(-1.0f, std::min(1.0f, t2));
    angles[1] = std::asin(t2);

    // Yaw (z-axis rotation)
    float t3 = 2.0f * (w * z + x * y);
    float t4 = 1.0f - 2.0f * (y * y + z * z);
    angles[2] = std::atan2(t3, t4);

    return angles;
}

inline void setup_vamp_environments(std::string filename) {

    /* the object are static objects */
    std::vector<CuboidData> object_cuboids;
    std::vector<SphereData> object_spheres;
    std::vector<CylinderData> object_cylinders;

    /* Define maps for obstacles, as we may have more than one obstacle */
    std::map<std::string, std::vector<CuboidData>> obstacle_cuboids_map;
    std::map<std::string, std::vector<SphereData>> obstacle_spheres_map;
    std::map<std::string, std::vector<CylinderData>> obstacle_cylinders_map;

    std::vector<CuboidData> target_cuboids;
    std::vector<SphereData> target_spheres;
    std::vector<CylinderData> target_cylinders;

    std::ifstream env_file(filename);
    if (!env_file) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    std::string line;

    while (std::getline(env_file, line)) {
        std::istringstream iss(line);
        std::string category, obj_type;

        iss >> category >> obj_type;

        std::string obs_name;
        if (category.find("OBSTACLE") != std::string::npos) {
            obs_name = category;
            if (!obs_name.empty() && obs_name.back() == '_') {
                obs_name.pop_back();
            }
        }

        if(obj_type == "BOX") {
            float x, y, z, qx, qy, qz, qw, dim_x, dim_y, dim_z;
            iss >> x >> y >> z >> qx >> qy >> qz >> qw >> dim_x >> dim_y >> dim_z;
            std::array<float, 3> angles = quaternion_to_euler(qx, qy, qz, qw);
            const CuboidData cuboid(x, y, z, angles[0], angles[1], angles[2], dim_x, dim_y, dim_z);

            if (category.find("OBJECT") != std::string::npos) {
                object_cuboids.push_back(cuboid);
                for (auto& env : environments) {
                    env.z_aligned_cuboids.emplace_back(vamp::collision::factory::cuboid::array(cuboid.center, cuboid.euler_angles, cuboid.half_dimens));
                }
                env_static.z_aligned_cuboids.emplace_back(vamp::collision::factory::cuboid::array(cuboid.center, cuboid.euler_angles, cuboid.half_dimens));
            } else if (category.find("OBSTACLE") != std::string::npos) {
                obstacle_cuboids_map[obs_name].push_back(cuboid);
            } else if (category.find("TARGET") != std::string::npos) {
                target_cuboids.push_back(cuboid);
            }
        } else if(obj_type == "SPHERE") {
            float x, y, z, radius;

            iss >> x >> y >> z >> radius;
            const SphereData sphere(x, y, z, radius);

            if (category.find("OBJECT") != std::string::npos) {
                object_spheres.push_back(sphere);
                for (auto& env : environments) {
                    env.spheres.emplace_back(vamp::collision::factory::sphere::array(sphere.center, radius));
                }
                env_static.spheres.emplace_back(vamp::collision::factory::sphere::array(sphere.center, radius));
            } else if (category.find("OBSTACLE") != std::string::npos) {
                obstacle_spheres_map[obs_name].push_back(sphere);
            } else if (category.find("TARGET") != std::string::npos) {
                target_spheres.push_back(sphere);
            }
        } else if(obj_type == "CYLINDER") {
            float x, y, z, qx, qy, qz, qw, height, radius;

            iss >> x >> y >> z >> qx >> qy >> qz >> qw >> height >> radius;
            std::array<float, 3> angles = quaternion_to_euler(qx, qy, qz, qw);
            const CylinderData cylinder(x, y, z, angles[0], angles[1], angles[2], radius, height);

            if (category.find("OBJECT") != std::string::npos) {
                object_cylinders.push_back(cylinder);
                for (auto& env : environments) {
                    env.cylinders.emplace_back(vamp::collision::factory::cylinder::center::array(cylinder.center, cylinder.euler_angles, cylinder.radius, cylinder.length));
                }
                env_static.cylinders.emplace_back(vamp::collision::factory::cylinder::center::array(cylinder.center, cylinder.euler_angles, cylinder.radius, cylinder.length));
            } else if (category.find("OBSTACLE") != std::string::npos) {
                obstacle_cylinders_map[obs_name].push_back(cylinder);
            } else if (category.find("TARGET") != std::string::npos) {
                target_cylinders.push_back(cylinder);
            }
        }
    }

    env_static.sort();
    env_static_vec = EnvironmentVector(env_static);
    env_file.close();

    for (const auto& cuboid_entry : obstacle_cuboids_map) {
        std::string obs_name = cuboid_entry.first;
        std::vector<CuboidData> cuboids = cuboid_entry.second;

        if(cuboids.size() == 0)
            continue;
        if(cuboids.size() < MAX_WORLD_SAMPLES_EVAL) {
            std::cerr << "Buzz! Evironment samples not matched (Obstacle Cuboids)!" << std::endl;
            exit(-1);
        }
        for (size_t i = 0; i < MAX_WORLD_SAMPLES_EVAL; ++i)
            environments[i].cuboids.emplace_back(vamp::collision::factory::cuboid::array(cuboids[i].center, cuboids[i].euler_angles, cuboids[i].half_dimens));
    }

    for (const auto& sphere_entry : obstacle_spheres_map) {
        std::string obs_name = sphere_entry.first;
        std::vector<SphereData> spheres = sphere_entry.second;

        if(spheres.size() == 0)
            continue;
        if(spheres.size() < MAX_WORLD_SAMPLES_EVAL) {
            std::cerr << "Buzz! Evironment samples not matched (Obstacle Sphere)!" << std::endl;
            exit(-1);
        }
        for (size_t i = 0; i < MAX_WORLD_SAMPLES_EVAL; ++i)
            environments[i].spheres.emplace_back(vamp::collision::factory::sphere::array(spheres[i].center, spheres[i].radius));
    }


    for (const auto& cylinder_entry : obstacle_cylinders_map) {
        std::string obs_name = cylinder_entry.first;
        std::vector<CylinderData> cylinders = cylinder_entry.second;

        if(cylinders.size() == 0)
            continue;
        if(cylinders.size() < MAX_WORLD_SAMPLES_EVAL) {
            std::cerr << "Buzz! Evironment samples not matched (Obstacle Cylinders)!" << std::endl;
            exit(-1);
        }
        for (size_t i = 0; i < MAX_WORLD_SAMPLES_EVAL; ++i)
            environments[i].cylinders.emplace_back(vamp::collision::factory::cylinder::center::array(cylinders[i].center,
                            cylinders[i].euler_angles, cylinders[i].radius, cylinders[i].length));
    }

    assert(("Buzz! Evironment samples not matched (Target Cuboids)! " + std::to_string(target_cuboids.size()),
                                        ((target_cuboids.size() >= MAX_WORLD_SAMPLES) || target_cuboids.size() == 0)));
    assert(("Buzz! Evironment samples not matched (Target Spheres)! " + std::to_string(target_spheres.size()),
                                        ((target_spheres.size() >= MAX_WORLD_SAMPLES) || target_spheres.size() == 0)));
    assert(("Buzz! Evironment samples not matched (Target Cylinders)! " + std::to_string(target_cylinders.size()),
                                        ((target_cylinders.size() >= MAX_WORLD_SAMPLES) || target_cylinders.size() == 0)));
    for (size_t i = 0; i < target_cuboids.size(); ++i) {
        if(i >= environments.size())
            break;
        environments[i].cuboids.emplace_back(vamp::collision::factory::cuboid::array(target_cuboids[i].center, target_cuboids[i].euler_angles, target_cuboids[i].half_dimens));
    }
    for (size_t i = 0; i < target_spheres.size(); ++i) {
        if(i >= environments.size())
            break;
        environments[i].spheres.emplace_back(vamp::collision::factory::sphere::array(target_spheres[i].center, target_spheres[i].radius));
    }
    for (size_t i = 0; i < target_cylinders.size(); ++i) {
        if(i >= environments.size())
            break;
        environments[i].cylinders.emplace_back(vamp::collision::factory::cylinder::center::array(target_cylinders[i].center,
                            target_cylinders[i].euler_angles, target_cylinders[i].radius, target_cylinders[i].length));
    }

    for (size_t i = 0; i < environments.size(); ++i) {
        environments[i].sort();
        env_vectors[i] = EnvironmentVector(environments[i]);
    }

    std::cout << "###################################################################\n";
    std::cout << "Finish parsing the scene file and populating the vamp environment!" << std::endl;
    std::cout << env_static.z_aligned_cuboids.size() + env_static.cuboids.size() +
            env_static.spheres.size() + env_static.cylinders.size() << " | ";
    for (size_t i = 0; i < environments.size(); ++i) {
        std::cout << environments[i].z_aligned_cuboids.size() + environments[i].cuboids.size() +
                     environments[i].spheres.size() + environments[i].cylinders.size() << " | ";
    }
    std::cout << std::endl;
    std::cout << "###################################################################\n";
}
