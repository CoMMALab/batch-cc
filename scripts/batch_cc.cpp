#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>

#include "src/collision/environment.hh"
#include "src/collision/factory.hh"
#include "src/Planners.hh"
#include "src/pRRTC_settings.hh"

#include "src/batch_cc.hh"

#include "src/random/halton.hh"

using json = nlohmann::json;

using namespace ppln::collision;


Environment<float> problem_dict_to_env(const json& problem, const std::string& name) {
    Environment<float> env{};
    
    std::vector<Sphere<float>> spheres;
    std::vector<Capsule<float>> capsules;
    std::vector<Cuboid<float>> cuboids;
    // Fill spheres
    for (const auto& obj : problem["sphere"]) {
        const json& position = obj["position"];
        Sphere<float> sphere(position[0], position[1], position[2], obj["radius"]);
        sphere.name = obj["name"];
        spheres.push_back(sphere);
    }
    // Handle cylinders based on name
    if (name == "box") {
        for (const auto& obj : problem["cylinder"]) {
            const json& position = obj["position"];
            const json& orientation = obj["orientation_euler_xyz"];
            const float radius = obj["radius"];
            const std::array<float, 3> dims = {radius, radius, radius/2.0f};
            auto cuboid = factory::cuboid::array(
                position, orientation,
                dims
            );
            cuboid.name = obj["name"];
            cuboids.push_back(cuboid);
        }
    } else {
        for (const auto& obj : problem["cylinder"]) {
            const json& position = obj["position"];
            const json& orientation = obj["orientation_euler_xyz"];
            const float radius = obj["radius"];
            const float length = obj["length"];
            auto cylinder = factory::cylinder::center::array(
                position, orientation,
                radius, length
            );
            cylinder.name = obj["name"];
            capsules.push_back(cylinder);
        }
    }

    // Fill boxes
    for (const auto& obj : problem["box"]) {
        const json& position = obj["position"];
        const json& orientation = obj["orientation_euler_xyz"];
        const json& half_extents = obj["half_extents"];
        auto cuboid = factory::cuboid::array(
            position, orientation, half_extents
        );
        cuboid.name = obj["name"];
        cuboids.push_back(cuboid);
    }

    // Allocate memory on the heap for the arrays
    if (!spheres.empty()) {
        env.spheres = new Sphere<float>[spheres.size()];
        std::copy(spheres.begin(), spheres.end(), env.spheres);
        env.num_spheres = spheres.size();
    }

    if (!capsules.empty()) {
        env.capsules = new Capsule<float>[capsules.size()];
        std::copy(capsules.begin(), capsules.end(), env.capsules);
        env.num_capsules = capsules.size();
    }

    if (!cuboids.empty()) {
        env.cuboids = new Cuboid<float>[cuboids.size()];
        std::copy(cuboids.begin(), cuboids.end(), env.cuboids);
        env.num_cuboids = cuboids.size();
    }

    return env;
}

template<typename Robot>
void run_test(int num_envs, int num_edges, int resolution, std::string robot_name) {
    std::string path = "scripts/" + robot_name + "_problems.json";
    std::ifstream f(path);
    json all_data = json::parse(f);
    json problems = all_data["problems"];
    std::vector<ppln::collision::Environment<float>*> h_envs;
    for (auto& [name, pset] : problems.items()) {
        for (int i = 0; i < pset.size(); i++) {
            json data = pset[i];
            if (not data["valid"]) {
                continue;
            }
            auto env = new ppln::collision::Environment<float>(problem_dict_to_env(data, name));
            // print environment fields
            // std::cout << env->num_spheres << " spheres, "
            //           << env->num_capsules << " capsules, "
            //           << env->num_cuboids << " cuboids\n";

            h_envs.push_back(env);
            if (h_envs.size() >= num_envs) {
                break;
            }
        }
        if (h_envs.size() >= num_envs) {
            break;
        }
    }

    rng::Halton<Robot::dimension> rng;
    std::vector<std::array<typename Robot::Configuration, 2>> edges(num_edges);
    for (int i = 0; i < num_edges; i++) {
        typename Robot::Configuration start, end;
        auto start_sample = rng.next();
        auto end_sample = rng.next();
        Robot::scale_cfg(start_sample.data());
        Robot::scale_cfg(end_sample.data());
        for (int j = 0; j < Robot::dimension; j++) {
            start[j] = start_sample[j];
            end[j] = end_sample[j];
        }
        edges[i][0] = start;
        edges[i][1] = end;
    }
    // Print the edges for debugging
    // std::cout << "Edges:\n";
    // for (int i = 0; i < num_edges; i++) {
    //     std::cout << "Edge " << i << ": Start: ";
    //     for (int j = 0; j < Robot::dimension; j++) {
    //         std::cout << edges[i][0][j] << " ";
    //     }
    //     std::cout << " End: ";
    //     for (int j = 0; j < Robot::dimension; j++) {
    //         std::cout << edges[i][1][j] << " ";
    //     }
    //     std::cout << "\n";
    // }
    
    std::vector<bool> results(num_edges * num_envs);
    batch_cc::batch_cc<Robot>(h_envs, edges, resolution, results);
    for (int i = 0; i < num_edges; i++) {
        std::cout << "Edge " << i << ": ";
        for (int j = 0; j < num_envs; j++) {
            std::cout << results[i * num_envs + j] << " ";
        }
        std::cout << "\n";
    }
    // print h_envs for debugging, including pointers
    // std::cout << "Number of environments: " << h_envs.size() << "\n";
    // for (int i = 0; i < h_envs.size(); i++) {
    //     std::cout << "Environment " << i << ":\n";
    //     std::cout << h_envs[i]->num_spheres << " spheres, "
    //               << h_envs[i]->num_capsules << " capsules, "
    //               << h_envs[i]->num_cuboids << " cuboids, "
    //               << h_envs[i]->spheres << " spheres pointer, "
    //               << h_envs[i]->capsules << " capsules pointer, "
    //               << h_envs[i]->cuboids << " cuboids pointer\n";
    // }

    // Clean up the environments
    for (auto env : h_envs) {
        delete env;
    }
}

int main(int argc, char* argv[]) {
    std::string robot_name = "panda";
    int num_envs = 1;
    int resolution = 32;
    int num_edges = 1;
    if (argc == 4) {
        robot_name = argv[1];
        num_envs = std::stoi(argv[2]);
        num_edges = std::stoi(argv[3]);
    }
    else {
        std::cout << "Usage: ./batch_cc <robot_name> <num_environments> <num_edges>\n";
        return 1;
    }

    size_t nanoseconds = -1;
    if (robot_name == "panda") {
        run_test<robots::Panda>(num_envs, num_edges, resolution, robot_name);
    }
    else if (robot_name == "fetch") {
        run_test<robots::Fetch>(num_envs, num_edges, resolution, robot_name);
    }
    else {
        std::cout << "Unknown robot name: " << robot_name << "\n";
        return -1;
    }
    return 0;
}