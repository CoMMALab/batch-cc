#pragma once


namespace batch_cc {
    void set_max_blocks(int max_blocks);

    template <typename Robot>
    void batch_cc(std::vector<ppln::collision::Environment<float>>& h_envs, std::vector<std::array<typename Robot::Configuration, 2>>& edges, int resolution, std::vector<uint8_t>& results);
} // namespace batch_cc
