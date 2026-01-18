#pragma once


namespace batch_cc {
    void set_max_blocks(int max_blocks);
    enum class TwoPhaseMode {
        FullOnly = 0,
        ApproxGated = 1
    };
    void set_two_phase_mode(TwoPhaseMode mode);

    template <typename Robot>
    void batch_cc(std::vector<ppln::collision::Environment<float>>& h_envs, std::vector<std::array<typename Robot::Configuration, 2>>& edges, int resolution, std::vector<uint8_t>& results);
} // namespace batch_cc
