#pragma once

struct pRRTC_settings {
    int max_samples = 1000000;
    int max_iters = 1000000;
    int num_new_configs = 600;
    int granularity = 64;
    float range = 0.5;

    int balance = 1; // 0 = no balance, 1 = balance strategy 1, 2 = balance strategy 2
    float tree_ratio = 0.5; // 0.5 for balance=1, 1.0 for balance=2

    bool dynamic_domain = true;
    float dd_alpha = 0.0001;
    float dd_radius = 4.0;
    float dd_min_radius = 1.0;
};