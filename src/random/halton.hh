// adapted from VAMP: https://github.com/kavrakiLab/vamp

#pragma once

#include <algorithm>

namespace rng
{
    template <std::size_t dim>
    struct RNG
    {
        using Ptr = std::shared_ptr<RNG<dim>>;
        virtual inline void reset() noexcept = 0;
        virtual inline auto next() noexcept -> std::array<float, dim> = 0;

        // Distribution dist;
    };
}  // namespace vamp::rng

namespace rng
{
    template <std::size_t dim>
    struct Halton : public RNG<dim>
    {
        // Numerical precision degrades around 1.4M iterations, this value can be increased up to that point.
        static constexpr const std::size_t max_iterations = 1000000U;

        static constexpr const std::array<float, 16> primes{
            3.F,
            5.F,
            7.F,
            11.F,
            13.F,
            17.F,
            19.F,
            23.F,
            29.F,
            31.F,
            37.F,
            41.F,
            43.F,
            47.F,
            53.F,
            59.F};

        explicit Halton(std::array<float, dim> b_in) noexcept : b_init(b_in), b(b_in)
        {
            n.fill(0);
            d.fill(1);
        }

        Halton(std::initializer_list<float> v) noexcept : Halton(v)
        {
        }

        explicit Halton() : Halton(bases())
        {
        }

        inline constexpr auto bases() noexcept -> std::array<float, dim>
        {
            std::array<float, dim> a;
            std::copy_n(primes.cbegin(), dim, a.begin());
            return std::array<float, dim>(a);
        }

        auto rotate_bases() noexcept
        {
            std::rotate(b.begin(), b.begin() + 1, b.end());
        }

        const std::array<float, dim> b_init;
        std::array<float, dim> b;
        std::array<float, dim> n;
        std::array<float, dim> d;
        std::size_t iterations = 0;

        inline void reset() noexcept override final
        {
            iterations = 0;
            b = b_init;
            n.fill(0);
            d.fill(1);
        }

        inline auto next() noexcept -> std::array<float, dim> override final
        {
            iterations++;
            if (iterations > max_iterations)
            {
                n.fill(0);
                d.fill(1);
                iterations = 0;
                rotate_bases();
            }
            std::array<float, dim> result;
            for (size_t i = 0; i < dim; i++) {
                float xf = d[i] - n[i];
                bool x_eq_1 = (xf == 1.0f);
                
                if (x_eq_1) {
                    // x == 1 case
                    d[i] = floorf(d[i] * b[i]);
                    n[i] = 1.0f;
                } else {
                    // x != 1 case
                    float y = floorf(d[i] / b[i]);
                    
                    // Continue dividing by b until we find the right digit position
                    while (xf <= y) {
                        y = floorf(y / b[i]);
                    }
                    
                    n[i] = floorf((b[i] + 1.0f) * y) - xf;
                }
                
                result[i] = n[i] / d[i];
            }
            return result;
        }
    };
}  // namespace vamp::rng
