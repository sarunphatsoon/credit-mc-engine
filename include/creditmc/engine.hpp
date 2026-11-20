#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace creditmc {

struct Portfolio {
    std::vector<double> pd;   // probability of default per obligor
    std::vector<double> ead;  // exposure at default
    std::vector<double> lgd;  // loss given default
    double rho;               // asset correlation
};

double norm_cdf(double x);
double norm_inv_cdf(double p);

std::vector<double> simulate_losses(const Portfolio& portfolio,
                                    std::size_t n_scenarios,
                                    std::uint64_t seed,
                                    unsigned n_threads = 0);

double irb_asymptotic_loss(double pd, double lgd, double rho, double confidence);

}  // namespace creditmc

