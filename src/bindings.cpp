#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstring>
#include <stdexcept>
#include <string>

#include "creditmc/engine.hpp"

namespace py = pybind11;
using DoubleArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

namespace {

std::vector<double> to_vector(const DoubleArray& arr, const char* name) {
    if (arr.ndim() != 1) {
        throw std::invalid_argument(std::string(name) + " must be a 1-D array");
    }
    const double* data = arr.data();
    return std::vector<double>(data, data + arr.size());
}

py::array_t<double> simulate(const DoubleArray& pd, const DoubleArray& ead, const DoubleArray& lgd, 
                            double rho, std::size_t n_scenarios, std::uint64_t seed,
                            unsigned n_threads) {
    creditmc::Portfolio portfolio{to_vector(pd, "pd"), to_vector(ead, "ead"),
                                  to_vector(lgd, "lgd"), rho};

    std::vector<double> losses;
    {
        py::gil_scoped_release release;
        losses = creditmc::simulate_losses(portfolio, n_scenarios, seed, n_threads);
    }

    py::array_t<double> result(static_cast<py::ssize_t>(losses.size()));
    std::memcpy(result.mutable_data(), losses.data(), losses.size() * sizeof(double));
    return result;
}

}

PYBIND11_MODULE(creditmc_core, m) {
    m.def("simulate", &simulate,
          py::arg("pd"), py::arg("ead"), py::arg("lgd"), py::arg("rho"),
          py::arg("n_scenarios"), py::arg("seed") = 42, py::arg("n_threads") = 0);
    
    m.def("irb_asymptotic_loss", &creditmc::irb_asymptotic_loss,
          py::arg("pd"), py::arg("lgd"), py::arg("rho"), py::arg("confidence") = 0.999);

    m.def("norm_inv_cdf", &creditmc::norm_inv_cdf, py::arg("p"));
    m.def("norm_cdf", &creditmc::norm_cdf, py::arg("x"));
}