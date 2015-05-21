//=======================================================================
// Copyright Baptiste Wicht 2015.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef WORD_SPOTTER_CONFIG_HPP
#define WORD_SPOTTER_CONFIG_HPP

#include <vector>
#include <string>

static constexpr const std::size_t WIDTH = 660;
static constexpr const std::size_t HEIGHT = 120;

static constexpr const bool generate_graphs = false;

constexpr const std::size_t MAX_N = 25;

using weight = double;

struct config {
    std::vector<std::string> args;
    std::vector<std::string> files;
    std::string command;
    bool method_0 = false;
    bool method_1 = false;
    bool method_2 = false;
    bool half = false;
    bool quarter = false;
    bool third = false;
    bool svm = false;
    bool view = false;
    bool sub = false;

    //The following values are set during execution
    std::size_t downscale = 1;
    std::size_t patch_width = 0;
    std::size_t patch_stride = 0;

    std::vector<double> scale_a;
    std::vector<double> scale_b;

    config() : scale_a(4096), scale_b(4096) {}

    config(const config& config) = default;
    config(config&& config) = default;
    config& operator=(const config& config) = default;
    config& operator=(config&& config) = default;
};

void print_usage();

config parse_args(int argc, char** argv);

#define LOCAL_MEAN_SCALING

static_assert(WIDTH % 2 == 0, "Width must be divisible by 2");
static_assert(HEIGHT % 2 == 0, "Height must be divisible by 2");

static_assert(WIDTH % 3 == 0, "Width must be divisible by 4");
static_assert(HEIGHT % 3 == 0, "Height must be divisible by 4");

static_assert(WIDTH % 4 == 0, "Width must be divisible by 4");
static_assert(HEIGHT % 4 == 0, "Height must be divisible by 4");

#endif
