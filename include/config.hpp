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

struct config {
    std::vector<std::string> args;
    std::vector<std::string> files;
    std::string command;
    bool method_1 = false;
    bool half = false;
    bool quarter = false;
    bool third = false;
    bool svm = false;
    bool view = false;
};

void print_usage();

config parse_args(int argc, char** argv);

#endif
