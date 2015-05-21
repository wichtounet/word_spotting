//=======================================================================
// Copyright Baptiste Wicht 2015.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <string>

#include <sys/stat.h>

static constexpr const bool generate_graphs = false;
using weight = double; //TODO Should probably be declared globally

#include "reports.hpp"

std::string select_folder(const std::string& base_folder){
    std::cout << "Select a folder ..." << std::endl;

    mkdir(base_folder.c_str(), 0777);

    std::size_t result_name = 0;

    std::string result_folder;

    struct stat buffer;
    do {
        ++result_name;
        result_folder = base_folder + std::to_string(result_name);
    } while(stat(result_folder.c_str(), &buffer) == 0);

    mkdir(result_folder.c_str(), 0777);

    std::cout << "... " << result_folder << std::endl;

    return result_folder;
}
