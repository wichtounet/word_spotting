//=======================================================================
// Copyright Baptiste Wicht 2015-2017.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <string>
#include <vector>

struct grid_info {
    std::string password;
    std::vector<std::string> machines;
    std::string dist_folder;
};

grid_info& load_grid_info();
