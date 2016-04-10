//=======================================================================
// Copyright Baptiste Wicht 2015-2016.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <fstream>

#include "grid.hpp"

grid_info& load_grid_info(){
    static grid_info info;

    if (info.password.empty()) {
        {
            std::ifstream is(".grid_passwd");
            is >> info.password;
        }

        std::cout << "Grid Password loaded" << std::endl;

        {
            std::ifstream is(".grid_machines");

            std::string line;
            while (std::getline(is, line)) {
                info.machines.push_back(line);
            }
        }

        std::cout << info.machines.size() << " grid machines loaded" << std::endl;
    }

    return info;
}
