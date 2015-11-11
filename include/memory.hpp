//=======================================================================
// Copyright Baptiste Wicht 2015.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef WORD_SPOTTER_MEMORY_HPP
#define WORD_SPOTTER_MEMORY_HPP

#ifdef MEMORY_DEBUG

#include <unistd.h>
#include <ios>
#include <iostream>
#include <fstream>
#include <string>

inline std::string memory_to_string(double mem) {
    if (mem < 1024.0) {
        return std::to_string(mem) + "B";
    } else if (mem < 1024.0 * 1024.0) {
        return std::to_string(mem / 1024.0) + "KB";
    } else if (mem < 1024.0 * 1024.0 * 1024.0) {
        return std::to_string(mem / (1024.0 * 1024.0)) + "MB";
    } else {
        return std::to_string(mem / (1024.0 * 1024.0 * 1024.0)) + "GB";
    }
}

inline void memory_debug(const std::string& title) {
    unsigned long vsize;
    long rss;
    {
        std::string ignore;
        std::ifstream ifs("/proc/self/stat", std::ios_base::in);
        ifs >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> vsize >> rss;
    }

    long page_size      = sysconf(_SC_PAGE_SIZE);
    double vm_usage     = vsize;
    double resident_set = rss * page_size;

    std::cout << "memory: " << title << std::endl;
    std::cout << "memory: vm: " << memory_to_string(vm_usage) << std::endl;
    std::cout << "memory: rss: " << memory_to_string(resident_set) << std::endl;
    std::cout << std::endl;
}

#else

inline void memory_debug(const std::string& /*title*/) {}

#endif

#endif
