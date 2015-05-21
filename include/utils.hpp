//=======================================================================
// Copyright Baptiste Wicht 2015.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef WORD_SPOTTER_UTILS_HPP
#define WORD_SPOTTER_UTILS_HPP

#include <vector>
#include <string>

template<typename T>
std::ostream& operator<<(std::ostream& stream, const std::vector<T>& vec){
    std::string comma = "";
    stream << "[";
    for(auto& v : vec){
        stream << comma << v;
        comma = ", ";
    }
    stream << "]";

    return stream;
}

template<typename T>
std::string keyword_to_string(const std::vector<T>& vec){
    std::string comma = "";
    std::string result;
    result += "[";
    for(auto& v : vec){
        result += comma;
        result += v;
        comma = ", ";
    }
    result += "]";

    return result;
}

#endif
