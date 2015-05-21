//=======================================================================
// Copyright Baptiste Wicht 2015.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef WORD_SPOTTER_REPORTS_HPP
#define WORD_SPOTTER_REPORTS_HPP

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

#include "washington.hpp"

std::string select_folder(const std::string& base_folder);

void generate_rel_files(
    const std::string& result_folder,
    const washington_dataset& dataset, const washington_dataset_set& set,
    const std::vector<std::string>& test_image_names);

void update_stats(
    std::size_t k,
    const std::string& result_folder,
    const washington_dataset& dataset,
    const std::vector<std::string>& keyword,
    std::vector<std::pair<std::string, weight>> diffs_a,
    std::vector<double>& eer, std::vector<double>& ap,
    std::ofstream& global_top_stream, std::ofstream& local_top_stream,
    const std::vector<std::string>& test_image_names);

#endif