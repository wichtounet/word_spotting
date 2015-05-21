//=======================================================================
// Copyright Baptiste Wicht 2015.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef WORD_SPOTTER_HOLISTIC_METHOD_HPP
#define WORD_SPOTTER_HOLISTIC_METHOD_HPP

#include "config.hpp"
#include "washington.hpp"

void holistic_method(const washington_dataset& dataset, const washington_dataset_set& set, config& conf, const std::vector<std::string>& train_word_names, const std::vector<std::string>& train_image_names, const std::vector<std::string>& test_image_names);

#endif
