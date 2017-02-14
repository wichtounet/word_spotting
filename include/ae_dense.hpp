//=======================================================================
// Copyright Baptiste Wicht 2015-2016.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef WORD_SPOTTER_AE_DENSE_HPP
#define WORD_SPOTTER_AE_DENSE_HPP

#include "ae_config.hpp" // for type aliases
#include "config.hpp"
#include "dataset.hpp"
#include "parameters.hpp" // for parameters

void dense_evaluate_all(const spot_dataset& dataset, const spot_dataset_set& set, config& conf, names train_word_names, names test_image_names, parameters params, const std::vector<image_t>& training_patches);
void deep_dense_evaluate_all(const spot_dataset& dataset, const spot_dataset_set& set, config& conf, names train_word_names, names test_image_names, parameters params, const std::vector<image_t>& training_patches);

#endif
