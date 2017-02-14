//=======================================================================
// Copyright Baptiste Wicht 2015-2017.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef WORD_SPOTTER_HOLISTIC_METHOD_HPP
#define WORD_SPOTTER_HOLISTIC_METHOD_HPP

#include "config.hpp"
#include "dataset.hpp"

void holistic_train(const spot_dataset& dataset, const spot_dataset_set& set, config& conf,
                    names train_word_names,
                    names train_image_names, names valid_image_names, names test_image_names, bool features = false);

void holistic_features(const spot_dataset& dataset, const spot_dataset_set& set, config& conf,
                       names train_word_names,
                       names train_image_names, names valid_image_names, names test_image_names);

#endif
