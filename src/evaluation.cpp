//=======================================================================
// Copyright Baptiste Wicht 2015-2016.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "evaluation.hpp"

std::vector<std::string> select_training_images(const spot_dataset& dataset, names keyword, names train_names) {
    std::vector<std::string> training_images;

    for (auto& labels : dataset.word_labels) {
        if (keyword == labels.second && std::find(train_names.begin(), train_names.end(), labels.first) != train_names.end()) {
            training_images.push_back(labels.first);
        }
    }

    return training_images;
}
