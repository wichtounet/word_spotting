//=======================================================================
// Copyright Baptiste Wicht 2015.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

/*!
 * \brief Select all the training images for the given keyword
 * \param dataset The current dataset
 * \param keyword The search keyword
 * \param train_names The used train names
 * \return A vector of all the relevant training images
 */
template <typename Dataset>
std::vector<std::string> select_training_images(const Dataset& dataset, names keyword, names train_names) {
    std::vector<std::string> training_images;

    for (auto& labels : dataset.word_labels) {
        if (keyword == labels.second && std::find(train_names.begin(), train_names.end(), labels.first) != train_names.end()) {
            training_images.push_back(labels.first);
        }
    }

    return training_images;
}
