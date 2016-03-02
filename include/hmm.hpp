//=======================================================================
// Copyright Baptiste Wicht 2015-2016.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef WORD_SPOTTER_HMM_HPP
#define WORD_SPOTTER_HMM_HPP

#ifndef SPOTTER_NO_HMM

#include <mlpack/core.hpp>
#include <mlpack/methods/hmm/hmm.hpp>
#include <mlpack/methods/gmm/gmm.hpp>

using GMM = mlpack::gmm::GMM;

template<typename Distribution>
using HMM = mlpack::hmm::HMM<Distribution>;

using hmm_p = std::unique_ptr<HMM<GMM>>;

//Number of gaussians
static constexpr const std::size_t n_gaussians = 2;

template <typename Dataset, typename Ref>
hmm_p train_ref_hmm(const Dataset& dataset, Ref& ref_a, names training_images) {
    auto characters = dataset.word_labels.at(training_images[0]).size();

    const auto n_states_per_char = 10;
    const auto n_states = characters * n_states_per_char;
    const auto n_features = ref_a[0][0].size();

    auto hmm = std::make_unique<HMM<GMM>>(n_states, GMM(n_gaussians, n_features));

    hmm->Tolerance() *= 100;

    std::vector<arma::mat> images;
    std::vector<arma::Row<size_t>> labels;

    for(std::size_t i = 0; i < ref_a.size(); ++i){
        auto& ref_image = ref_a[i];

        std::cout << ref_image.size() << std::endl;

        auto width = ref_image.size();

        images.emplace_back(n_features, width);
        labels.emplace_back(width);

        auto& training_image = images.back();
        auto& training_labels = labels.back();

        std::size_t current_label = 0;
        std::size_t label_distance = width / n_states;

        for(std::size_t i = 0; i < width; ++i){
            for(std::size_t j = 0; j < ref_image[i].size(); ++j){
                training_image.col(i)[j] = ref_image[i][j];
            }

            training_labels[i] = std::min(current_label, n_states - 1);

            if(i > 0 && i % label_distance == 0){
                ++current_label;
            }
        }
    }

    std::cout << image.size() << ":" << labels.size() << std::endl;

    hmm->Train(images, labels);

    return hmm;
}

#else

using hmm_p = int;

template <typename Dataset, typename Ref>
hmm_p train_ref_hmm(const Dataset& /*dataset*/, Ref& /*ref_a*/, names /*training_images*/) {
    //Disabled HMM
    std::cerr << "HMM has been disabled, -hmm should not be used" << std::endl;
}

#endif

#endif
