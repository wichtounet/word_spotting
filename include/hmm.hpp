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
static constexpr const std::size_t n_gaussians = 1;

template <typename Dataset, typename Ref>
hmm_p train_ref_hmm(const Dataset& dataset, Ref& ref_a, names training_images) {
    auto characters = dataset.word_labels.at(training_images[0]).size();

    const auto n_states_per_char = 1;
    const auto n_states = characters * n_states_per_char;
    const auto n_features = ref_a[0][0].size();

    auto hmm = std::make_unique<HMM<GMM>>(n_states, GMM(n_gaussians, n_features));

    std::vector<arma::mat> images(ref_a.size());
    std::vector<arma::Row<size_t>> labels(ref_a.size());

    for(std::size_t image = 0; image < ref_a.size(); ++image){
        auto& ref_image = ref_a[image];

        auto width = ref_image.size();

        images[image] = arma::mat(n_features, width);
        labels[image] = arma::Row<size_t>(width);

        std::size_t current_label = 0;
        std::size_t label_distance = width / n_states;

        for(std::size_t i = 0; i < width; ++i){
            for(std::size_t j = 0; j < ref_image[i].size(); ++j){
                images[image](j, i) = ref_image[i][j];
            }

            labels[image](i) = std::min(current_label, n_states - 1);

            if(i > 0 && i % label_distance == 0){
                ++current_label;
            }
        }
    }

    try {
        hmm->Train(images, labels);
        std::cout << "HMM succesfully converged (with " << images.size() << " images)" << std::endl;
    } catch (const std::logic_error& e){
        std::cout << "frakking HMM failed: " << e.what() << std::endl;
        std::cout << "\tn_images: " << images.size() << std::endl;
        std::cout << "\tn_features: " << n_features << std::endl;
        std::cout << "\tn_states: " << n_states << std::endl;

        for(std::size_t i = 0; i < images.size(); ++i){
            auto& image = images[i];
            auto& label = labels[i];

            image.print("Image");
            label.print("Label");
        }
    } catch (const std::runtime_error& e){
        std::cout << "frakking HMM failed to converge: " << e.what() << std::endl;
        std::cout << "\tn_images: " << images.size() << std::endl;
        std::cout << "\tn_features: " << n_features << std::endl;
        std::cout << "\tn_states: " << n_states << std::endl;
    }

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
