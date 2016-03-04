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

//Number of gaussians for the HMM
static constexpr const std::size_t n_hmm_gaussians = 8;

//Number of gaussians for the GMM
static constexpr const std::size_t n_gmm_gaussians = 64;

template <typename RefFunctor>
hmm_p train_global_hmm(names train_word_names, RefFunctor functor) {
    dll::auto_timer("gmm_train");

    auto ref_a = functor(train_word_names);

    const auto n_states = 1;
    const auto n_features = ref_a[0][0].size();

    auto hmm = std::make_unique<HMM<GMM>>(n_states, GMM(n_gmm_gaussians, n_features));

    std::vector<arma::mat> images;

    //TODO Configure how the subset if sleected

    for(std::size_t image = 0; image < ref_a.size(); image += 10){
        auto& ref_image = ref_a[image];

        auto width = ref_image.size();

        images.emplace_back(n_features, width);

        for(std::size_t i = 0; i < width; ++i){
            for(std::size_t j = 0; j < ref_image[i].size(); ++j){
                images.back()(j, i) = ref_image[i][j];
            }
        }
    }

    try {
        std::cout << "Start training the global HMM (with " << images.size() << " images)" << std::endl;
        hmm->Train(images);
        std::cout << "HMM succesfully converged (with " << images.size() << " images)" << std::endl;
    } catch (const std::logic_error& e){
        std::cout << "frakking HMM failed: " << e.what() << std::endl;
        std::cout << "\tn_images: " << images.size() << std::endl;
        std::cout << "\tn_features: " << n_features << std::endl;
        std::cout << "\tn_states: " << n_states << std::endl;

        for(auto& image : images){
            image.print("Image");
        }
    } catch (const std::runtime_error& e){
        std::cout << "frakking HMM failed to converge: " << e.what() << std::endl;
        std::cout << "\tn_images: " << images.size() << std::endl;
        std::cout << "\tn_features: " << n_features << std::endl;
        std::cout << "\tn_states: " << n_states << std::endl;
    }

    return hmm;
}

template <typename Dataset, typename Ref>
hmm_p train_ref_hmm(const Dataset& dataset, Ref& ref_a, names training_images) {
    dll::auto_timer("hmm_train");

    auto characters = dataset.word_labels.at(training_images[0]).size();

    const auto n_states_per_char = 5;
    const auto n_states = characters * n_states_per_char;
    const auto n_features = ref_a[0][0].size();

    auto hmm = std::make_unique<HMM<GMM>>(n_states, GMM(n_hmm_gaussians, n_features));

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

template <typename Dataset, typename V1>
double hmm_distance(const Dataset& dataset, const hmm_p& global_hmm, const hmm_p& hmm, std::size_t pixel_width, const V1& test_image, names training_images) {
    double ref_width = 0;

    for(auto& image : training_images){
        ref_width += dataset.word_images.at(image + ".png").size().width;
    }

    ref_width /= training_images.size();

    auto ratio = ref_width / pixel_width;

    if (ratio > 2.0 || ratio < 0.5) {
        return 1e100;
    }

    //const auto n_states = hmm->Initial().size();
    const auto n_features = test_image[0].size();
    const auto width = test_image.size();

    arma::mat image(n_features, width);

    for(std::size_t i = 0; i < width; ++i){
        for(std::size_t j = 0; j < test_image[i].size(); ++j){
            image(j, i) = test_image[i][j];
        }
    }

    return -(hmm->LogLikelihood(image) / global_hmm->LogLikelihood(image));
}

#else

using hmm_p = int;

template <typename Dataset, typename Ref>
hmm_p train_ref_hmm(const Dataset& /*dataset*/, Ref& /*ref_a*/, names /*training_images*/) {
    //Disabled HMM
    std::cerr << "HMM has been disabled, -hmm should not be used" << std::endl;
}

template <typename Dataset, typename V1>
double hmm_distance(const Dataset& /*dataset*/, const hmm_p& /*hmm*/, std::size_t /*pixel_width*/, const V1& /*s*/, names /*training_images*/) {
    //Disabled HMM
    std::cerr << "HMM has been disabled, -hmm should not be used" << std::endl;
}

#endif

#endif
