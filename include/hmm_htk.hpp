//=======================================================================
// Copyright Baptiste Wicht 2015-2016.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#ifndef SPOTTER_NO_HMM

#define FULL_DEBUG

#include <random>

namespace hmm_htk {

using gmm_p = std::string;
using hmm_p = std::string;

//Number of states per character
static constexpr const auto n_states_per_char = 10;

template <typename RefFunctor>
gmm_p train_global_hmm(names train_word_names, RefFunctor functor) {
    dll::auto_timer timer("htk_gmm_train");

    auto ref_a = functor(train_word_names);

    const auto n_features = ref_a[0][0].size();

    //TODO Better Configure how the subset if selected
    std::size_t step = 5;

    //Collect information on the dataset

    std::size_t n_observations = 0;
    std::size_t n_images = 0;

    for(std::size_t image = 0; image < ref_a.size(); image += step){
        n_observations += ref_a[image].size();
        ++n_images;
    }

    //TODO

    return "frakking_gmm";
}

template <typename Dataset, typename Ref>
hmm_p train_ref_hmm(const Dataset& dataset, Ref& ref_a, names training_images) {
    dll::auto_timer timer("htk_hmm_train");

    auto characters = dataset.word_labels.at(training_images[0]).size();

    const auto n_states = characters * n_states_per_char;
    const auto n_features = ref_a[0][0].size();

    //TODO

    return "frakking_hmm";
}

template <typename Dataset, typename V1>
double hmm_distance(const Dataset& dataset, const gmm_p& gmm, const hmm_p& hmm, std::size_t pixel_width, const V1& test_image, names training_images) {
    double ref_width = 0;

    for(auto& image : training_images){
        ref_width += dataset.word_images.at(image + ".png").size().width;
    }

    ref_width /= training_images.size();

    auto ratio = ref_width / pixel_width;

    if (ratio > 2.0 || ratio < 0.5) {
        return 1e8;
    }

    const auto n_features = test_image[0].size();
    const auto width = test_image.size();

    //TODO

    return 1e8;
}

} //end of namespace hmm_mlpack

#else

namespace hmm_htk {

using gmm_p = std::string;
using hmm_p = std::string;

template <typename RefFunctor>
gmm_p train_global_hmm(names /*train_word_names*/, RefFunctor /*functor*/) {
    //Disabled HMM
    std::cerr << "HMM has been disabled, -hmm should not be used" << std::endl;
}

template <typename Dataset, typename Ref>
hmm_p train_ref_hmm(const Dataset& /*dataset*/, Ref& /*ref_a*/, names /*training_images*/) {
    //Disabled HMM
    std::cerr << "HMM has been disabled, -hmm should not be used" << std::endl;
}

template <typename Dataset, typename V1>
double hmm_distance(const Dataset& /*dataset*/, const gmm_p& /*global_hmm*/, const hmm_p& /*hmm*/, std::size_t /*pixel_width*/, const V1& /*test_image*/, names /*training_images*/) {
    //Disabled HMM
    std::cerr << "HMM has been disabled, -hmm should not be used" << std::endl;
}

} //end of namespace hmm_mlpack

#endif
