//=======================================================================
// Copyright Baptiste Wicht 2015-2016.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "config.hpp"
#include "scaling.hpp"

namespace spot {

template <typename Patch>
void normalize_patch_features(Patch& features){
    cpp_unused(features);

#ifdef LOCAL_FRAME_NORMALIZATION
    for (std::size_t i = 0; i < etl::dim<0>(features); ++i) {
        features(i) /= etl::sum(features(i));
    }
#endif

#ifdef LOCAL_L2_NORMALIZATION
    for (std::size_t i = 0; i < etl::dim<0>(features); ++i) {
        features(i) /= std::sqrt(etl::sum(features(i) >> features(i)) + 16.0 * 16.0);
    }
#endif

#ifdef GLOBAL_FRAME_NORMALIZATION
    features /= etl::sum(features);
#endif

#ifdef GLOBAL_L2_NORMALIZATION
    features /= std::sqrt(etl::sum(features >> features) + 16.0 * 16.0);
#endif
}

template <typename Features>
void normalize_feature_vector(Features& vec){
    // 1. Normalize the features of each patch
    for(auto& features : vec){
        normalize_patch_features(features);
    }

    // 2. Globally normalize the features

#ifdef LOCAL_LINEAR_SCALING
    local_linear_feature_scaling(vec);
#endif

#ifdef LOCAL_MEAN_SCALING
    local_mean_feature_scaling(vec);
#endif
}

template <typename Features>
void normalize_features(const config& conf, bool training, Features& features){
    cpp_unused(features);
    cpp_unused(conf);
    cpp_unused(training);

#ifdef GLOBAL_LINEAR_SCALING
    auto scale = global_linear_scaling(features, conf, training);
#endif

#ifdef GLOBAL_MEAN_SCALING
    auto scale = global_mean_scaling(features, conf, training);
#endif

#ifdef GLOBAL_SCALING
    for (std::size_t t = 0; t < features.size(); ++t) {
        for (std::size_t i = 0; i < features[t].size(); ++i) {
            for (std::size_t f = 0; f < features.back().back().size(); ++f) {
                features[t][i][f] = scale(features[t][i][f], conf.scale_a[f], conf.scale_b[f]);
            }
        }
    }
#endif
}

} // end of namespace spot
