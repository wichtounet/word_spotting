//=======================================================================
// Copyright Baptiste Wicht 2015-2017.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef WORD_SPOTTER_FEATURES_HPP
#define WORD_SPOTTER_FEATURES_HPP

#include "hmm_htk.hpp"

template <typename Features>
void export_features(const config& conf, const std::vector<std::string>& images, Features& all_features, const std::string& suffix) {
    for (std::size_t t = 0; t < images.size(); ++t) {
        decltype(auto) features = all_features[t];

        {
            auto features_path = conf.data_full_path + images[t] + suffix;

            std::ofstream os(features_path);

            for (auto& f : features) {
                std::string comma;

                for (auto& v : f) {
                    os << comma << v;
                    comma = ";";
                }

                os << '\n';
            }
        }

        // Export HTK features file if asked
        if(conf.htk){
            auto htk_features_path = conf.data_full_path + images[t] + suffix + ".htk";

            std::ofstream os(htk_features_path, std::ofstream::binary);
            hmm_htk::htk_features_write(os, features);
        }
    }
}

template <typename Features>
void export_features_flat(const config& conf, const std::vector<std::string>& images, Features& all_features, const std::string& suffix) {
    for (std::size_t t = 0; t < images.size(); ++t) {
        auto features_path = conf.data_full_path + images[t] + suffix;
        decltype(auto) features = all_features[t];

        std::ofstream os(features_path);

        std::string comma;

        for (auto& v : features) {
            os << comma << v;
            comma = ";";
        }
    }
}

#endif
