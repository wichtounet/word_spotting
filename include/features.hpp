//=======================================================================
// Copyright Baptiste Wicht 2015.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef WORD_SPOTTER_FEATURES_HPP
#define WORD_SPOTTER_FEATURES_HPP

template<typename Features>
void export_features(config& conf, const std::vector<std::string>& images, Features& all_features, const std::string& suffix){
    for(std::size_t t = 0; t < images.size(); ++t){
        auto features_path = conf.data_full_path + images[t] + suffix;
        decltype(auto) features = all_features[t];

        std::ofstream os(features_path);

        for(auto& f : features){
            std::string comma;

            for(auto& v : f){
                os << comma << v;
                comma = ";";
            }

            os << '\n';
        }
    }
}

#endif
