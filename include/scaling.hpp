//=======================================================================
// Copyright Baptiste Wicht 2015.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef WORD_SPOTTER_SCALING_HPP
#define WORD_SPOTTER_SCALING_HPP

template<typename Features>
void local_linear_feature_scaling(std::vector<Features>& features){
    const auto width = features.size();

    for(std::size_t f = 0; f < features.back().size(); ++f){
        double A = features[0][f];
        double B = features[0][f];

        for(std::size_t i = 1; i < width; ++i){
            A = std::min(A, features[i][f]);
            B = std::max(B, features[i][f]);
        }

        auto scale = [A,B](auto x){ return (x - A) / (B - A); };

        for(std::size_t i = 0; i < width; ++i){
            features[i][f] = scale(features[i][f]);
        }
    }
}

template<typename Features>
void local_mean_feature_scaling(std::vector<Features>& features){
    const auto width = features.size();

    for(std::size_t f = 0; f < features.back().size(); ++f){
        // Compute the mean
        double mean = 0.0;
        for(std::size_t i = 0; i < width; ++i){
            mean += features[i][f];
        }
        mean /= width;
        //Normalize to zero-mean
        for(std::size_t i = 0; i < width; ++i){
            features[i][f] -= mean;
        }
        //Compute the variance
        double std = 0.0;
        for(std::size_t i = 0; i < width; ++i){
            std += features[i][f] * features[i][f];
        }
        std = std::sqrt(std / width);
        //Normalize to unit variance
        if(std != 0.0){
            for(std::size_t i = 0; i < width; ++i){
                features[i][f] /= std;
            }
        }
    }
}


template<typename Features>
auto global_mean_scaling(Features& features, config& conf, bool training){
    if(training){
        for(std::size_t f = 0; f < features.back().back().size(); ++f){
            // Compute the mean
            double mean = 0.0;
            double count = 0;
            for(std::size_t t = 0; t < features.size(); ++t){
                for(std::size_t i = 0; i < features[t].size(); ++i){
                    mean += features[t][i][f];
                    ++count;
                }
            }
            mean /= count;

            //Compute the variance
            double std = 0.0;
            count = 0.0;
            for(std::size_t t = 0; t < features.size(); ++t){
                for(std::size_t i = 0; i < features[t].size(); ++i){
                    std += (features[t][i][f] - mean) * (features[t][i][f] - mean);
                    ++count;
                }
            }
            std = std::sqrt(std / count);

            conf.scale_a[f] = mean;
            conf.scale_b[f] = std;
        }
    }

    return [](double x, double mean, double std) -> double { return std == 0.0 ? x - mean : (x - mean) / std; };
}

template<typename Features>
auto global_linear_scaling(Features& features, config& conf, bool training){
    if(training){
        for(std::size_t f = 0; f < features.back().back().size(); ++f){
            double A = features[0][0][f];
            double B = features[0][0][f];

            for(std::size_t t = 0; t < features.size(); ++t){
                for(std::size_t i = 0; i < features[t].size(); ++i){
                    A = std::min(A, features[t][i][f]);
                    B = std::max(B, features[t][i][f]);
                }
            }

            conf.scale_a[f] = A;
            conf.scale_b[f] = B;
        }
    }

    return [](double x, double A, double B) -> double { return (x - A) / (B - A); };
}

#endif
