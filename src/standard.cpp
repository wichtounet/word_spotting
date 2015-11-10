//=======================================================================
// Copyright Baptiste Wicht 2015.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <string>

#include "etl/etl.hpp"

#include "cpp_utils/parallel.hpp"

#include "config.hpp"
#include "standard.hpp"
#include "utils.hpp"
#include "reports.hpp"
#include "dtw.hpp"        //Dynamic time warping

#define LOCAL_MEAN_SCALING
#include "scaling.hpp"      //Scaling functions

namespace {

std::vector<etl::dyn_vector<weight>> standard_features(const cv::Mat& clean_image){
    std::vector<etl::dyn_vector<weight>> features;

    const auto width = static_cast<std::size_t>(clean_image.size().width);
    const auto height = static_cast<std::size_t>(clean_image.size().height);

    for(std::size_t i = 0; i < width; ++i){
        double lower = 0.0;
        for(std::size_t y = 0; y < height; ++y){
            if(clean_image.at<uint8_t>(y, i) == 0){
                lower = y;
                break;
            }
        }

        double upper = 0.0;
        for(std::size_t y = height - 1; y > 0; --y){
            if(clean_image.at<uint8_t>(y, i) == 0){
                upper = y;
                break;
            }
        }

        std::size_t black = 0;
        for(std::size_t y = 0; y < height; ++y){
            if(clean_image.at<uint8_t>(y, i) == 0){
                ++black;
            }
        }

        std::size_t inner_black = 0;
        for(std::size_t y = lower; y < upper + 1; ++y){
            if(clean_image.at<uint8_t>(y, i) == 0){
                ++inner_black;
            }
        }

        std::size_t transitions = 0;
        for(std::size_t y = 1; y < height; ++y){
            if(clean_image.at<uint8_t>(y-1, i) == 0 && clean_image.at<uint8_t>(y, i) != 0){
                ++transitions;
            }
        }

        double gravity = 0;
        double moment = 0;
        for(std::size_t y = 0; y < height; ++y){
            auto pixel = clean_image.at<uint8_t>(y, i) == 0 ? 0.0 : 1.0;
            gravity += y * pixel;
            moment += y * y * pixel;
        }
        gravity /= height;
        moment /= (height * height);

        features.emplace_back(9);

        auto& f = features.back();

        f[0] = black;
        f[1] = gravity;
        f[2] = moment;
        f[3] = lower;
        f[4] = upper;
        f[5] = 0.0;
        f[6] = 0.0;
        f[7] = transitions;
        f[8] = inner_black;
    }

    for(std::size_t i = 0; i < width - 1; ++i){
        features[i][5] = features[i+1][1] - features[i][1];
        features[i][6] = features[i+1][2] - features[i][2];
    }

#ifdef LOCAL_LINEAR_SCALING
    local_linear_feature_scaling(features);
#endif

#ifdef LOCAL_MEAN_SCALING
    local_mean_feature_scaling(features);
#endif

    return features;
}


template<typename Dataset, typename Set>
void evaluate_dtw(const Dataset& dataset, const Set& set, config& conf, const std::vector<std::string>& train_word_names, const std::vector<std::string>& test_image_names, bool training){
    auto keywords = select_keywords(dataset, set, train_word_names, test_image_names);

    auto result_folder = select_folder("./dtw_results/");

    generate_rel_files(result_folder, dataset, test_image_names, keywords);

    std::cout << "Evaluate performance..." << std::endl;

    std::size_t evaluated = 0;

    std::vector<double> eer(keywords.size());
    std::vector<double> ap(keywords.size());

    std::ofstream global_top_stream(result_folder + "/global_top_file");
    std::ofstream local_top_stream(result_folder + "/local_top_file");

    std::vector<std::vector<etl::dyn_vector<weight>>> test_features;

    for(std::size_t t = 0; t < test_image_names.size(); ++t){
        decltype(auto) test_image = test_image_names[t];

        test_features.push_back(standard_features(dataset.word_images.at(test_image)));
    }

#ifdef GLOBAL_MEAN_SCALING
    auto scale = global_mean_scaling(test_features, conf, training);
#endif

#ifdef GLOBAL_LINEAR_SCALING
    auto scale = global_linear_scaling(test_features, conf, training);
#endif

#if defined(GLOBAL_MEAN_SCALING) || defined(GLOBAL_LINEAR_SCALING)
    for(std::size_t t = 0; t < test_features.size(); ++t){
        for(std::size_t i = 0; i < test_features[t].size(); ++i){
            for(std::size_t f = 0; f < test_features.back().back().size(); ++f){
                test_features[t][i][f] = scale(test_features[t][i][f], conf.scale_a[f], conf.scale_b[f]);
            }
        }
    }
#else
    cpp_unused(training);
    cpp_unused(conf);
#endif

    cpp::default_thread_pool<> pool;

    for(std::size_t k = 0; k < keywords.size(); ++k){
        auto& keyword = keywords[k];

        std::string training_image;
        for(auto& labels : dataset.word_labels){
            if(keyword == labels.second && std::find(train_word_names.begin(), train_word_names.end(), labels.first) != train_word_names.end()){
                training_image = labels.first;
                break;
            }
        }

        //Make sure that there is a sample in the training set
        if(training_image.empty()){
            std::cout << "Skipped " << keyword << " since there are no example in the training set" << std::endl;
            continue;
        }

        ++evaluated;

        auto ref_a = standard_features(dataset.word_images.at(training_image + ".png"));

#if defined(GLOBAL_MEAN_SCALING) || defined(GLOBAL_LINEAR_SCALING)
        for(std::size_t i = 0; i < ref_a.size(); ++i){
            for(std::size_t f = 0; f < ref_a[i].size(); ++f){
                ref_a[i][f] = scale(ref_a[i][f], conf.scale_a[f], conf.scale_b[f]);
            }
        }
#endif

        std::vector<std::pair<std::string, weight>> diffs_a(test_image_names.size());

        cpp::parallel_foreach_i(pool, test_image_names.begin(), test_image_names.end(),
            [&](auto& test_image, std::size_t t){
                decltype(auto) test_a = test_features[t];

                double diff_a = dtw_distance(ref_a, test_a, true);
                diffs_a[t] = std::make_pair(std::string(test_image.begin(), test_image.end() - 4), diff_a);
            });

        update_stats(k, result_folder, dataset, keyword, diffs_a, eer, ap, global_top_stream, local_top_stream, test_image_names);
    }

    std::cout << "... done" << std::endl;

    std::cout << evaluated << " keywords evaluated" << std::endl;

    double mean_eer = std::accumulate(eer.begin(), eer.end(), 0.0) / eer.size();
    double mean_ap = std::accumulate(ap.begin(), ap.end(), 0.0) / ap.size();

    std::cout << "Mean EER: " << mean_eer << std::endl;
    std::cout << "Mean AP: " << mean_ap << std::endl;
}

} //end of anonymous namespace

void standard_train(
        const spot_dataset& dataset, const spot_dataset_set& set, config& conf,
        names train_word_names, names train_image_names, names valid_image_names, names test_image_names){
    std::cout << "Use method 0 (Standard Features + DTW)" << std::endl;

    std::cout << "Evaluate on training set" << std::endl;
    evaluate_dtw(dataset, set, conf, train_word_names, train_image_names, true);

    std::cout << "Evaluate on validation set" << std::endl;
    evaluate_dtw(dataset, set, conf, train_word_names, valid_image_names, true);

    std::cout << "Evaluate on test set" << std::endl;
    evaluate_dtw(dataset, set, conf, train_word_names, test_image_names, false);
}

void standard_features(
        const spot_dataset& dataset, const spot_dataset_set& set, config& conf,
        names train_word_names, names train_image_names, names valid_image_names, names test_image_names){
    std::cout << "Use method 0 (Standard Features + DTW)" << std::endl;

    //TODO
}
