//=======================================================================
// Copyright Baptiste Wicht 2015-2016.
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
#include "features.hpp"   //Features exporting
#include "evaluation.hpp" //Global evaluation functions

#define LOCAL_MEAN_SCALING
#include "scaling.hpp" //Scaling functions

namespace {

std::vector<etl::dyn_vector<weight>> standard_features(const config& conf, const cv::Mat& clean_image) {
    std::vector<etl::dyn_vector<weight>> features;

    const auto width  = static_cast<std::size_t>(clean_image.size().width);
    const auto height = static_cast<std::size_t>(clean_image.size().height);

    for (std::size_t i = 0; i < width; ++i) {
        double lower = 0.0;
        for (std::size_t y = 0; y < height; ++y) {
            if (clean_image.at<uint8_t>(y, i) == 0) {
                lower = y;
                break;
            }
        }

        double upper = 0.0;
        for (std::size_t y = height - 1; y > 0; --y) {
            if (clean_image.at<uint8_t>(y, i) == 0) {
                upper = y;
                break;
            }
        }

        std::size_t black = 0;
        for (std::size_t y = 0; y < height; ++y) {
            if (clean_image.at<uint8_t>(y, i) == 0) {
                ++black;
            }
        }

        std::size_t inner_black = 0;
        for (std::size_t y = lower; y < upper + 1; ++y) {
            if (clean_image.at<uint8_t>(y, i) == 0) {
                ++inner_black;
            }
        }

        std::size_t transitions = 0;
        for (std::size_t y = 1; y < height; ++y) {
            if (clean_image.at<uint8_t>(y - 1, i) == 0 && clean_image.at<uint8_t>(y, i) != 0) {
                ++transitions;
            }
        }

        double gravity = 0;
        double moment = 0;
        for (std::size_t y = 0; y < height; ++y) {
            auto pixel = clean_image.at<uint8_t>(y, i) == 0 ? 0.0 : 1.0;
            gravity += y * pixel;
            moment += y * y * pixel;
        }
        gravity /= height;
        moment /= (height * height);

        if(conf.method == Method::Standard){
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
        } else if(conf.method == Method::Manmatha){
            features.emplace_back(4);

            auto& f = features.back();

            f[0] = black; //Number of black pixels
            f[1] = upper;
            f[2] = lower;
            f[3] = transitions;
        }
    }

    if(conf.method == Method::Standard){
        for (std::size_t i = 0; i < width - 1; ++i) {
            features[i][5] = features[i + 1][1] - features[i][1];
            features[i][6] = features[i + 1][2] - features[i][2];
        }
    } else if (conf.method == Method::Manmatha){
        //Interpolate contour gaps

        //1. Fill the gap starting from column 0 (if any)

        if(features[0][1] == 0.0 && features[0][2] == 0.0){
            for (std::size_t i = 1; i < width; ++i) {
                if(!(features[i][1] == 0.0 && features[i][2] == 0.0)){
                    auto upper = features[i][1];
                    auto lower = features[i][2];

                    while(i-- > 0){
                        features[i][1] = upper;
                        features[i][2] = lower;
                    }

                    break;
                }
            }
        }

        //2. Fill the gap starting from the end (if any)

        if(features[width - 1][1] == 0.0 && features[width - 1][2] == 0.0){
            for (std::size_t i = width - 1; i > 0; --i) {
                if(!(features[i][1] == 0.0 && features[i][2] == 0.0)){
                    auto upper = features[i][1];
                    auto lower = features[i][2];

                    while(i++ < width - 1){
                        features[i][1] = upper;
                        features[i][2] = lower;
                    }

                    break;
                }
            }
        }

        //3. Fill the middle gaps

        for (std::size_t i = 1; i < width - 1; ++i) {
            if(features[i][1] == 0.0 && features[i][2] == 0.0){
                std::size_t end = i;
                for (std::size_t j = i; j < width; ++j) {
                    if(!(features[j][1] == 0.0 && features[j][2] == 0.0)){
                        end = j;
                        break;
                    }
                }

                auto upper_diff = features[end][1] - features[i - 1][1];
                auto lower_diff = features[end][2] - features[i - 1][2];

                auto step = 1.0 / (end - i + 1);

                for(std::size_t j = i; j < end; ++j){
                    features[j][1] = features[i - 1][1] + upper_diff * step * (j - i + 1);
                    features[j][2] = features[i - 1][2] + lower_diff * step * (j - i + 1);
                }
            }
        }
    }

#ifdef LOCAL_LINEAR_SCALING
    local_linear_feature_scaling(features);
#endif

#ifdef LOCAL_MEAN_SCALING
    local_mean_feature_scaling(features);
#endif

    return features;
}

void scale(std::vector<std::vector<etl::dyn_vector<weight>>>& test_features, config& conf, bool training) {
#ifdef GLOBAL_MEAN_SCALING
    auto scale = global_mean_scaling(test_features, conf, training);
#endif

#ifdef GLOBAL_LINEAR_SCALING
    auto scale = global_linear_scaling(test_features, conf, training);
#endif

#if defined(GLOBAL_MEAN_SCALING) || defined(GLOBAL_LINEAR_SCALING)
    for (std::size_t t = 0; t < test_features.size(); ++t) {
        for (std::size_t i = 0; i < test_features[t].size(); ++i) {
            for (std::size_t f = 0; f < test_features.back().back().size(); ++f) {
                test_features[t][i][f] = scale(test_features[t][i][f], conf.scale_a[f], conf.scale_b[f]);
            }
        }
    }
#else
    cpp_unused(test_features);
    cpp_unused(training);
    cpp_unused(conf);
#endif
}

template <typename Dataset>
std::vector<std::vector<etl::dyn_vector<weight>>> prepare_outputs(const Dataset& dataset, config& conf, names test_image_names, bool training){
    std::vector<std::vector<etl::dyn_vector<weight>>> test_features;

    for (auto& test_image : test_image_names) {
        test_features.push_back(standard_features(conf, dataset.word_images.at(test_image)));
    }

    scale(test_features, conf, training);

    return test_features;
}

template <typename Dataset>
std::vector<std::vector<etl::dyn_vector<weight>>> compute_reference(thread_pool& pool, const Dataset& dataset, const config& conf, names training_images) {
    std::vector<std::vector<etl::dyn_vector<weight>>> ref_a(training_images.size());

    cpp::parallel_foreach_i(pool, training_images.begin(), training_images.end(), [&](auto& training_image, std::size_t e) {
        ref_a[e] = standard_features(conf, dataset.word_images.at(training_image + ".png"));

#ifdef GLOBAL_MEAN_SCALING
        auto scale = global_mean_scaling(ref_a[e], conf, false);
#endif

#ifdef GLOBAL_LINEAR_SCALING
        auto scale = global_linear_scaling(ref_a[e], conf, false);
#endif

#if defined(GLOBAL_MEAN_SCALING) || defined(GLOBAL_LINEAR_SCALING)
        for (std::size_t i = 0; i < ref_a.size(); ++i) {
            for (std::size_t f = 0; f < ref_a[i].size(); ++f) {
                ref_a[e][i][f] = scale(ref_a[i][f], conf.scale_a[f], conf.scale_b[f]);
            }
        }
#endif
    });

    return ref_a;
}

template <typename Dataset, typename Set>
void evaluate_dtw(const Dataset& dataset, const Set& set, config& conf, names train_word_names, names test_image_names, bool training) {
    thread_pool pool;

    parameters parameters;
    parameters.sc_band = 0.11;

    // 0. Select the keywords

    auto keywords = select_keywords(dataset, set, train_word_names, test_image_names);

    // 1. Select a folder

    auto result_folder = select_folder("./dtw_results/");

    // 2. Generate the rel files

    generate_rel_files(result_folder, dataset, test_image_names, keywords);

    // 3. Prepare all the outputs

    auto test_features = prepare_outputs(dataset, conf, test_image_names, training);

    // 4. Evaluate the performances

    std::cout << "Evaluate performance..." << std::endl;

    std::vector<double> eer(keywords.size());
    std::vector<double> ap(keywords.size());

    std::ofstream global_top_stream(result_folder + "/global_top_file");
    std::ofstream local_top_stream(result_folder + "/local_top_file");

    for (std::size_t k = 0; k < keywords.size(); ++k) {
        auto& keyword = keywords[k];

        // a) Select the training images

        auto training_images = select_training_images(dataset, keyword, train_word_names);

        // b) Compute the reference features

        auto ref = compute_reference(pool, dataset, conf, training_images);

        // c) Compute the distances

        auto diffs = compute_distances(pool, dataset, test_features, ref, training_images, test_image_names, parameters);

        // d) Update the local stats

        update_stats(k, result_folder, dataset, keyword, diffs, eer, ap, global_top_stream, local_top_stream, test_image_names);
    }

    std::cout << "... done" << std::endl;

    // 5. Finalize the results

    std::cout << keywords.size() << " keywords evaluated" << std::endl;

    double mean_eer = std::accumulate(eer.begin(), eer.end(), 0.0) / eer.size();
    double mean_ap  = std::accumulate(ap.begin(), ap.end(), 0.0) / ap.size();

    std::cout << "Mean EER: " << mean_eer << std::endl;
    std::cout << "Mean AP: " << mean_ap << std::endl;
}

template <typename Dataset>
void extract_features(const Dataset& dataset, config& conf, const std::vector<std::string>& test_image_names, bool training) {
    std::cout << "Extract features ..." << std::endl;

    std::vector<std::vector<etl::dyn_vector<weight>>> test_features;

    for (auto& test_image : test_image_names) {
        test_features.push_back(standard_features(conf, dataset.word_images.at(test_image)));
    }

    scale(test_features, conf, training);

    export_features(conf, test_image_names, test_features, ".0");

    std::cout << "... done" << std::endl;
}

} //end of anonymous namespace

void standard_train(
    const spot_dataset& dataset, const spot_dataset_set& set, config& conf,
    names train_word_names, names train_image_names, names valid_image_names, names test_image_names) {

    std::cout << "Evaluate on training set" << std::endl;
    evaluate_dtw(dataset, set, conf, train_word_names, train_image_names, true);

    std::cout << "Evaluate on validation set" << std::endl;
    evaluate_dtw(dataset, set, conf, train_word_names, valid_image_names, true);

    std::cout << "Evaluate on test set" << std::endl;
    evaluate_dtw(dataset, set, conf, train_word_names, test_image_names, false);
}

void standard_features(
    const spot_dataset& dataset, const spot_dataset_set& /*set*/, config& conf,
    names /*train_word_names*/, names train_image_names, names valid_image_names, names test_image_names) {

    std::cout << "Extract features on training set" << std::endl;
    extract_features(dataset, conf, train_image_names, true);

    std::cout << "Extract features on validation set" << std::endl;
    extract_features(dataset, conf, valid_image_names, false);

    std::cout << "Extract features on test set" << std::endl;
    extract_features(dataset, conf, test_image_names, false);
}
