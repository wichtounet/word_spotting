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
#include "dtw.hpp"      //Dynamic time warping
#include "features.hpp" //Features exporting

#define LOCAL_MEAN_SCALING
#include "scaling.hpp" //Scaling functions

#include "memory.hpp"            //First for debug reasons
#include "evaluate_features.hpp" //The header of this source file
#include "evaluation.hpp"        //evaluation utilities
#include "patches_config.hpp"    //The different configurations

#define SUFFIX .
#define SUFFIX_CAT_I(A, B) A ## B
#define SUFFIX_CAT(A, B) SUFFIX_CAT_I(A, B)
#define STRINGIFY_I(A) #A
#define STRINGIFY(A) STRINGIFY_I(A)

namespace {

std::string get_suffix(config& conf){
    switch (conf.method) {
        case Method::Bunke2001:
        case Method::Rath2007:
        case Method::Rath2003:
        case Method::Rodriguez2008:
        case Method::Vinciarelli2004:
        case Method::Terasawa2009:
            return ".0";
        case Method::Holistic:
            return ".1";
        case Method::Patches:
            if(conf.half){
                return STRINGIFY(SUFFIX_CAT(SUFFIX, HALF_LEVELS));
            } else if(conf.third){
                return STRINGIFY(SUFFIX_CAT(SUFFIX, THIRD_LEVELS));
            } else {
                return STRINGIFY(SUFFIX_CAT(SUFFIX, FULL_LEVELS));
            }
    }

    return ".invalid";
}

std::vector<etl::dyn_vector<weight>> load_features(config& conf, const std::string& image){
    auto file_path = conf.data_full_path + image + get_suffix(conf);
    std::ifstream file_stream(file_path);

    std::vector<etl::dyn_vector<weight>> features;

    while(file_stream.good()){
        std::vector<weight> patch;

        std::string line;
        std::getline(file_stream, line);

        if(line.size() > 3){
            std::stringstream line_stream(line);

            while(line_stream.good()){
                double value;
                line_stream >> value;

                patch.push_back(value);

                char sep;
                line_stream >> sep;
            }

            features.emplace_back(patch.size());

            for(std::size_t i = 0; i < patch.size(); ++i){
                features.back()[i] = patch[i];
            }
        } else {
            break;
        }
    }

    return features;
}

template <typename Dataset, typename Set>
void evaluate_features(const Dataset& dataset, const Set& set, config& conf, const std::vector<std::string>& train_word_names, const std::vector<std::string>& test_image_names) {
    auto keywords = select_keywords(dataset, set, train_word_names, test_image_names);

    auto result_folder = select_folder("./dtw_results/");

    generate_rel_files(result_folder, dataset, test_image_names, keywords);

    std::cout << "Evaluate performance..." << std::endl;

    std::vector<double> eer(keywords.size());
    std::vector<double> ap(keywords.size());

    std::ofstream global_top_stream(result_folder + "/global_top_file");
    std::ofstream local_top_stream(result_folder + "/local_top_file");

    std::vector<std::vector<etl::dyn_vector<weight>>> test_features;

    for (auto& test_image : test_image_names) {
        test_features.push_back(load_features(conf, test_image));
    }

    const double sc_band = 0.05;

    cpp::default_thread_pool<> pool;

    for (std::size_t k = 0; k < keywords.size(); ++k) {
        auto& keyword = keywords[k];

        // a) Select the training images

        auto training_images = select_training_images(dataset, keyword, train_word_names);

        // b) Load the references features
        std::vector<std::vector<etl::dyn_vector<weight>>> ref_a;
        for (auto& training_image : training_images) {
            ref_a.push_back(load_features(conf, training_image + ".png"));
        }

        // c) Compute the distances

        std::vector<std::pair<std::string, weight>> diffs_a(test_image_names.size());

        cpp::parallel_foreach_i(pool, test_image_names.begin(), test_image_names.end(),
                                [&](auto& test_image, std::size_t t) {
                                    auto t_size = dataset.word_images.at(test_image).size().width;

                                    double best_diff_a = 100000000.0;

                                    for (std::size_t i = 0; i < ref_a.size(); ++i) {
                                        auto ref_size = dataset.word_images.at(training_images[i] + ".png").size().width;

                                        double diff_a;
                                        auto ratio = static_cast<double>(ref_size) / t_size;
                                        if (ratio > 2.0 || ratio < 0.5) {
                                            diff_a = 100000000.0;
                                        } else {
                                            diff_a = dtw_distance(ref_a[i], test_features[t], true, sc_band);
                                        }

                                        best_diff_a = std::min(best_diff_a, diff_a);
                                    }

                                    diffs_a[t] = std::make_pair(std::string(test_image.begin(), test_image.end() - 4), best_diff_a);
                                });

        // d) Update the local stats

        update_stats(k, result_folder, dataset, keyword, diffs_a, eer, ap, global_top_stream, local_top_stream, test_image_names);
    }

    std::cout << "... done" << std::endl;

    std::cout << keywords.size() << " keywords evaluated" << std::endl;

    double mean_eer = std::accumulate(eer.begin(), eer.end(), 0.0) / eer.size();
    double mean_ap  = std::accumulate(ap.begin(), ap.end(), 0.0) / ap.size();

    std::cout << "Mean EER: " << mean_eer << std::endl;
    std::cout << "Mean AP: " << mean_ap << std::endl;
}

} // end of anonymous namespace


void evaluate_features(const spot_dataset& dataset, const spot_dataset_set& set, config& conf,
                      names train_word_names,
                      names train_image_names, names valid_image_names, names test_image_names){
    std::cout << "Evaluate on training set" << std::endl;
    evaluate_features(dataset, set, conf, train_word_names, train_image_names);

    std::cout << "Evaluate on validation set" << std::endl;
    evaluate_features(dataset, set, conf, train_word_names, valid_image_names);

    std::cout << "Evaluate on test set" << std::endl;
    evaluate_features(dataset, set, conf, train_word_names, test_image_names);
}
