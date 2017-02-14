//=======================================================================
// Copyright Baptiste Wicht 2015-2017.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "memory.hpp" //First for debug reasons

#include "cpp_utils/parallel.hpp"

#include <iostream>
#include <string>

#include "evaluation.hpp"    //Evaluation utilities
#include "features.hpp"      //Features exporting
#include "normalization.hpp" //Normalization functions
#include "reports.hpp"
#include "reports.hpp"
#include "standard.hpp"
#include "utils.hpp"

namespace spot {

template <size_t L, typename DBN>
using dbn_output_t = decltype(std::declval<DBN>().template prepare_output<L, typename DBN::input_one_t>());

template <size_t L, typename DBN>
using features_t = std::vector<std::vector<dbn_output_t<L, DBN>>>;

template <size_t L, typename Input, typename DBN>
features_t<L, DBN> prepare_outputs_ae(
    thread_pool& pool, const spot_dataset& dataset, const DBN& dbn, const config& conf,
    names test_image_names, bool training, bool normalize = true) {

    features_t<L, DBN> test_features_a(test_image_names.size());

    auto feature_extractor = [&](auto& test_image, std::size_t i) {
        auto& vec = test_features_a[i];

        //Get features from DBN
        auto patches = mat_to_patches_t<Input>(conf, dataset.word_images.at(test_image), training);

        vec.reserve(patches.size());

        for(auto& patch : patches){
            vec.push_back(dbn.template prepare_output<L, Input>());
            vec.back() = dbn.template features_sub<L>(patch);

        }

        if(normalize){
            spot::normalize_feature_vector(vec);
        }
    };

    cpp::parallel_foreach_i(pool, test_image_names.begin(), test_image_names.end(), feature_extractor);

    if (normalize) {
        spot::normalize_features(conf, training, test_features_a);
    }

    return test_features_a;
}

template <size_t L, typename Input, typename DBN>
features_t<L, DBN> compute_reference_ae(
    thread_pool& pool, const spot_dataset& dataset, const DBN& dbn, const config& conf,
    names training_images) {

    features_t<L, DBN> ref_a(training_images.size());

    auto feature_extractor = [&](auto& test_image, std::size_t i) {
        auto& vec = ref_a[i];

        //Get features from DBN
        auto patches = mat_to_patches_t<Input>(conf, dataset.word_images.at(test_image + ".png"), false);

        vec.reserve(patches.size());

        for(auto& patch : patches){
            vec.push_back(dbn.template prepare_output<L, Input>());
            vec.back() = dbn.template features_sub<L>(patch);

        }

        spot::normalize_feature_vector(vec);
    };

    cpp::parallel_foreach_i(pool, training_images.begin(), training_images.end(), feature_extractor);

    spot::normalize_features(conf, false, ref_a);

    return ref_a;
}

template <size_t L, typename Input, typename Set, typename DBN>
std::string evaluate_patches_ae(const spot_dataset& dataset, const Set& set, config& conf, const DBN& dbn, names train_word_names, names test_image_names, bool training, parameters parameters) {
    thread_pool pool;

    // 0. Select the keywords

    auto keywords = select_keywords(dataset, set, train_word_names, test_image_names, false);

    // 1. Select a folder

    auto result_folder = select_folder("./results/", false);

    // 2. Generate the rel files

    generate_rel_files(result_folder, dataset, test_image_names, keywords, false);

    // 3. Prepare all the outputs

    auto test_features_a = prepare_outputs_ae<L, Input>(pool, dataset, dbn, conf, test_image_names, training);

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

        auto ref_a = compute_reference_ae<L, Input>(pool, dataset, dbn, conf, training_images);

        // c) Compute the distances

        auto diffs_a = compute_distances(conf, pool, dataset, test_features_a, ref_a, training_images,
            test_image_names, train_word_names,
            parameters, [&](names train_names){ return compute_reference_ae<L, Input>(pool, dataset, dbn, conf, train_names);});

        // d) Update the local stats

        update_stats(k, result_folder, dataset, keyword, diffs_a, eer, ap, global_top_stream, local_top_stream, test_image_names);

        if((k + 1) % (keywords.size() / 10) == 0){
            std::cout << ((k + 1) / (keywords.size() / 10)) * 10 << "%" << std::endl;
        }
    }

    std::cout << "... done" << std::endl;

    // 5. Finalize the results

    double mean_eer = std::accumulate(eer.begin(), eer.end(), 0.0) / eer.size();
    double mean_ap  = std::accumulate(ap.begin(), ap.end(), 0.0) / ap.size();

    std::cout << "Mean EER: " << mean_eer << std::endl;
    std::cout << "Mean AP: " << mean_ap << std::endl;

    return result_folder;
}

template <size_t L, typename Input, typename DBN1, typename DBN2>
features_t<L, DBN2> prepare_outputs_ae_stacked_2(
    thread_pool& pool, const spot_dataset& dataset, const DBN1& dbn1, const DBN2& dbn2, const config& conf,
    names test_image_names, bool training) {

    features_t<L, DBN1> test_features_a(test_image_names.size());
    features_t<L, DBN2> test_features_b(test_image_names.size());

    auto feature_extractor = [&](auto& test_image, std::size_t i) {
        auto& vec_a = test_features_a[i];
        auto& vec_b = test_features_b[i];

        //Get features from DBN
        auto patches = mat_to_patches_t<Input>(conf, dataset.word_images.at(test_image), training);

        vec_a.reserve(patches.size());
        vec_b.reserve(patches.size());

        for(auto& patch : patches){
            vec_a.push_back(dbn1.template prepare_output<0, Input>());
            vec_a.back() = dbn1.template features_sub<0>(patch);

            vec_b.push_back(dbn2.template prepare_output<0, decltype(vec_a.back())>());
            vec_b.back() = dbn2.template features_sub<0>(vec_a.back());
        }

        spot::normalize_feature_vector(vec_b);
    };

    cpp::parallel_foreach_i(pool, test_image_names.begin(), test_image_names.end(), feature_extractor);

    spot::normalize_features(conf, training, test_features_b);

    return test_features_b;
}

template <size_t L, typename Input, typename DBN1, typename DBN2>
features_t<L, DBN2> compute_reference_ae_stacked_2(
    thread_pool& pool, const spot_dataset& dataset, const DBN1& dbn1, const DBN2& dbn2, const config& conf,
    names training_images) {

    features_t<L, DBN1> ref_a(training_images.size());
    features_t<L, DBN2> ref_b(training_images.size());

    auto feature_extractor = [&](auto& test_image, std::size_t i) {
        auto& vec_a = ref_a[i];
        auto& vec_b = ref_b[i];

        //Get features from DBN
        auto patches = mat_to_patches_t<Input>(conf, dataset.word_images.at(test_image + ".png"), false);

        vec_a.reserve(patches.size());
        vec_b.reserve(patches.size());

        for(auto& patch : patches){
            vec_a.push_back(dbn1.template prepare_output<L, Input>());
            vec_a.back() = dbn1.template features_sub<L>(patch);

            vec_b.push_back(dbn2.template prepare_output<L, decltype(vec_a.back())>());
            vec_b.back() = dbn2.template features_sub<L>(vec_a.back());
        }

        spot::normalize_feature_vector(vec_b);
    };

    cpp::parallel_foreach_i(pool, training_images.begin(), training_images.end(), feature_extractor);

    spot::normalize_features(conf, false, ref_b);

    return ref_b;
}

template <size_t L, typename Input, typename Set, typename DBN1, typename DBN2>
std::string evaluate_patches_ae_stacked_2(const spot_dataset& dataset, const Set& set, config& conf, const DBN1& dbn1, const DBN2& dbn2, names train_word_names, names test_image_names, bool training, parameters parameters) {
    thread_pool pool;

    // 0. Select the keywords

    auto keywords = select_keywords(dataset, set, train_word_names, test_image_names, false);

    // 1. Select a folder

    auto result_folder = select_folder("./results/", false);

    // 2. Generate the rel files

    generate_rel_files(result_folder, dataset, test_image_names, keywords, false);

    // 3. Prepare all the outputs

    auto test_features_a = prepare_outputs_ae_stacked_2<L, Input>(pool, dataset, dbn1, dbn2, conf, test_image_names, training);

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

        auto ref_a = compute_reference_ae_stacked_2<L, Input>(pool, dataset, dbn1, dbn2, conf, training_images);

        // c) Compute the distances

        auto diffs_a = compute_distances(conf, pool, dataset, test_features_a, ref_a, training_images,
            test_image_names, train_word_names,
            parameters, [&](names train_names){ return compute_reference_ae_stacked_2<L, Input>(pool, dataset, dbn1, dbn2, conf, train_names);});

        // d) Update the local stats

        update_stats(k, result_folder, dataset, keyword, diffs_a, eer, ap, global_top_stream, local_top_stream, test_image_names);

        if((k + 1) % (keywords.size() / 10) == 0){
            std::cout << ((k + 1) / (keywords.size() / 10)) * 10 << "%" << std::endl;
        }
    }

    std::cout << "... done" << std::endl;

    // 5. Finalize the results

    double mean_eer = std::accumulate(eer.begin(), eer.end(), 0.0) / eer.size();
    double mean_ap  = std::accumulate(ap.begin(), ap.end(), 0.0) / ap.size();

    std::cout << "Mean EER: " << mean_eer << std::endl;
    std::cout << "Mean AP: " << mean_ap << std::endl;

    return result_folder;
}

} // end of namespace spot
