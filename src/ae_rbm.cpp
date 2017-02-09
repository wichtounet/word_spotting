//=======================================================================
// Copyright Baptiste Wicht 2015-2016.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <string>

#include "memory.hpp" //First for debug reasons

#include "cpp_utils/parallel.hpp"

#include "dll/rbm/rbm.hpp"
#include "dll/dbn.hpp"

#include "ae_rbm.hpp"
#include "config.hpp"
#include "standard.hpp"
#include "utils.hpp"
#include "reports.hpp"
#include "features.hpp"   //Features exporting
#include "evaluation.hpp" //evaluation utilities

//#define LOCAL_FRAME_NORMALIZATION
//#define LOCAL_L2_NORMALIZATION
//#define GLOBAL_FRAME_NORMALIZATION
#define GLOBAL_L2_NORMALIZATION

//#define LOCAL_LINEAR_SCALING
#define LOCAL_MEAN_SCALING
#include "scaling.hpp" //Scaling functions

namespace {

template <size_t L, typename DBN>
using dbn_output_t = decltype(std::declval<DBN>().template prepare_output<L, typename DBN::input_one_t>());

template <size_t L, typename DBN>
using features_t = std::vector<std::vector<dbn_output_t<L, DBN>>>;

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

template <size_t L, typename Input, typename DBN>
features_t<L, DBN> prepare_outputs_ae(
    thread_pool& pool, const spot_dataset& dataset, const DBN& dbn, const config& conf,
    names test_image_names, bool training) {

    features_t<L, DBN> test_features_a(test_image_names.size());

    std::cout << "Prepare the outputs ..." << std::endl;

    auto feature_extractor = [&](auto& test_image, std::size_t i) {
        auto& vec = test_features_a[i];

        //Get features from DBN
        auto patches = mat_to_patches_t<Input>(conf, dataset.word_images.at(test_image), training);

        vec.reserve(patches.size());

        for(auto& patch : patches){
            vec.push_back(dbn.template prepare_output<L, Input>());
            vec.back() = dbn.template features_sub<L>(patch);

        }

        normalize_feature_vector(vec);
    };

    cpp::parallel_foreach_i(pool, test_image_names.begin(), test_image_names.end(), feature_extractor);

    normalize_features(conf, training, test_features_a);

    std::cout << "... done" << std::endl;

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

        normalize_feature_vector(vec);
    };

    cpp::parallel_foreach_i(pool, training_images.begin(), training_images.end(), feature_extractor);

    normalize_features(conf, false, ref_a);

    return ref_a;
}

template <size_t L, typename Input, typename Set, typename DBN>
std::string evaluate_patches_ae(const spot_dataset& dataset, const Set& set, config& conf, const DBN& dbn, names train_word_names, names test_image_names, bool training, parameters parameters) {
    thread_pool pool;

    // 0. Select the keywords

    auto keywords = select_keywords(dataset, set, train_word_names, test_image_names);

    // 1. Select a folder

    auto result_folder = select_folder("./results/");

    // 2. Generate the rel files

    generate_rel_files(result_folder, dataset, test_image_names, keywords);

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

    std::cout << keywords.size() << " keywords evaluated" << std::endl;

    double mean_eer = std::accumulate(eer.begin(), eer.end(), 0.0) / eer.size();
    double mean_ap  = std::accumulate(ap.begin(), ap.end(), 0.0) / ap.size();

    std::cout << "Mean EER: " << mean_eer << std::endl;
    std::cout << "Mean AP: " << mean_ap << std::endl;

    return result_folder;
}

constexpr const auto patch_width  = 20;
constexpr const auto patch_height = 40;
constexpr size_t batch_size       = 128;
constexpr size_t epochs           = 10;
constexpr const auto train_stride = 1;
constexpr const auto test_stride  = 1;

using image_t = etl::fast_dyn_matrix<float, 1, patch_height, patch_width>;

template<size_t N>
void rbm_evaluate(const spot_dataset& dataset, const spot_dataset_set& set, config& conf, names train_word_names, names test_image_names, parameters params, const std::vector<image_t>& training_patches) {
    using network_t = typename dll::dbn_desc<
        dll::dbn_layers<
            typename dll::rbm_desc<
                patch_height * patch_width, N,
                dll::batch_size<batch_size>,
                dll::momentum
        >::layer_t
    >>::dbn_t;

    auto net = std::make_unique<network_t>();

    net->display();

    // Configure the network
    net->template layer_get<0>().learning_rate    = 0.1;
    net->template layer_get<0>().initial_momentum = 0.9;
    net->template layer_get<0>().momentum         = 0.9;

    // Train as RBM
    net->pretrain(training_patches, epochs);

    auto folder = evaluate_patches_ae<0, image_t>(dataset, set, conf, *net, train_word_names, test_image_names, false, params);
    std::cout << "AE-Result: RBM(" << N << "):" << folder << std::endl;
}

} // end of anonymous namespace

void rbm_evaluate_all(const spot_dataset& dataset, const spot_dataset_set& set, config& conf, names train_word_names, names test_image_names, parameters params, const std::vector<image_t>& training_patches){
    if (conf.rbm) {
        rbm_evaluate<10>(dataset, set, conf, train_word_names, test_image_names, params, training_patches);
        rbm_evaluate<50>(dataset, set, conf, train_word_names, test_image_names, params, training_patches);
        rbm_evaluate<100>(dataset, set, conf, train_word_names, test_image_names, params, training_patches);
        rbm_evaluate<200>(dataset, set, conf, train_word_names, test_image_names, params, training_patches);
        rbm_evaluate<300>(dataset, set, conf, train_word_names, test_image_names, params, training_patches);
        rbm_evaluate<400>(dataset, set, conf, train_word_names, test_image_names, params, training_patches);
        rbm_evaluate<500>(dataset, set, conf, train_word_names, test_image_names, params, training_patches);
    }
}
