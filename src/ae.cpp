//=======================================================================
// Copyright Baptiste Wicht 2015-2016.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <string>

#include "memory.hpp" //First for debug reasons

#include "etl/etl.hpp"

#include "cpp_utils/parallel.hpp"

#include "dll/rbm/conv_rbm.hpp"
#include "dll/rbm/conv_rbm_mp.hpp"
#include "dll/pooling/mp_layer.hpp"
#include "dll/dbn.hpp"

#include "ae.hpp"
#include "config.hpp"
#include "standard.hpp"
#include "utils.hpp"
#include "reports.hpp"
#include "dtw.hpp"        //Dynamic time warping
#include "features.hpp"   //Features exporting
#include "evaluation.hpp" //evaluation utilities

//#define LOCAL_FRAME_NORMALIZATION
//#define LOCAL_L2_NORMALIZATION
//#define GLOBAL_FRAME_NORMALIZATION
#define GLOBAL_L2_NORMALIZATION

//#define LOCAL_LINEAR_SCALING
#define LOCAL_MEAN_SCALING
#include "scaling.hpp" //Scaling functions

//The different configurations
#include "patches_config.hpp"

namespace {

void log_scaling(){
#ifdef LOCAL_FRAME_NORMALIZATION
    std::cout << "Local Frame Normalization" << std::endl;
#endif

#ifdef LOCAL_L2_NORMALIZATION
    std::cout << "Local L2 Normalization" << std::endl;
#endif

#ifdef GLOBAL_FRAME_NORMALIZATION
    std::cout << "Global Frame Normalization" << std::endl;
#endif

#ifdef GLOBAL_L2_NORMALIZATION
    std::cout << "Global L2 Normalization" << std::endl;
#endif

#ifdef LOCAL_LINEAR_SCALING
    std::cout << "Local Linear Scaling" << std::endl;
#endif

#ifdef LOCAL_MEAN_SCALING
    std::cout << "Local Mean Scaling" << std::endl;
#endif
}

template <bool DBN_Patch, typename DBN>
using dbn_output_t = decltype(std::declval<DBN>().template prepare_one_output<typename DBN::input_one_t>());

template <bool DBN_Patch, typename DBN>
using features_t = std::vector<std::vector<dbn_output_t<DBN_Patch, DBN>>>;

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

template <bool DBN_Patch, typename DBN>
features_t<DBN_Patch, DBN> prepare_outputs(
    thread_pool& pool, const spot_dataset& dataset, const DBN& dbn, const config& conf,
    names test_image_names, bool training, bool runtime = false) {

    features_t<DBN_Patch, DBN> test_features_a(test_image_names.size());

    if(!runtime){
        std::cout << "Prepare the outputs ..." << std::endl;
    }

    auto feature_extractor = [&](auto& test_image, std::size_t i) {
        auto& vec = test_features_a[i];

        //Get features from DBN
        cpp::static_if<DBN_Patch>([&](auto f) {
            auto image = mat_for_patches(conf, dataset.word_images.at(test_image));
            vec = f(dbn).activation_probabilities(image);
        }).else_([&](auto f) {
            auto patches = mat_to_patches<DBN>(conf, dataset.word_images.at(test_image), training);

            vec.reserve(patches.size());

            for(auto& patch : patches){
                f(vec).push_back(f(dbn).template prepare_one_output<typename DBN::input_t>());
                vec.back() = f(dbn).activation_probabilities(patch);

            }
        });

        normalize_feature_vector(vec);
    };

    if(!runtime){
        cpp::parallel_foreach_i(pool, test_image_names.begin(), test_image_names.end(), feature_extractor);
    } else {
        cpp::foreach_i(test_image_names.begin(), test_image_names.end(), feature_extractor);
    }

    normalize_features(conf, training, test_features_a);

    if(!runtime){
        std::cout << "... done" << std::endl;
    }

    return test_features_a;
}

template <bool DBN_Patch, typename DBN>
features_t<DBN_Patch, DBN> compute_reference(
    thread_pool& pool, const spot_dataset& dataset, const DBN& dbn, const config& conf,
    names training_images) {

    features_t<DBN_Patch, DBN> ref_a(training_images.size());

    auto feature_extractor = [&](auto& test_image, std::size_t i) {
        auto& vec = ref_a[i];

        //Get features from DBN
        cpp::static_if<DBN_Patch>([&](auto f) {
            auto image = mat_for_patches(conf, dataset.word_images.at(test_image + ".png"));
            vec = f(dbn).activation_probabilities(image);
        }).else_([&](auto f) {
            auto patches = mat_to_patches<DBN>(conf, dataset.word_images.at(test_image + ".png"), false);

            vec.reserve(patches.size());

            for(auto& patch : patches){
                f(vec).push_back(f(dbn).template prepare_one_output<typename DBN::input_t>());
                vec.back() = f(dbn).activation_probabilities(patch);

            }
        });

        normalize_feature_vector(vec);
    };

    cpp::parallel_foreach_i(pool, training_images.begin(), training_images.end(), feature_extractor);

    normalize_features(conf, false, ref_a);

    return ref_a;
}

template <bool DBN_Patch, typename Set, typename DBN>
std::string evaluate_patches(const spot_dataset& dataset, const Set& set, config& conf, const DBN& dbn, names train_word_names, names test_image_names, bool training, parameters parameters, bool features, bool runtime = false) {
    thread_pool pool;

    if (features) {
        auto test_features_a = prepare_outputs<DBN_Patch>(pool, dataset, dbn, conf, test_image_names, training, runtime);

        if(!runtime){
            export_features(conf, test_image_names, test_features_a, ".2");
        }

        return {};
    } else {
        // 0. Select the keywords

        auto keywords = select_keywords(dataset, set, train_word_names, test_image_names);

        // 1. Select a folder

        auto result_folder = select_folder("./results/");

        // 2. Generate the rel files

        generate_rel_files(result_folder, dataset, test_image_names, keywords);

        // 3. Prepare all the outputs

        auto test_features_a = prepare_outputs<DBN_Patch>(pool, dataset, dbn, conf, test_image_names, training, runtime);

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

            auto ref_a = compute_reference<DBN_Patch>(pool, dataset, dbn, conf, training_images);

            // c) Compute the distances

            auto diffs_a = compute_distances(conf, pool, dataset, test_features_a, ref_a, training_images,
                test_image_names, train_word_names,
                parameters, [&](names train_names){ return compute_reference<DBN_Patch>(pool, dataset, dbn, conf, train_names);});

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
}

} // end of anonymous namespace

void ae_train(
    const spot_dataset& dataset, const spot_dataset_set& set, config& conf,
    names train_word_names, names train_image_names, names test_image_names, bool features, bool runtime) {

    auto pretraining_image_names = train_image_names;

    log_scaling();

            std::cout << "Use a third of the resolution" << std::endl;

        copy_from_namespace(third);

        using cdbn_t =
            dll::dbn_desc<
                dll::dbn_layers<
                    dll::conv_rbm_desc<
                        1, NV1_1, NV1_2, K1, NH1_1, NH1_2, dll::weight_type<weight>, dll::batch_size<third::B1>, dll::momentum, dll::weight_decay<third::DT1>, dll::visible<third::VT1>, dll::hidden<third::HT1>, dll::sparsity<third::SM1>, dll::shuffle_cond<shuffle_1>, dll::clipping_cond<clipping_1>, dll::dbn_only>::layer_t,
                    dll::mp_layer_3d_desc<K1, NH1_1, NH1_2, 1, C1, C1, dll::weight_type<weight>>::layer_t,
                    dll::conv_rbm_desc<
                        K1, NV2_1, NV2_2, K2, NH2_1, NH2_2, dll::weight_type<weight>, dll::batch_size<third::B2>, dll::momentum, dll::weight_decay<third::DT2>, dll::visible<third::VT1>, dll::hidden<third::HT2>, dll::sparsity<third::SM2>, dll::shuffle_cond<shuffle_2>, dll::clipping_cond<clipping_2>, dll::dbn_only>::layer_t,
                    dll::mp_layer_3d_desc<K2, NH2_1, NH2_2, 1, C2, C2, dll::weight_type<weight>>::layer_t>
                , dll::batch_mode>::dbn_t;

        //Max pooling layers models have more layers
        constexpr const std::size_t L1 = 0;
        constexpr const std::size_t L2 = 2;
        constexpr const std::size_t L3 = 4;

        auto cdbn = std::make_unique<cdbn_t>();

        auto cdbn_train = std::make_unique<cdbn_t>();
        auto& cdbn_ref = cdbn_train;

        // Level 1
        third::rate_0(cdbn_ref->template layer_get<L1>().learning_rate);
        third::momentum_0(cdbn_ref->template layer_get<L1>().initial_momentum, cdbn_ref->template layer_get<L1>().final_momentum);
        third::wd_l1_0(cdbn_ref->template layer_get<L1>().l1_weight_cost);
        third::wd_l2_0(cdbn_ref->template layer_get<L1>().l2_weight_cost);
        third::pbias_0(cdbn_ref->template layer_get<L1>().pbias);
        third::pbias_lambda_0(cdbn_ref->template layer_get<L1>().pbias_lambda);
        third::sparsity_target_0(cdbn_ref->template layer_get<L1>().sparsity_target);
        third::clip_norm_1(cdbn_ref->template layer_get<L1>().gradient_clip);

        //Level 2
        third::rate_1(cdbn_ref->template layer_get<L2>().learning_rate);
        third::momentum_1(cdbn_ref->template layer_get<L2>().initial_momentum, cdbn_ref->template layer_get<L2>().final_momentum);
        third::wd_l1_1(cdbn_ref->template layer_get<L2>().l1_weight_cost);
        third::wd_l2_1(cdbn_ref->template layer_get<L2>().l2_weight_cost);
        third::pbias_1(cdbn_ref->template layer_get<L2>().pbias);
        third::pbias_lambda_1(cdbn_ref->template layer_get<L2>().pbias_lambda);
        third::sparsity_target_1(cdbn_ref->template layer_get<L2>().sparsity_target);
        third::clip_norm_2(cdbn_ref->template layer_get<L2>().gradient_clip);

        cdbn_ref->display();
        std::cout << cdbn->output_size() << " output features" << std::endl;

        constexpr const auto patch_width  = third::patch_width;
        constexpr const auto patch_height = third::patch_height;
        constexpr const auto train_stride = third::train_stride;
        constexpr const auto test_stride  = third::test_stride;

        //Pass information to the next passes (evaluation)
        conf.patch_width  = patch_width;
        conf.train_stride = train_stride;
        conf.test_stride  = test_stride;

        static constexpr const bool DBN_Patch = false;

            //Train the DBN
                std::vector<cdbn_t::template layer_type<0>::input_one_t> training_patches;
                training_patches.reserve(pretraining_image_names.size() * 10);

                std::cout << "Generate patches ..." << std::endl;

                for (auto& name : pretraining_image_names) {
                    decltype(auto) image = dataset.word_images.at(name);

                    // Insert the patches from the original image
                    auto patches = mat_to_patches<cdbn_t>(conf, image, true);
                    std::copy(patches.begin(), patches.end(), std::back_inserter(training_patches));
                }

                std::cout << "... " << training_patches.size() << " patches extracted" << std::endl;

                cdbn->pretrain(training_patches, third::epochs);

                std::cout << "Switch to optimal parameters" << std::endl;
                parameters params;
                params.sc_band = 0.05;
                std::cout << "\tsc_band: " << params.sc_band << std::endl;

            std::cout << "Evaluate on test set" << std::endl;
            auto folder = evaluate_patches<DBN_Patch>(dataset, set, conf, *cdbn, train_word_names, test_image_names, false, params, features, runtime);

        silence_l3_warnings();
}
