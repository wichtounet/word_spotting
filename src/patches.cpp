//=======================================================================
// Copyright Baptiste Wicht 2015-2017.
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
#include "dll/pooling/avgp_layer.hpp"
#include "dll/pooling/mp_layer.hpp"
#include "dll/transform/lcn_layer.hpp"
#include "dll/dbn.hpp"

#ifndef OPENCV_23
#include "dll/ocv_visualizer.hpp"
#endif

#include "nice_svm.hpp"

#include "patches.hpp"
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
#include "normalization.hpp" //Normalization functions

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

// TODO This should probably be reviewed, a lot!

template <typename DBN>
struct patch_iterator : std::iterator<std::input_iterator_tag, typename DBN::template layer_type<0>::input_one_t> {
    using value_t = typename DBN::template layer_type<0>::input_one_t;

    config& conf;
    const spot_dataset& dataset;
    names image_names;

    std::size_t current_image = 0;
    std::vector<value_t> patches;
    std::size_t current_patch = 0;

    patch_iterator(config& conf, const spot_dataset& dataset, names image_names, std::size_t i = 0)
            : conf(conf), dataset(dataset), image_names(image_names), current_image(i) {
        if (current_image < image_names.size()) {
            patches = mat_to_patches<DBN>(conf, dataset.word_images.at(image_names[current_image]), true);
        }
    }

    patch_iterator(const patch_iterator& rhs) = default;

    // Note: This is highly inefficient and should be avoided
    patch_iterator& operator=(const patch_iterator& rhs){
        if (this != &rhs) {
            cpp_assert(&conf == &rhs.conf, "Can only copy assign similar patch_iterator");
            cpp_assert(&dataset == &rhs.dataset, "Can only copy assign similar patch_iterator");
            cpp_assert(&image_names == &rhs.image_names, "Can only copy assign similar patch_iterator");

            current_image = rhs.current_image;
            patches = rhs.patches;
            current_patch = rhs.current_patch;
        }

        return *this;
    }

    bool operator==(const patch_iterator& rhs) const {
        if (current_image == image_names.size() && current_image == rhs.current_image) {
            return true;
        } else {
            return current_image == rhs.current_image && current_patch == rhs.current_patch;
        }
    }

    bool operator!=(const patch_iterator& rhs) const {
        return !(*this == rhs);
    }

    value_t& operator*() {
        return patches[current_patch];
    }

    const value_t& operator*() const {
        return patches[current_patch];
    }

    value_t* operator->() {
        return &patches[current_patch];
    }

    const value_t* operator->() const {
        return &patches[current_patch];
    }

    patch_iterator operator++() {
        if (current_patch == patches.size() - 1) {
            ++current_image;
            current_patch = 0;

            if (current_image < image_names.size()) {
                patches = mat_to_patches<DBN>(conf, dataset.word_images.at(image_names[current_image]), true);
            }
        } else {
            ++current_patch;
        }

        return *this;
    }

    patch_iterator operator++(int) {
        patch_iterator it = *this;
        ++(*this);
        return it;
    }
};

template <typename DBN>
using dbn_output_t = decltype(std::declval<DBN>().template prepare_one_output<typename DBN::input_one_t>());

template <typename DBN>
using features_t = std::vector<std::vector<dbn_output_t<DBN>>>;

template <typename DBN>
features_t<DBN> prepare_outputs(
    const spot_dataset& dataset, const DBN& dbn, const config& conf,
    names test_image_names, bool training, bool runtime = false) {

    features_t<DBN> test_features_a(test_image_names.size());

    if(!runtime){
        std::cout << "Prepare the outputs ..." << std::endl;
    }

    for(size_t i = 0; i < test_image_names.size(); ++i){
        auto& test_image = test_image_names[i];

        auto& vec = test_features_a[i];

        //Get features from DBN
        auto patches = mat_to_patches<DBN>(conf, dataset.word_images.at(test_image), training);

        vec.reserve(patches.size());

        for(auto& patch : patches){
            vec.push_back(dbn.template prepare_one_output<typename DBN::input_t>());
            vec.back() = dbn.features(patch);
        }

        spot::normalize_feature_vector(vec);
    }

    spot::normalize_features(conf, training, test_features_a);

    if(!runtime){
        std::cout << "... done" << std::endl;
    }

    return test_features_a;
}

template <typename DBN>
features_t<DBN> compute_reference(
    const spot_dataset& dataset, const DBN& dbn, const config& conf,
    names training_images) {

    features_t<DBN> ref_a(training_images.size());

    for(size_t i = 0; i < training_images.size(); ++i){
        auto& test_image = training_images[i];

        auto& vec = ref_a[i];

        //Get features from DBN
        auto patches = mat_to_patches<DBN>(conf, dataset.word_images.at(test_image + ".png"), false);

        vec.reserve(patches.size());

        for(auto& patch : patches){
            vec.push_back(dbn.template prepare_one_output<typename DBN::input_t>());
            vec.back() = dbn.features(patch);
        }

        spot::normalize_feature_vector(vec);
    }

    spot::normalize_features(conf, false, ref_a);

    return ref_a;
}

template <typename TF, typename KV, typename DBN>
double evaluate_patches_param(thread_pool& pool, TF& test_features_a, KV& keywords, const spot_dataset& dataset, config& conf, const DBN& dbn, names train_word_names, names test_image_names, parameters parameters) {
    // 2. Evaluate the performances

    std::vector<double> ap(keywords.size());

    for (std::size_t k = 0; k < keywords.size(); ++k) {
        auto& keyword = keywords[k];

        // a) Select the training images

        auto training_images = select_training_images(dataset, keyword, train_word_names);

        // b) Compute the reference features

        auto ref_a = compute_reference(dataset, dbn, conf, training_images);

        // c) Compute the distances

        auto diffs_a = compute_distances(conf, pool, dataset, test_features_a, ref_a, training_images,
            test_image_names, train_word_names,
            parameters, [&](names train_names){ return compute_reference(dataset, dbn, conf, train_names); });

        // d) Update the local stats

        update_stats_light(k, dataset, keyword, diffs_a, ap, test_image_names);
    }

    return std::accumulate(ap.begin(), ap.end(), 0.0) / ap.size();
}

template <typename Set, typename DBN>
void optimize_parameters(const spot_dataset& dataset, const Set& set, config& conf, const DBN& dbn, names train_word_names, names test_image_names, parameters& param) {
    std::vector<double> sc_band_values;

    for (double sc = 0.01; sc < 0.03; sc += 0.001) {
        sc_band_values.push_back(sc);
    }

    for (double sc = 0.030; sc <= 0.15; sc += 0.005) {
        sc_band_values.push_back(sc);
    }

    for (double sc = 0.2; sc <= 0.5; sc += 0.1) {
        sc_band_values.push_back(sc);
    }

    std::cout << sc_band_values.size() << " Sikoe-Chiba bands to evaluate" << std::endl;

    thread_pool pool;

    // 0. Select the keywords

    auto keywords = select_keywords(conf, dataset, set, train_word_names, test_image_names);

    // 1. Prepare all the outputs

    auto test_features_a = prepare_outputs(dataset, dbn, conf, test_image_names, false);

    double best_mean_ap = 0.0;

    parameters best_param;
    best_param.sc_band = sc_band_values.front();

    std::size_t i = 0;
    for (auto sc : sc_band_values) {
        parameters current_param;
        current_param.sc_band = sc;

        double mean_ap = evaluate_patches_param(
            pool, test_features_a, keywords, dataset, conf, dbn, train_word_names, test_image_names, current_param);

        std::cout << "(" << i++ << "/" << sc_band_values.size() << ") sc:" << sc << " map: " << mean_ap << std::endl;

        if (mean_ap > best_mean_ap) {
            best_param   = current_param;
            best_mean_ap = mean_ap;
        }
    }

    std::cout << "Selected as the best parameters" << std::endl;
    std::cout << "\tsc_band: " << best_param.sc_band << std::endl;

    param = best_param;
}

template <typename Set, typename DBN>
std::string evaluate_patches(const spot_dataset& dataset, const Set& set, config& conf, const DBN& dbn, names train_word_names, names test_image_names, bool training, parameters parameters, bool features, bool runtime = false) {
    thread_pool pool;

    if (features) {
        auto test_features_a = prepare_outputs(dataset, dbn, conf, test_image_names, training, runtime);

        if(!runtime){
            export_features(conf, test_image_names, test_features_a, ".2");
        }

        return {};
    } else {
        // 0. Select the keywords

        auto keywords = select_keywords(conf, dataset, set, train_word_names, test_image_names);

        // 1. Select a folder

        auto result_folder = select_folder("./results/");

        // 2. Generate the rel files

        generate_rel_files(result_folder, dataset, test_image_names, keywords);

        // 3. Prepare all the outputs

        auto test_features_a = prepare_outputs(dataset, dbn, conf, test_image_names, training, runtime);

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

            auto ref_a = compute_reference(dataset, dbn, conf, training_images);

            // c) Compute the distances

            auto diffs_a = compute_distances(conf, pool, dataset, test_features_a, ref_a, training_images,
                test_image_names, train_word_names,
                parameters, [&](names train_names){ return compute_reference(dataset, dbn, conf, train_names);});

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

void patches_train(
    const spot_dataset& dataset, const spot_dataset_set& set, config& conf,
    names train_word_names, names train_image_names, names valid_image_names, names test_image_names, bool features, bool runtime) {

    auto pretraining_image_names = train_image_names;

    log_scaling();

    if (conf.all) {
        std::cout << "Use all images from pretraining" << std::endl;

        pretraining_image_names.reserve(pretraining_image_names.size() + valid_image_names.size() + test_image_names.size());

        //Copy valid and image into the pretraining set
        std::copy(valid_image_names.begin(), valid_image_names.end(), std::back_inserter(pretraining_image_names));
        std::copy(test_image_names.begin(), test_image_names.end(), std::back_inserter(pretraining_image_names));
    }

    if(conf.sub){
        std::cout << "Use only 200 images from pretraining" << std::endl;

        static std::random_device rd;
        static std::mt19937_64 g(rd());

        std::shuffle(pretraining_image_names.begin(), pretraining_image_names.end(), g);

        pretraining_image_names.resize(200);
    }

    if (conf.half) {
        if(!runtime){
            std::cout << "Use a half of the resolution" << std::endl;
        }

        copy_from_namespace(half);

        cpp_unused(clipping_1);
        cpp_unused(clipping_2);
        cpp_unused(clipping_3);

#if defined(HALF_CRBM_PMP_1)
        using cdbn_t =
            dll::dbn_desc<
                dll::dbn_layers<
                    dll::conv_rbm_mp_desc<
                        1, NV1_1, NV1_2, K1, NF1, NF1, C1, dll::weight_type<weight>, dll::batch_size<half::B1>, dll::momentum, dll::weight_decay<half::DT1>, dll::hidden<half::HT1>, dll::sparsity<half::SM1>, dll::shuffle_cond<shuffle_1>, dll::dbn_only>::layer_t>>::dbn_t;
#elif defined(HALF_CRBM_PMP_2)
        using cdbn_t =
            dll::dbn_desc<
                dll::dbn_layers<
                    dll::conv_rbm_mp_desc<
                        1, NV1_1, NV1_2, K1, NF1, NF1, C1, dll::weight_type<weight>, dll::batch_size<half::B1>, dll::momentum, dll::weight_decay<half::DT1>, dll::hidden<half::HT1>, dll::sparsity<half::SM1>, dll::shuffle_cond<shuffle_1>, dll::dbn_only>::layer_t,
                    dll::conv_rbm_mp_desc<
                        K1, NV2_1, NV2_2, K2, NF2, NF2, C2, dll::weight_type<weight>, dll::batch_size<half::B2>, dll::momentum, dll::weight_decay<half::DT2>, dll::hidden<half::HT2>, dll::sparsity<half::SM2>, dll::shuffle_cond<shuffle_2>, dll::dbn_only>::layer_t>>::dbn_t;
#elif defined(HALF_CRBM_PMP_3)
        using cdbn_t =
            dll::dbn_desc<
                dll::dbn_layers<
                    dll::conv_rbm_mp_desc<
                        1, NV1_1, NV1_2, K1, NF1, NF1, C1, dll::weight_type<weight>, dll::batch_size<half::B1>, dll::momentum, dll::weight_decay<half::DT1>, dll::hidden<half::HT1>, dll::sparsity<half::SM1>, dll::shuffle_cond<shuffle_1>, dll::dbn_only>::layer_t,
                    dll::conv_rbm_mp_desc<
                        K1, NV2_1, NV2_2, K2, NF2, NF2, C2, dll::weight_type<weight>, dll::batch_size<half::B2>, dll::momentum, dll::weight_decay<half::DT2>, dll::hidden<half::HT2>, dll::sparsity<half::SM2>, dll::shuffle_cond<shuffle_2>, dll::dbn_only>::layer_t,
                    dll::conv_rbm_mp_desc<
                        K2, NV3_1, NV3_2, K3, NF3, NF3, C3, dll::weight_type<weight>, dll::batch_size<half::B3>, dll::momentum, dll::weight_decay<half::DT3>, dll::hidden<half::HT3>, dll::sparsity<half::SM3>, dll::shuffle_cond<shuffle_3>, dll::dbn_only>::layer_t>>::dbn_t;
#elif defined(HALF_CRBM_MP_1)
        using cdbn_t =
            dll::dbn_desc<
                dll::dbn_layers<
                    dll::conv_rbm_desc<
                        1, NV1_1, NV1_2, K1, NF1, NF1, dll::weight_type<weight>, dll::batch_size<half::B1>, dll::momentum, dll::weight_decay<half::DT1>, dll::hidden<half::HT1>, dll::sparsity<half::SM1>, dll::shuffle_cond<shuffle_1>, dll::dbn_only>::layer_t,
                    dll::mp_3d_layer<K1, NH1_1, NH1_2, 1, C1, C1, dll::weight_type<weight>>>,
                dll::batch_mode,
                >::dbn_t;
#elif defined(HALF_CRBM_MP_2)
        using cdbn_t =
            dll::dbn_desc<
                dll::dbn_layers<
                    dll::conv_rbm_desc<
                        1, NV1_1, NV1_2, K1, NF1, NF1, dll::weight_type<weight>, dll::batch_size<half::B1>, dll::momentum, dll::weight_decay<half::DT1>, dll::hidden<half::HT1>, dll::sparsity<half::SM1>, dll::shuffle_cond<shuffle_1>, dll::dbn_only>::layer_t,
                    dll::mp_3d_layer<K1, NH1_1, NH1_2, 1, C1, C1, dll::weight_type<weight>>,
                    dll::conv_rbm_desc<
                        K1, NV2_1, NV2_2, K2, NF2, NF2, dll::weight_type<weight>, dll::batch_size<half::B2>, dll::momentum, dll::weight_decay<half::DT2>, dll::hidden<half::HT2>, dll::sparsity<half::SM2>, dll::shuffle_cond<shuffle_2>, dll::dbn_only>::layer_t,
                    dll::mp_3d_layer<K2, NH2_1, NH2_2, 1, C2, C2, dll::weight_type<weight>>>>::dbn_t;
#elif defined(HALF_CRBM_MP_3)
        using cdbn_t =
            dll::dbn_desc<
                dll::dbn_layers<
                    dll::conv_rbm_desc<
                        1, NV1_1, NV1_2, K1, NF1, NF1, dll::weight_type<weight>, dll::batch_size<half::B1>, dll::momentum, dll::weight_decay<half::DT1>, dll::hidden<half::HT1>, dll::sparsity<half::SM1>, dll::shuffle_cond<shuffle_1>, dll::dbn_only>::layer_t,
                    dll::mp_3d_layer<K1, NH1_1, NH1_2, 1, C1, C1, dll::weight_type<weight>>,
                    dll::conv_rbm_desc<
                        K1, NV2_1, NV2_2, K2, NF2, NF2, dll::weight_type<weight>, dll::batch_size<half::B2>, dll::momentum, dll::weight_decay<half::DT2>, dll::hidden<half::HT2>, dll::sparsity<half::SM2>, dll::shuffle_cond<shuffle_2>, dll::dbn_only>::layer_t,
                    dll::mp_3d_layer<K2, NH2_1, NH2_1, 1, C2, C2, dll::weight_type<weight>>,
                    dll::conv_rbm_desc<
                        K2, NV3_1, NV3_2, K3, NF3, NF3, dll::weight_type<weight>, dll::batch_size<half::B3>, dll::momentum, dll::weight_decay<half::DT3>, dll::hidden<half::HT3>, dll::sparsity<half::SM3>, dll::shuffle_cond<shuffle_3>, dll::dbn_only>::layer_t,
                    dll::mp_3d_layer<K3, NH3_1, NH3_1, 1, C3, C3, dll::weight_type<weight>>>>::dbn_t;
#else
        static_assert(false, "No architecture has been selected");
#endif

#if defined(HALF_CRBM_PMP_1) || defined(HALF_CRBM_PMP_2) || defined(HALF_CRBM_PMP_3)
        //Probabilistic max poolin models have less layers
        constexpr const std::size_t L1 = 0;
        constexpr const std::size_t L2 = 1;
        constexpr const std::size_t L3 = 2;
#else
        constexpr const std::size_t L1 = 0;
        constexpr const std::size_t L2 = 2;
        constexpr const std::size_t L3 = 4;
#endif

        memory_debug("before cdbn");

        if(!runtime){
            std::cout << "DBN Size: " << sizeof(cdbn_t) << std::endl;
        }

        auto cdbn = std::make_unique<cdbn_t>();

        // Level 1
        half::rate_0(cdbn->template layer_get<L1>().learning_rate);
        half::momentum_0(cdbn->template layer_get<L1>().initial_momentum, cdbn->template layer_get<L1>().final_momentum);
        half::wd_l1_0(cdbn->template layer_get<L1>().l1_weight_cost);
        half::wd_l2_0(cdbn->template layer_get<L1>().l2_weight_cost);
        half::pbias_0(cdbn->template layer_get<L1>().pbias);
        half::pbias_lambda_0(cdbn->template layer_get<L1>().pbias_lambda);

#if HALF_LEVELS >= 2
        //Level 2
        half::rate_1(cdbn->template layer_get<L2>().learning_rate);
        half::momentum_1(cdbn->template layer_get<L2>().initial_momentum, cdbn->template layer_get<L2>().final_momentum);
        half::wd_l1_1(cdbn->template layer_get<L2>().l1_weight_cost);
        half::wd_l2_1(cdbn->template layer_get<L2>().l2_weight_cost);
        half::pbias_1(cdbn->template layer_get<L2>().pbias);
        half::pbias_lambda_1(cdbn->template layer_get<L2>().pbias_lambda);
#endif

#if HALF_LEVELS >= 3
        //Level 3
        half::rate_2(cdbn->template layer_get<L3>().learning_rate);
        half::momentum_2(cdbn->template layer_get<L3>().initial_momentum, cdbn->template layer_get<L3>().final_momentum);
        half::wd_l1_2(cdbn->template layer_get<L3>().l1_weight_cost);
        half::wd_l2_2(cdbn->template layer_get<L3>().l2_weight_cost);
        half::pbias_2(cdbn->template layer_get<L3>().pbias);
        half::pbias_lambda_2(cdbn->template layer_get<L3>().pbias_lambda);
#endif

        if (!runtime) {
            cdbn->display();
            std::cout << cdbn->output_size() << " output features" << std::endl;
        }

        constexpr const auto patch_width  = half::patch_width;
        constexpr const auto patch_height = half::patch_height;
        constexpr const auto train_stride = half::train_stride;
        constexpr const auto test_stride  = half::test_stride;

        if (!runtime) {
            std::cout << "patch_height=" << patch_height << std::endl;
            std::cout << "patch_width=" << patch_width << std::endl;
            std::cout << "train_stride=" << train_stride << std::endl;
            std::cout << "test_stride=" << test_stride << std::endl;
        }

        //Pass information to the next passes (evaluation)
        conf.patch_width  = patch_width;
        conf.train_stride = train_stride;
        conf.test_stride  = test_stride;

        const std::string file_name("method_2_half.dat");

        if(!runtime){
            memory_debug("before training");

            if (conf.load || features) {
                cdbn->load(file_name);
            } else {
                std::vector<cdbn_t::template layer_type<0>::input_one_t> training_patches;
                training_patches.reserve(pretraining_image_names.size() * 5);

                std::cout << "Generate patches ..." << std::endl;

                for (auto& name : pretraining_image_names) {
                    auto patches = mat_to_patches<cdbn_t>(conf, dataset.word_images.at(name), true);
                    std::move(patches.begin(), patches.end(), std::back_inserter(training_patches));
                }

                std::cout << "... " << training_patches.size() << " patches extracted" << std::endl;

                memory_debug("after patches extraction");

                cdbn->pretrain(training_patches, half::epochs);
                cdbn->store(file_name);
            }

            memory_debug("after training");
        }

        parameters params;
        params.sc_band = 0.1;

        if(global_scaling || features || runtime || !(conf.load && conf.notrain)){
            if(!runtime){
                std::cout << "Evaluate on training set" << std::endl;
            }

            auto folder = evaluate_patches(dataset, set, conf, *cdbn, train_word_names, train_image_names, true, params, features, runtime);

            if(!runtime){
                cdbn->store(folder + "/" + file_name);
            }
        }

        if (!conf.hmm && !runtime) {
            if (conf.fix) {
                if (conf.ak || conf.botany) {
                    params.sc_band = 0.09;
                } else {
                    params.sc_band = 0.06;
                }

                std::cout << "Switch to optimal parameters" << std::endl;
                std::cout << "\tsc_band: " << params.sc_band << std::endl;
            } else {
                std::cout << "Optimize parameters" << std::endl;
                optimize_parameters(dataset, set, conf, *cdbn, train_word_names, valid_image_names, params);
            }
        }

        if(!runtime && (features || !(conf.load && conf.novalid))){
            std::cout << "Evaluate on validation set" << std::endl;
            auto folder = evaluate_patches(dataset, set, conf, *cdbn, train_word_names, valid_image_names, false, params, features, runtime);

            if(!runtime){
                cdbn->store(folder + "/" + file_name);
            }
        }

        if(!runtime){
            std::cout << "Evaluate on test set" << std::endl;
            auto folder = evaluate_patches(dataset, set, conf, *cdbn, train_word_names, test_image_names, false, params, features, runtime);

            if(!runtime){
                cdbn->store(folder + "/" + file_name);
            }
        }

#if HALF_LEVELS < 2
        silence_l2_warnings();
#endif

#if HALF_LEVELS < 3
        silence_l3_warnings();
#endif
    } else if (conf.third) {
        if(!runtime){
            std::cout << "Use a third of the resolution" << std::endl;
        }

        copy_from_namespace(third);

#if defined(THIRD_CRBM_PMP_1)
        using cdbn_t =
            dll::dbn_desc<
                dll::dbn_layers<
                    dll::conv_rbm_mp_desc<
                        1, NV1_1, NV1_2, K1, NF1, NF1, C1, dll::weight_type<weight>, dll::batch_size<third::B1>, dll::momentum, dll::weight_decay<third::DT1>, dll::hidden<third::HT1>, dll::sparsity<third::SM1>, dll::shuffle_cond<shuffle_1>, dll::clipping_cond<clipping_1>, dll::dbn_only>::layer_t>>::dbn_t;
#elif defined(THIRD_CRBM_PMP_2)
        using cdbn_t =
            dll::dbn_desc<
                dll::dbn_layers<
                    dll::conv_rbm_mp_desc<
                        1, NV1_1, NV1_2, K1, NF1, NF1, C1, dll::weight_type<weight>, dll::batch_size<third::B1>, dll::momentum, dll::weight_decay<third::DT1>, dll::hidden<third::HT1>, dll::sparsity<third::SM1>, dll::shuffle_cond<shuffle_1>, dll::clipping_cond<clipping_1>, dll::dbn_only>::layer_t,
                    dll::conv_rbm_mp_desc<
                        K1, NV2_1, NV2_2, K2, NF2, NF2, C2, dll::weight_type<weight>, dll::batch_size<third::B2>, dll::momentum, dll::weight_decay<third::DT2>, dll::hidden<third::HT2>, dll::sparsity<third::SM2>, dll::shuffle_cond<shuffle_2>, dll::clipping_cond<clipping_2>, dll::dbn_only>::layer_t>, dll::batch_mode>::dbn_t;
#elif defined(THIRD_CRBM_PMP_3)
        using cdbn_t =
            dll::dbn_desc<
                dll::dbn_layers<
                    dll::conv_rbm_mp_desc<
                        1, NV1_1, NV1_2, K1, NF1, NF1, C1, dll::weight_type<weight>, dll::batch_size<third::B1>, dll::momentum, dll::weight_decay<third::DT1>, dll::hidden<third::HT1>, dll::sparsity<third::SM1>, dll::shuffle_cond<shuffle_1>, dll::clipping_cond<clipping_1>, dll::dbn_only>::layer_t,
                    dll::conv_rbm_mp_desc<
                        K1, NV2_1, NV2_2, K2, NF2, NF2, C2, dll::weight_type<weight>, dll::batch_size<third::B2>, dll::momentum, dll::weight_decay<third::DT2>, dll::hidden<third::HT2>, dll::sparsity<third::SM2>, dll::shuffle_cond<shuffle_2>, dll::clipping_cond<clipping_2>, dll::dbn_only>::layer_t,
                    dll::conv_rbm_mp_desc<
                        K2, NV3_1, NV3_2, K3, NF3, NF3, C3, dll::weight_type<weight>, dll::batch_size<third::B3>, dll::momentum, dll::weight_decay<third::DT3>, dll::hidden<third::HT3>, dll::sparsity<third::SM3>, dll::shuffle_cond<shuffle_3>, dll::clipping_cond<clipping_3>, dll::dbn_only>::layer_t>, dll::batch_mode>::dbn_t;
#elif defined(THIRD_CRBM_MP_1)
        using cdbn_t =
            dll::dbn_desc<
                dll::dbn_layers<
                    dll::conv_rbm_desc<
                        1, NV1_1, NV1_2, K1, NF1, NF1, dll::weight_type<weight>, dll::batch_size<third::B1>, dll::momentum, dll::weight_decay<third::DT1>, dll::hidden<third::HT1>, dll::sparsity<third::SM1>, dll::shuffle_cond<shuffle_1>, dll::clipping_cond<clipping_1>, dll::dbn_only>::layer_t,
                    dll::mp_3d_layer<K1, NH1_1, NH1_2, 1, C1, C1, dll::weight_type<weight>>>,
                dll::batch_mode>::dbn_t;
#elif defined(THIRD_CRBM_MP_2)
        using cdbn_t =
            dll::dbn_desc<
                dll::dbn_layers<
                    dll::conv_rbm_desc<
                        1, NV1_1, NV1_2, K1, NF1, NF1, dll::weight_type<weight>, dll::batch_size<third::B1>, dll::momentum, dll::weight_decay<third::DT1>, dll::visible<third::VT1>, dll::hidden<third::HT1>, dll::sparsity<third::SM1>, dll::shuffle_cond<shuffle_1>, dll::clipping_cond<clipping_1>, dll::dbn_only>::layer_t,
                    dll::mp_3d_layer<K1, NH1_1, NH1_2, 1, C1, C1, dll::weight_type<weight>>,
                    dll::conv_rbm_desc<
                        K1, NV2_1, NV2_2, K2, NF2, NF2, dll::weight_type<weight>, dll::batch_size<third::B2>, dll::momentum, dll::weight_decay<third::DT2>, dll::visible<third::VT1>, dll::hidden<third::HT2>, dll::sparsity<third::SM2>, dll::shuffle_cond<shuffle_2>, dll::clipping_cond<clipping_2>, dll::dbn_only>::layer_t,
                    dll::mp_3d_layer<K2, NH2_1, NH2_2, 1, C2, C2, dll::weight_type<weight>>>
                , dll::batch_mode>::dbn_t;
#elif defined(THIRD_CRBM_MP_3)
        using cdbn_t =
            dll::dbn_desc<
                dll::dbn_layers<
                    dll::conv_rbm_desc<
                        1, NV1_1, NV1_2, K1, NF1, NF1, dll::weight_type<weight>, dll::batch_size<third::B1>, dll::momentum, dll::weight_decay<third::DT1>, dll::hidden<third::HT1>, dll::sparsity<third::SM1>, dll::shuffle_cond<shuffle_1>, dll::clipping_cond<clipping_1>, dll::dbn_only>::layer_t,
                    dll::mp_3d_layer<K1, NH1_1, NH1_2, 1, C1, C1, dll::weight_type<weight>>,
                    dll::conv_rbm_desc<
                        K1, NV2_1, NV2_2, K2, NF2, NF2, dll::weight_type<weight>, dll::batch_size<third::B2>, dll::momentum, dll::weight_decay<third::DT2>, dll::hidden<third::HT2>, dll::sparsity<third::SM2>, dll::shuffle_cond<shuffle_2>, dll::clipping_cond<clipping_2>, dll::dbn_only>::layer_t,
                    dll::mp_3d_layer<K2, NH2_1, NH2_1, 1, C2, C2, dll::weight_type<weight>>,
                    dll::conv_rbm_desc<
                        K2, NV3_1, NV3_2, K3, NF3, NF3, dll::weight_type<weight>, dll::batch_size<third::B3>, dll::momentum, dll::weight_decay<third::DT3>, dll::hidden<third::HT3>, dll::sparsity<third::SM3>, dll::shuffle_cond<shuffle_3>, dll::clipping_cond<clipping_3>, dll::dbn_only>::layer_t,
                    dll::mp_3d_layer<K3, NH3_1, NH3_1, 1, C3, C3, dll::weight_type<weight>>>, dll::batch_mode>::dbn_t;
#elif defined(THIRD_RBM_1)
        using cdbn_t =
            dll::dbn_desc<
                dll::dbn_layers<
                    dll::rbm_desc<
                        NV1_1 * NV1_2 * 1, NF1, dll::weight_type<weight>, dll::batch_size<third::B1>, dll::momentum, dll::weight_decay<third::DT1>, dll::hidden<third::HT1>, dll::sparsity<third::SM1>, dll::shuffle_cond<shuffle_1>, dll::clipping_cond<clipping_1>, dll::dbn_only>::layer_t>, dll::batch_mode>::dbn_t;
#elif defined(THIRD_RBM_2)
        using cdbn_t =
            dll::dbn_desc<
                dll::dbn_layers<
                    dll::rbm_desc<
                        NV1_1 * NV1_2 * 1, NF1, dll::weight_type<weight>, dll::batch_size<third::B1>, dll::momentum, dll::weight_decay<third::DT1>, dll::hidden<third::HT1>, dll::sparsity<third::SM1>, dll::shuffle_cond<shuffle_1>, dll::clipping_cond<clipping_1>, dll::dbn_only>::layer_t,
                    dll::rbm_desc<
                        NF1, NF2, dll::weight_type<weight>, dll::batch_size<third::B2>, dll::momentum, dll::weight_decay<third::DT2>, dll::hidden<third::HT2>, dll::sparsity<third::SM2>, dll::shuffle_cond<shuffle_2>,  dll::clipping_cond<clipping_2>, dll::dbn_only>::layer_t>, dll::batch_mode>::dbn_t;
#elif defined(THIRD_RBM_3)
        using cdbn_t =
            dll::dbn_desc<
                dll::dbn_layers<
                    dll::rbm_desc<
                        NV1_1 * NV1_2 * 1, NF1, dll::weight_type<weight>, dll::batch_size<third::B1>, dll::momentum, dll::weight_decay<third::DT1>, dll::hidden<third::HT1>, dll::sparsity<third::SM1>, dll::shuffle_cond<shuffle_1>, dll::clipping_cond<clipping_1>, dll::dbn_only>::layer_t,
                    dll::rbm_desc<
                        NF1, NF2, dll::weight_type<weight>, dll::batch_size<third::B2>, dll::momentum, dll::weight_decay<third::DT2>, dll::hidden<third::HT2>, dll::sparsity<third::SM2>, dll::shuffle_cond<shuffle_2>, dll::clipping_cond<clipping_2>, dll::dbn_only>::layer_t,
                    dll::rbm_desc<
                        NF2, NF3, dll::weight_type<weight>, dll::batch_size<third::B3>, dll::momentum, dll::weight_decay<third::DT3>, dll::hidden<third::HT3>, dll::sparsity<third::SM3>, dll::shuffle_cond<shuffle_3>, dll::clipping_cond<clipping_3>, dll::dbn_only>::layer_t>, dll::batch_mode>::dbn_t;
#elif defined(THIRD_COMPLEX_2)
        using cdbn_t =
            dll::dbn_desc<
                dll::dbn_layers<
                    dll::conv_rbm_desc<
                        1, NV1_1, NV1_2, K1, NF1, NF1, dll::weight_type<weight>, dll::batch_size<third::B1>, dll::momentum, dll::weight_decay<third::DT1>, dll::hidden<third::HT1>, dll::sparsity<third::SM1>, dll::shuffle_cond<shuffle_1>, dll::clipping_cond<clipping_1>, dll::dbn_only>::layer_t,
                    //dll::lcn_layer_desc<5>::layer_t,
                    dll::mp_3d_layer<K1, NH1_1, NH1_2, 1, C1, C1, dll::weight_type<weight>>,
                    dll::conv_rbm_desc<
                        K1, NV2_1, NV2_2, K2, NF2, NF2, dll::weight_type<weight>, dll::batch_size<third::B2>, dll::momentum, dll::weight_decay<third::DT2>, dll::hidden<third::HT2>, dll::sparsity<third::SM2>, dll::shuffle_cond<shuffle_2>, dll::clipping_cond<clipping_2>, dll::dbn_only>::layer_t,
                    dll::lcn_layer_desc<3>::layer_t,
                    dll::mp_3d_layer<K2, NH2_1, NH2_2, 1, C2, C2, dll::weight_type<weight>>>
                , dll::batch_mode>::dbn_t;
#else
        static_assert(false, "No architecture has been selected");
#endif

#if defined(THIRD_CRBM_MP_1) || defined(THIRD_CRBM_MP_2) || defined(THIRD_CRBM_MP_3)
        //Max pooling layers models have more layers
        constexpr const std::size_t L1 = 0;
        constexpr const std::size_t L2 = 2;
        constexpr const std::size_t L3 = 4;
#elif defined(THIRD_PATCH_CRBM_MP_2)
        //Max pooling layers models have more layers
        constexpr const std::size_t L1 = 1;
        constexpr const std::size_t L2 = 3;
        constexpr const std::size_t L3 = 5;
#elif defined(THIRD_COMPLEX_2)
        // CRBM -> LCN -> MP
        constexpr const std::size_t L1 = 0;
        constexpr const std::size_t L2 = 2;
        constexpr const std::size_t L3 = 1000;
#elif defined(THIRD_MODERN)
        // Distort -> Patches - CRBM -> MP
        constexpr const std::size_t L1 = 1;
        constexpr const std::size_t L2 = 2;
        constexpr const std::size_t L3 = 1000;
#else
        constexpr const std::size_t L1        = 0;
        constexpr const std::size_t L2        = 1;
        constexpr const std::size_t L3        = 2;
#endif

        auto cdbn = std::make_unique<cdbn_t>();

#if defined(THIRD_MODERN)
        auto cdbn_train = std::make_unique<cdbn_train_t>();
        auto& cdbn_ref = cdbn_train;
#elif defined(THIRD_PATCH_CRBM_MP_2)
        auto cdbn_train = std::make_unique<cdbn_t>();
        auto& cdbn_ref = cdbn_train;
#else
        auto& cdbn_ref = cdbn;
#endif

        // Level 1
        third::rate_0(cdbn_ref->template layer_get<L1>().learning_rate);
        third::momentum_0(cdbn_ref->template layer_get<L1>().initial_momentum, cdbn_ref->template layer_get<L1>().final_momentum);
        third::wd_l1_0(cdbn_ref->template layer_get<L1>().l1_weight_cost);
        third::wd_l2_0(cdbn_ref->template layer_get<L1>().l2_weight_cost);
        third::pbias_0(cdbn_ref->template layer_get<L1>().pbias);
        third::pbias_lambda_0(cdbn_ref->template layer_get<L1>().pbias_lambda);
        third::sparsity_target_0(cdbn_ref->template layer_get<L1>().sparsity_target);
        third::clip_norm_1(cdbn_ref->template layer_get<L1>().gradient_clip);

#if THIRD_LEVELS >= 2
        //Level 2
        third::rate_1(cdbn_ref->template layer_get<L2>().learning_rate);
        third::momentum_1(cdbn_ref->template layer_get<L2>().initial_momentum, cdbn_ref->template layer_get<L2>().final_momentum);
        third::wd_l1_1(cdbn_ref->template layer_get<L2>().l1_weight_cost);
        third::wd_l2_1(cdbn_ref->template layer_get<L2>().l2_weight_cost);
        third::pbias_1(cdbn_ref->template layer_get<L2>().pbias);
        third::pbias_lambda_1(cdbn_ref->template layer_get<L2>().pbias_lambda);
        third::sparsity_target_1(cdbn_ref->template layer_get<L2>().sparsity_target);
        third::clip_norm_2(cdbn_ref->template layer_get<L2>().gradient_clip);
#endif

#if THIRD_LEVELS >= 3
        //Level 3
        third::rate_2(cdbn_ref->template layer_get<L3>().learning_rate);
        third::momentum_2(cdbn_ref->template layer_get<L3>().initial_momentum, cdbn_ref->template layer_get<L3>().final_momentum);
        third::wd_l1_2(cdbn_ref->template layer_get<L3>().l1_weight_cost);
        third::wd_l2_2(cdbn_ref->template layer_get<L3>().l2_weight_cost);
        third::pbias_2(cdbn_ref->template layer_get<L3>().pbias);
        third::pbias_lambda_2(cdbn_ref->template layer_get<L3>().pbias_lambda);
        third::sparsity_target_2(cdbn_ref->template layer_get<L3>().sparsity_target);
        third::clip_norm_3(cdbn_ref->template layer_get<L3>().gradient_clip);
#endif

#ifdef THIRD_COMPLEX_2
        //cdbn->template layer_get<1>().sigma = 2.0;
        cdbn_ref->template layer_get<3>().sigma = 2.0;
#endif

        if (!runtime) {
            cdbn_ref->display();
            std::cout << cdbn->output_size() << " output features" << std::endl;
        }

        constexpr const auto patch_width  = third::patch_width;
        constexpr const auto patch_height = third::patch_height;
        constexpr const auto train_stride = third::train_stride;
        constexpr const auto test_stride  = third::test_stride;

        if (!runtime) {
            std::cout << "patch_height=" << patch_height << std::endl;
            std::cout << "patch_width=" << patch_width << std::endl;
            std::cout << "train_stride=" << train_stride << std::endl;
            std::cout << "test_stride=" << test_stride << std::endl;
        }

        //Pass information to the next passes (evaluation)
        conf.patch_width  = patch_width;
        conf.train_stride = train_stride;
        conf.test_stride  = test_stride;

        const std::string file_name("method_2_third.dat");

        if (!runtime) {
            //Train the DBN
            if (conf.load || features) {
                cdbn->load(file_name);
            } else {
#if defined(THIRD_MODERN) || defined(THIRD_PATCH_CRBM_MP_2)
                std::cout << "Training is done with inline distort/patches..." << std::endl;

                std::vector<etl::dyn_matrix<weight, 3>> training_images;
                training_images.reserve(pretraining_image_names.size());

                std::cout << "Generate images ..." << std::endl;

                for (auto& name : pretraining_image_names) {
                    training_images.push_back(mat_for_patches(conf, dataset.word_images.at(name)));
                }

                std::cout << "... done" << std::endl;

                cdbn_train->pretrain(training_images, third::epochs);

                // Exchance weights
                cdbn_train->store(file_name);
                cdbn->load(file_name);
#else
                if (conf.iam && !conf.sub) {
                    std::cout << "Training is done with patch iterators..." << std::endl;

                    if (third::elastic_augment) {
                        std::cout << "WARNING: Elastic distortions is not supported for patch_iterator yet" << std::endl;
                    }

                    patch_iterator<cdbn_t> it(conf, dataset, pretraining_image_names);
                    patch_iterator<cdbn_t> end(conf, dataset, pretraining_image_names, pretraining_image_names.size());

                    cdbn->pretrain(it, end, third::epochs);
                } else {
                    std::vector<cdbn_t::template layer_type<0>::input_one_t> training_patches;
                    training_patches.reserve(pretraining_image_names.size() * 10);

                    std::cout << "Generate patches ..." << std::endl;

                    for (auto& name : pretraining_image_names) {
                        decltype(auto) image = dataset.word_images.at(name);

                        // Insert the patches from the original image
                        auto patches = mat_to_patches<cdbn_t>(conf, image, true);
                        std::copy(patches.begin(), patches.end(), std::back_inserter(training_patches));

                        // Insert the patches from the distorted versions
                        for (std::size_t d = 0; d < third::elastic_augment; ++d) {
                            auto distorted_image = elastic_distort(image);

                            auto patches = mat_to_patches<cdbn_t>(conf, distorted_image, true);
                            std::copy(patches.begin(), patches.end(), std::back_inserter(training_patches));
                        }
                    }

                    std::cout << "... " << training_patches.size() << " patches extracted" << std::endl;

                    cdbn->pretrain(training_patches, third::epochs);
                }
#endif

                cdbn->store(file_name);
            }
        }

        parameters params;
        params.sc_band = 0.1;

        if(global_scaling || features || runtime || !conf.notrain){
            dll::auto_timer timer("evaluate:train");

            if(!runtime){
                std::cout << "Evaluate on training set" << std::endl;
            }

            auto folder = evaluate_patches(dataset, set, conf, *cdbn, train_word_names, train_image_names, true, params, features, runtime);

            if(!runtime){
                cdbn->store(folder + "/" + file_name);
            }
        }

        if (!conf.hmm && !runtime) {
            if (conf.fix) {
                std::cout << "Switch to optimal parameters" << std::endl;
                if(conf.parzival){
                    params.sc_band = 0.04;
                } else {
                    params.sc_band = 0.05;
                }
                std::cout << "\tsc_band: " << params.sc_band << std::endl;
            } else {
                std::cout << "Optimize parameters" << std::endl;
                optimize_parameters(dataset, set, conf, *cdbn, train_word_names, test_image_names, params);
            }
        }

        if(!runtime && (features || !conf.novalid)){
            dll::auto_timer timer("evaluate:validation");

            std::cout << "Evaluate on validation set" << std::endl;
            auto folder = evaluate_patches(dataset, set, conf, *cdbn, train_word_names, valid_image_names, false, params, features, runtime);

            if(!runtime){
                cdbn->store(folder + "/" + file_name);
            }
        }

        if(!runtime){
            dll::auto_timer timer("evaluate:test");

            std::cout << "Evaluate on test set" << std::endl;
            auto folder = evaluate_patches(dataset, set, conf, *cdbn, train_word_names, test_image_names, false, params, features, runtime);

            if(!runtime){
                cdbn->store(folder + "/" + file_name);
            }
        }

#if defined(THIRD_RBM_1) || defined(THIRD_RBM_2) || defined(THIRD_RBM_3)
        //Silence some warnings
        cpp_unused(K1);
        cpp_unused(K2);
        cpp_unused(K3);
#endif

#if THIRD_LEVELS < 2
        silence_l2_warnings();
#endif

#if THIRD_LEVELS < 3
        silence_l3_warnings();
#endif
    } else {
        if(!runtime){
            std::cout << "Use full resolution" << std::endl;
        }

        copy_from_namespace(full);

        // Clipping is not support now for "full"
        cpp_unused(clipping_1);
        cpp_unused(clipping_2);
        cpp_unused(clipping_3);

        // Shuffle is not supported now for "full"
        cpp_unused(shuffle_1);
        cpp_unused(shuffle_2);

#if defined(FULL_CRBM_PMP_1)
        using cdbn_t =
            dll::dbn_desc<
                dll::dbn_layers<
                    dll::conv_rbm_mp_desc<
                        1, NV1_1, NV1_2, K1, NF1, NF1, C1, dll::weight_type<weight>, dll::batch_size<full::B1>, dll::momentum, dll::weight_decay<full::DT1>, dll::hidden<full::HT1>, dll::sparsity<full::SM1>, /*dll::shuffle_cond<shuffle_1>,*/ dll::dbn_only>::layer_t>>::dbn_t;
#elif defined(FULL_CRBM_PMP_2)
        using cdbn_t =
            dll::dbn_desc<
                dll::dbn_layers<
                    dll::conv_rbm_mp_desc<
                        1, NV1_1, NV1_2, K1, NF1, NF1, C1, dll::weight_type<weight>, dll::batch_size<full::B1>, dll::momentum, dll::weight_decay<full::DT1>, dll::hidden<full::HT1>, dll::sparsity<full::SM1>, /*dll::shuffle_cond<shuffle_1>,*/ dll::dbn_only>::layer_t,
                    dll::conv_rbm_mp_desc<
                        K1, NV2_1, NV2_2, K2, NF2, NF2, C2, dll::weight_type<weight>, dll::batch_size<full::B2>, dll::momentum, dll::weight_decay<full::DT2>, dll::hidden<full::HT2>, dll::sparsity<full::SM2>, /*dll::shuffle_cond<shuffle_2>,*/ dll::dbn_only>::layer_t>,
                dll::batch_mode,
                dll::batch_size<5>>::dbn_t;
#elif defined(FULL_CRBM_PMP_3)
        using cdbn_t =
            dll::dbn_desc<
                dll::dbn_layers<
                    dll::conv_rbm_mp_desc<
                        1, NV1_1, NV1_2, K1, NF1, NF1, C1, dll::weight_type<weight>, dll::batch_size<full::B1>, dll::momentum, dll::weight_decay<full::DT1>, dll::hidden<full::HT1>, dll::sparsity<full::SM1>, /*dll::shuffle_cond<shuffle_1>,*/ dll::dbn_only>::layer_t,
                    dll::conv_rbm_mp_desc<
                        K1, NV2_1, NV2_2, K2, NF2, NF2, C2, dll::weight_type<weight>, dll::batch_size<full::B2>, dll::momentum, dll::weight_decay<full::DT2>, dll::hidden<full::HT2>, dll::sparsity<full::SM2>, /*dll::shuffle_cond<shuffle_2>,*/ dll::dbn_only>::layer_t,
                    dll::conv_rbm_mp_desc<
                        K2, NV3_1, NV3_2, K3, NF3, NF3, C3, dll::weight_type<weight>, dll::batch_size<full::B3>, dll::momentum, dll::weight_decay<full::DT3>, dll::hidden<full::HT3>, dll::sparsity<full::SM3>, /*dll::shuffle_cond<shuffle_3>,*/ dll::dbn_only>::layer_t>>::dbn_t;
#elif defined(FULL_CRBM_MP_1)
        using cdbn_t =
            dll::dbn_desc<
                dll::dbn_layers<
                    dll::conv_rbm_desc<
                        1, NV1_1, NV1_2, K1, NF1, NF1, dll::weight_type<weight>, dll::batch_size<full::B1>, dll::momentum, dll::weight_decay<full::DT1>, dll::hidden<full::HT1>, dll::sparsity<full::SM1>, /*dll::shuffle_cond<shuffle_1>,*/ dll::dbn_only>::layer_t,
                    dll::mp_3d_layer<K1, NH1_1, NH1_2, 1, C1, C1, dll::weight_type<weight>>>,
                dll::batch_mode>::dbn_t;
#elif defined(FULL_CRBM_MP_2)
        using cdbn_t =
            dll::dbn_desc<
                dll::dbn_layers<
                    dll::conv_rbm_desc<
                        1, NV1_1, NV1_2, K1, NF1, NF1, dll::weight_type<weight>, dll::batch_size<full::B1>, dll::momentum, dll::weight_decay<full::DT1>, dll::hidden<full::HT1>, dll::sparsity<full::SM1>, /*dll::shuffle_cond<shuffle_1>,*/ dll::dbn_only>::layer_t,
                    dll::mp_3d_layer<K1, NH1_1, NH1_2, 1, C1, C1, dll::weight_type<weight>>,
                    dll::conv_rbm_desc<
                        K1, NV2_1, NV2_2, K2, NF2, NF2, dll::weight_type<weight>, dll::batch_size<full::B2>, dll::momentum, dll::weight_decay<full::DT2>, dll::hidden<full::HT2>, dll::sparsity<full::SM2>, /*dll::shuffle_cond<shuffle_2>,*/ dll::dbn_only>::layer_t,
                    dll::mp_3d_layer<K2, NH2_1, NH2_2, 1, C2, C2, dll::weight_type<weight>>>,
                dll::batch_mode>::dbn_t;
#elif defined(FULL_CRBM_MP_3)
        using cdbn_t =
            dll::dbn_desc<
                dll::dbn_layers<
                    dll::conv_rbm_desc<
                        1, NV1_1, NV1_2, K1, NF1, NF1, dll::weight_type<weight>, dll::batch_size<full::B1>, dll::momentum, dll::weight_decay<full::DT1>, dll::hidden<full::HT1>, dll::sparsity<full::SM1>, /*dll::shuffle_cond<shuffle_1>,*/ dll::dbn_only>::layer_t,
                    dll::mp_3d_layer<K1, NH1_1, NH1_2, 1, C1, C1, dll::weight_type<weight>>,
                    dll::conv_rbm_desc<K1, NV2_1, NV2_2, K2, NF2, NF2, dll::weight_type<weight>, dll::batch_size<full::B2>, dll::momentum, dll::weight_decay<full::DT2>, dll::hidden<full::HT2>, dll::sparsity<full::SM2>, /*dll::shuffle_cond<shuffle_2>,*/ dll::dbn_only>::layer_t,
                    dll::mp_3d_layer<K2, NH2_1, NH2_1, 1, C2, C2, dll::weight_type<weight>>,
                    dll::conv_rbm_desc<
                        K2, NV3_1, NV3_2, K3, NF3, NF3, dll::weight_type<weight>, dll::batch_size<full::B3>, dll::momentum, dll::weight_decay<full::DT3>, dll::hidden<full::HT3>, dll::sparsity<full::SM3>, /*dll::shuffle_cond<shuffle_3>,*/ dll::dbn_only>::layer_t,
                    dll::mp_3d_layer<K3, NH3_1, NH3_1, 1, C3, C3, dll::weight_type<weight>>>>::dbn_t;
#else
        static_assert(false, "No architecture has been selected");
#endif

#if defined(FULL_CRBM_PMP_1) || defined(FULL_CRBM_PMP_2) || defined(FULL_CRBM_PMP_3)
        //Probabilistic max poolin models have less layers
        constexpr const std::size_t L1 = 0;
        constexpr const std::size_t L2 = 1;
        constexpr const std::size_t L3 = 2;
#else
        constexpr const std::size_t L1 = 0;
        constexpr const std::size_t L2 = 2;
        constexpr const std::size_t L3 = 4;
#endif

        auto cdbn = std::make_unique<cdbn_t>();

        // Level 1
        full::rate_0(cdbn->template layer_get<L1>().learning_rate);
        full::momentum_0(cdbn->template layer_get<L1>().initial_momentum, cdbn->template layer_get<L1>().final_momentum);
        full::wd_l1_0(cdbn->template layer_get<L1>().l1_weight_cost);
        full::wd_l2_0(cdbn->template layer_get<L1>().l2_weight_cost);
        full::pbias_0(cdbn->template layer_get<L1>().pbias);
        full::pbias_lambda_0(cdbn->template layer_get<L1>().pbias_lambda);

#if FULL_LEVELS >= 2
        //Level 2
        full::rate_1(cdbn->template layer_get<L2>().learning_rate);
        full::momentum_1(cdbn->template layer_get<L2>().initial_momentum, cdbn->template layer_get<L2>().final_momentum);
        full::wd_l1_1(cdbn->template layer_get<L2>().l1_weight_cost);
        full::wd_l2_1(cdbn->template layer_get<L2>().l2_weight_cost);
        full::pbias_1(cdbn->template layer_get<L2>().pbias);
        full::pbias_lambda_1(cdbn->template layer_get<L2>().pbias_lambda);
#endif

#if FULL_LEVELS >= 3
        //Level 3
        full::rate_2(cdbn->template layer_get<L3>().learning_rate);
        full::momentum_2(cdbn->template layer_get<L3>().initial_momentum, cdbn->template layer_get<L3>().final_momentum);
        full::wd_l1_2(cdbn->template layer_get<L3>().l1_weight_cost);
        full::wd_l2_2(cdbn->template layer_get<L3>().l2_weight_cost);
        full::pbias_2(cdbn->template layer_get<L3>().pbias);
        full::pbias_lambda_2(cdbn->template layer_get<L3>().pbias_lambda);
#endif

        if(!runtime){
            cdbn->display();
            std::cout << cdbn->output_size() << " output features" << std::endl;
        }

        constexpr const auto patch_width  = full::patch_width;
        constexpr const auto patch_height = full::patch_height;
        constexpr const auto train_stride = full::train_stride;
        constexpr const auto test_stride  = full::test_stride;

        if(!runtime){
            std::cout << "patch_height=" << patch_height << std::endl;
            std::cout << "patch_width=" << patch_width << std::endl;
            std::cout << "train_stride=" << train_stride << std::endl;
            std::cout << "test_stride=" << test_stride << std::endl;
        }

        //Pass information to the next passes (evaluation)
        conf.patch_width  = patch_width;
        conf.train_stride = train_stride;
        conf.test_stride  = test_stride;

        const std::string file_name("method_2_full.dat");

        if (!runtime) {
            //1. Pretraining
            if (conf.load || features) {
                cdbn->load(file_name);
            } else {
#ifdef FULL_CRBM_PMP_2
                std::vector<etl::dyn_matrix<weight, 3>> training_images;
                training_images.reserve(pretraining_image_names.size());

                std::cout << "Generate images ..." << std::endl;

                for (auto& name : pretraining_image_names) {
                    training_images.push_back(mat_for_patches(conf, dataset.word_images.at(name)));
                }

                std::cout << "... " << training_images.size() << " patches extracted" << std::endl;

                cdbn->pretrain(training_images, full::epochs);
                cdbn->store(file_name);

#else
                patch_iterator<cdbn_t> it(conf, dataset, pretraining_image_names);
                patch_iterator<cdbn_t> end(conf, dataset, pretraining_image_names, pretraining_image_names.size());

                cdbn->pretrain(it, end, full::epochs);
                cdbn->store(file_name);
#endif
            }
        }

        //2. Evaluation

        parameters params;
        params.sc_band = 0.1;

        if(global_scaling || runtime || features || !(conf.load && conf.notrain)){
            if(!runtime){
                std::cout << "Evaluate on training set" << std::endl;
            }

            auto folder = evaluate_patches(dataset, set, conf, *cdbn, train_word_names, train_image_names, true, params, features, runtime);

            if(!runtime){
                cdbn->store(folder + "/" + file_name);
            }
        }

        if (!conf.hmm && !runtime) {
            if (conf.fix) {
                std::cout << "Switch to optimal parameters" << std::endl;
                params.sc_band = 0.05;
                std::cout << "\tsc_band: " << params.sc_band << std::endl;
            } else {
                std::cout << "Optimize parameters" << std::endl;
                optimize_parameters(dataset, set, conf, *cdbn, train_word_names, valid_image_names, params);
            }
        }

        if(!runtime && (features || !(conf.load && conf.novalid))){
            std::cout << "Evaluate on validation set" << std::endl;
            auto folder = evaluate_patches(dataset, set, conf, *cdbn, train_word_names, valid_image_names, false, params, features, runtime);

            if(!runtime){
                cdbn->store(folder + "/" + file_name);
            }
        }

        if(!runtime){
            std::cout << "Evaluate on test set" << std::endl;
            auto folder = evaluate_patches(dataset, set, conf, *cdbn, train_word_names, test_image_names, false, params, features, runtime);

            if(!runtime){
                cdbn->store(folder + "/" + file_name);
            }
        }

#if FULL_LEVELS < 2
        silence_l2_warnings();
#endif

#if FULL_LEVELS < 3
        silence_l3_warnings();
#endif
    }
}

void patches_features(
    const spot_dataset& dataset, const spot_dataset_set& set, config& conf,
    names train_word_names, names train_image_names, names valid_image_names, names test_image_names) {
    //Generate features and save them
    patches_train(dataset, set, conf, train_word_names, train_image_names, valid_image_names, test_image_names, true);
}

void patches_runtime(
    const spot_dataset& dataset, const spot_dataset_set& set, config& conf,
    names train_word_names, names image_names) {
    //Generate features and save them
    patches_train(dataset, set, conf, train_word_names, image_names, image_names, image_names, true, true);
}
