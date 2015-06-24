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

#include "dll/conv_rbm.hpp"
#include "dll/conv_rbm_mp.hpp"
#include "dll/dbn.hpp"
#include "dll/avgp_layer.hpp"
#include "dll/mp_layer.hpp"
#include "dll/ocv_visualizer.hpp"
#include "dll/patches_layer.hpp"
#include "dll/patches_layer_pad.hpp"

#include "nice_svm.hpp"

#include "config.hpp"
#include "standard.hpp"
#include "utils.hpp"
#include "reports.hpp"
#include "dtw.hpp"        //Dynamic time warping

#define LOCAL_MEAN_SCALING
#include "scaling.hpp"      //Scaling functions

//The different configurations
#include "config_third.hpp"
#include "config_half.hpp"
#include "config_full.hpp"

#if defined(THIRD_CRBM_PMP_1) || defined(THIRD_CRBM_MP_1) || defined(THIRD_RBM_1)
#define THIRD_LEVELS 1
#endif

#if defined(THIRD_CRBM_PMP_2) || defined(THIRD_CRBM_MP_2) || defined(THIRD_RBM_2)
#define THIRD_LEVELS 2
#endif

#if defined(THIRD_CRBM_PMP_3) || defined(THIRD_CRBM_MP_3) || defined(THIRD_RBM_3)
#define THIRD_LEVELS 3
#endif

#if defined(HALF_CRBM_PMP_1) || defined(HALF_CRBM_MP_1)
#define HALF_LEVELS 1
#endif

#if defined(HALF_CRBM_PMP_2) || defined(HALF_CRBM_MP_2)
#define HALF_LEVELS 2
#endif

#if defined(HALF_CRBM_PMP_3) || defined(HALF_CRBM_MP_3)
#define HALF_LEVELS 3
#endif

#if defined(FULL_CRBM_PMP_1) || defined(FULL_CRBM_MP_1)
#define FULL_LEVELS 1
#endif

#if defined(FULL_CRBM_PMP_2) || defined(FULL_CRBM_MP_2)
#define FULL_LEVELS 2
#endif

#if defined(FULL_CRBM_PMP_3) || defined(FULL_CRBM_MP_3)
#define FULL_LEVELS 3
#endif

#if !defined(HALF_LEVELS) || !defined(THIRD_LEVELS) || !defined(FULL_LEVELS)
static_assert(false, "Invalid configuration");
#endif

namespace {

using thread_pool = cpp::default_thread_pool<>;

struct patch_iterator : std::iterator<std::input_iterator_tag, etl::dyn_matrix<weight, 3>> {
    config& conf;
    const spot_dataset& dataset;
    names image_names;

    std::size_t current_image = 0;
    std::vector<etl::dyn_matrix<weight, 3>> patches;
    std::size_t current_patch = 0;

    patch_iterator(config& conf, const spot_dataset& dataset, names image_names, std::size_t i = 0)
            : conf(conf), dataset(dataset), image_names(image_names), current_image(i) {
        if(current_image < image_names.size()){
            patches = mat_to_patches(conf, dataset.word_images.at(image_names[current_image]), true);
        }
    }

    patch_iterator(const patch_iterator& rhs) = default;
    patch_iterator& operator=(const patch_iterator& rhs) = default;

    bool operator==(const patch_iterator& rhs){
        if(current_image == image_names.size() && current_image == rhs.current_image){
            return true;
        } else {
            return current_image == rhs.current_image && current_patch == rhs.current_patch;
        }
    }

    bool operator!=(const patch_iterator& rhs){
        return !(*this == rhs);
    }
    etl::dyn_matrix<weight, 3>& operator*(){
        return patches[current_patch];
    }

    etl::dyn_matrix<weight, 3>* operator->(){
        return &patches[current_patch];
    }

    patch_iterator operator++(){
        if(current_patch == patches.size() - 1){
            ++current_image;
            current_patch = 0;

            if(current_image < image_names.size()){
                patches = mat_to_patches(conf, dataset.word_images.at(image_names[current_image]), true);
            }
        } else {
            ++current_patch;
        }

        return *this;
    }

    patch_iterator operator++(int){
        patch_iterator it = *this;
        ++(*this);
        return it;
    }
};

struct parameters {
    double sc_band;
};

template<bool D_P, typename Dataset, typename DBN>
std::vector<std::vector<typename DBN::output_t>> prepare_outputs(
        thread_pool& pool, const Dataset& dataset, const DBN& dbn, config& conf,
        names test_image_names, bool training){
    //Get some sizes
    const std::size_t patch_height = HEIGHT / conf.downscale;
    const std::size_t patch_width = conf.patch_width;

    std::vector<std::vector<typename DBN::output_t>> test_features_a(test_image_names.size());

    std::cout << "Prepare the outputs ..." << std::endl;

    cpp::parallel_foreach_i(pool, test_image_names.begin(), test_image_names.end(),
        [&,patch_height,patch_width](auto& test_image, std::size_t i){
            auto& vec = test_features_a[i];

            cpp::static_if<D_P>([&](auto f){
                //Get features from DBN
                auto image = mat_for_patches(conf, dataset.word_images.at(test_image));
                f(dbn).activation_probabilities(image, vec);
            }).else_([&](auto f){
                auto patches = mat_to_patches(conf, dataset.word_images.at(test_image), training);

                for(auto& patch : patches){
                    vec.push_back(dbn.prepare_one_output());
                    f(dbn).activation_probabilities(patch, vec.back());
                }
            });

#ifdef LOCAL_LINEAR_SCALING
            local_linear_feature_scaling(vec);
#endif

#ifdef LOCAL_MEAN_SCALING
            local_mean_feature_scaling(vec);
#endif
        });

#ifdef GLOBAL_MEAN_SCALING
    auto scale = global_mean_scaling(test_features_a, conf, training);
#endif

#ifdef GLOBAL_LINEAR_SCALING
    auto scale = global_linear_scaling(test_features_a, conf, training);
#endif

#if defined(GLOBAL_MEAN_SCALING) || defined(GLOBAL_LINEAR_SCALING)
    for(std::size_t t = 0; t < test_features_a.size(); ++t){
        for(std::size_t i = 0; i < test_features_a[t].size(); ++i){
            for(std::size_t f = 0; f < test_features_a.back().back().size(); ++f){
                test_features_a[t][i][f] = scale(test_features_a[t][i][f], conf.scale_a[f], conf.scale_b[f]);
            }
        }
    }
#else
    cpp_unused(training);
    cpp_unused(conf);
#endif

    std::cout << "... done" << std::endl;

    return test_features_a;
}

template<typename Dataset>
std::vector<std::string> select_training_images(const Dataset& dataset, names keyword, names train_names){
    std::vector<std::string> training_images;

    for(auto& labels : dataset.word_labels){
        if(keyword == labels.second && std::find(train_names.begin(), train_names.end(), labels.first) != train_names.end()){
            training_images.push_back(labels.first);
        }
    }

    return training_images;
}

template<bool D_P, typename Dataset, typename DBN>
std::vector<std::vector<typename DBN::output_t>> compute_reference(
        thread_pool& pool, const Dataset& dataset, const DBN& dbn,
        const config& conf, names training_images){
    std::vector<std::vector<typename DBN::output_t>> ref_a(training_images.size());

    cpp::parallel_foreach_i(pool, training_images.begin(), training_images.end(),
        [&](auto& training_image, std::size_t e){
            cpp::static_if<D_P>([&](auto f){
                //Compute the features
                auto image = mat_for_patches(conf, dataset.word_images.at(training_image + ".png"));
                f(dbn).activation_probabilities(image, ref_a[e]);
            }).else_([&](auto f){
                auto patches = mat_to_patches(conf, dataset.word_images.at(training_image + ".png"), false);

                ref_a[e].reserve(patches.size());

                for(std::size_t i = 0; i < patches.size(); ++i){
                    ref_a[e].push_back(dbn.prepare_one_output());
                    f(dbn).activation_probabilities(patches[i], ref_a[e][i]);
                }
            });

#ifdef LOCAL_LINEAR_SCALING
            local_linear_feature_scaling(ref_a[e]);
#endif

#ifdef LOCAL_MEAN_SCALING
            local_mean_feature_scaling(ref_a[e]);
#endif

#ifdef GLOBAL_MEAN_SCALING
            auto scale = global_mean_scaling(ref_a[e], conf, false);
#endif

#ifdef GLOBAL_LINEAR_SCALING
            auto scale = global_linear_scaling(ref_a[e], conf, false);
#endif

#if defined(GLOBAL_MEAN_SCALING) || defined(GLOBAL_LINEAR_SCALING)
            for(std::size_t i = 0; i < ref_a.size(); ++i){
                for(std::size_t f = 0; f < ref_a[i].size(); ++f){
                    ref_a[e][i][f] = scale(ref_a[i][f], conf.scale_a[f], conf.scale_b[f]);
                }
            }
#endif
        });

    return ref_a;
}

template<typename Dataset, typename Ref, typename Features>
std::vector<std::pair<std::string, weight>> compute_distances(
        thread_pool& pool, const Dataset& dataset, Features& test_features_a, Ref& ref_a,
        names training_images, names test_image_names, parameters parameters){
    std::vector<std::pair<std::string, weight>> diffs_a(test_image_names.size());

    cpp::parallel_foreach_i(pool, test_image_names.begin(), test_image_names.end(),
        [&](auto& test_image, std::size_t t){
            auto t_size = dataset.word_images.at(test_image).size().width;

            double best_diff_a = 100000000.0;

            for(std::size_t i = 0; i < ref_a.size(); ++i){
                auto ref_size = dataset.word_images.at(training_images[i] + ".png").size().width;

                double diff_a;
                auto ratio = static_cast<double>(ref_size) / t_size;
                if(ratio > 2.0 || ratio < 0.5){
                    diff_a = 100000000.0;
                } else {
                    diff_a = dtw_distance(ref_a[i], test_features_a[t], true, parameters.sc_band);
                }

                best_diff_a = std::min(best_diff_a, diff_a);
            }

            diffs_a[t] = std::make_pair(std::string(test_image.begin(), test_image.end() - 4), best_diff_a);
        });

    return diffs_a;
}

template<typename Dataset, typename Set>
std::vector<std::vector<std::string>> select_keywords(const Dataset& dataset, const Set& set, names train_word_names, names test_image_names){
    std::vector<std::vector<std::string>> keywords;

    for(std::size_t k = 0; k < set.keywords.size(); ++k){
        auto& keyword = set.keywords[k];

        bool found = false;

        for(auto& labels : dataset.word_labels){
            if(keyword == labels.second && std::find(train_word_names.begin(), train_word_names.end(), labels.first) != train_word_names.end()){
                found = true;
                break;
            }
        }

        if(found){
            auto total_test = std::count_if(test_image_names.begin(), test_image_names.end(),
                [&dataset, &keyword](auto& i){ return dataset.word_labels.at({i.begin(), i.end() - 4}) == keyword; });

            if(total_test > 0){
                keywords.push_back(keyword);
            }
        }
    }

    std::cout << "Selected " << keywords.size() << " keyword out of " << set.keywords.size() << std::endl;

    return keywords;
}

template<bool D_P, typename TF, typename KV, typename Dataset, typename DBN>
double evaluate_patches_param(thread_pool& pool, TF& test_features_a, KV& keywords, const Dataset& dataset, config& conf, const DBN& dbn, names train_word_names, names test_image_names, parameters parameters){
    // 2. Evaluate the performances

    std::vector<double> ap(keywords.size());

    for(std::size_t k = 0; k < keywords.size(); ++k){
        auto& keyword = keywords[k];

        // a) Select the training images

        auto training_images = select_training_images(dataset, keyword, train_word_names);

        // b) Compute the reference features

        auto ref_a = compute_reference<D_P>(pool, dataset, dbn, conf, training_images);

        // c) Compute the distances

        auto diffs_a = compute_distances(pool, dataset, test_features_a, ref_a, training_images, test_image_names, parameters);

        // d) Update the local stats

        update_stats_light(k, dataset, keyword, diffs_a, ap, test_image_names);
    }

    double mean_ap = std::accumulate(ap.begin(), ap.end(), 0.0) / ap.size();

    return mean_ap;
}

template<bool D_P, typename Dataset, typename Set, typename DBN>
void optimize_parameters(const Dataset& dataset, const Set& set, config& conf, const DBN& dbn, names train_word_names, names test_image_names, parameters& param){
    std::vector<double> sc_band_values;

    for(double sc = 0.005; sc < 0.03; sc += 0.001){
        sc_band_values.push_back(sc);
    }

    for(double sc = 0.030; sc < 0.2; sc += 0.005){
        sc_band_values.push_back(sc);
    }

    for(double sc = 0.2; sc <= 0.9; sc += 0.05){
        sc_band_values.push_back(sc);
    }

    std::cout << sc_band_values.size() << " Sikoe-Chiba bands to evaluate" << std::endl;

    thread_pool pool;

    // 0. Select the keywords

    auto keywords = select_keywords(dataset, set, train_word_names, test_image_names);

    // 1. Prepare all the outputs

    auto test_features_a = prepare_outputs<D_P>(pool, dataset, dbn, conf, test_image_names, false);

    double best_mean_ap = 0.0;

    parameters best_param;

    std::size_t i = 0;
    for(auto sc : sc_band_values){
        parameters current_param;
        current_param.sc_band = sc;

        double mean_ap = evaluate_patches_param<D_P>(
            pool, test_features_a, keywords, dataset, conf, dbn, train_word_names, test_image_names, current_param);

        std::cout << "(" << i++ << "/" << sc_band_values.size() << ") sc:" << sc << " map: " << mean_ap << std::endl;

        if(mean_ap > best_mean_ap){
            best_param = current_param;
            best_mean_ap = mean_ap;
        }
    }

    std::cout << "Selected as the best parameters" << std::endl;
    std::cout << "\tsc_band: " << best_param.sc_band << std::endl;

    param = best_param;
}


template<bool D_P, typename Dataset, typename Set, typename DBN>
void evaluate_patches(const Dataset& dataset, const Set& set, config& conf, const DBN& dbn, names train_word_names, names test_image_names, bool training, parameters parameters){
    thread_pool pool;

    // 0. Select the keywords

    auto keywords = select_keywords(dataset, set, train_word_names, test_image_names);

    // 1. Select a folder

    auto result_folder = select_folder("./results/");

    // 2. Generate the rel files

    generate_rel_files(result_folder, dataset, test_image_names, keywords);

    // 3. Prepare all the outputs

    auto test_features_a = prepare_outputs<D_P>(pool, dataset, dbn, conf, test_image_names, training);

    // 4. Evaluate the performances

    std::cout << "Evaluate performance..." << std::endl;

    std::vector<double> eer(keywords.size());
    std::vector<double> ap(keywords.size());

    std::ofstream global_top_stream(result_folder + "/global_top_file");
    std::ofstream local_top_stream(result_folder + "/local_top_file");

    for(std::size_t k = 0; k < keywords.size(); ++k){
        auto& keyword = keywords[k];

        // a) Select the training images

        auto training_images = select_training_images(dataset, keyword, train_word_names);

        // b) Compute the reference features

        auto ref_a = compute_reference<D_P>(pool, dataset, dbn, conf, training_images);

        // c) Compute the distances

        auto diffs_a = compute_distances(pool, dataset, test_features_a, ref_a, training_images, test_image_names, parameters);

        // d) Update the local stats

        update_stats(k, result_folder, dataset, keyword, diffs_a, eer, ap, global_top_stream, local_top_stream, test_image_names);
    }

    std::cout << "... done" << std::endl;

    // 5. Finalize the results

    std::cout << keywords.size() << " keywords evaluated" << std::endl;

    double mean_eer = std::accumulate(eer.begin(), eer.end(), 0.0) / eer.size();
    double mean_ap = std::accumulate(ap.begin(), ap.end(), 0.0) / ap.size();

    std::cout << "Mean EER: " << mean_eer << std::endl;
    std::cout << "Mean AP: " << mean_ap << std::endl;
}

} // end of anonymous namespace

void patches_method(
        const spot_dataset& dataset, const spot_dataset_set& set, config& conf,
        names train_word_names, names train_image_names, names valid_image_names, names test_image_names){
    std::cout << "Use method 2 (patches)" << std::endl;

    auto pretraining_image_names = train_image_names;

    if(conf.all){
        std::cout << "Use all images from pretraining" << std::endl;

        pretraining_image_names.reserve(pretraining_image_names.size() + valid_image_names.size() + test_image_names.size());

        //Copy valid and image into the pretraining set
        std::copy(valid_image_names.begin(), valid_image_names.end(), std::back_inserter(pretraining_image_names));
        std::copy(test_image_names.begin(), test_image_names.end(), std::back_inserter(pretraining_image_names));
    }

    if(conf.half){
        std::cout << "Use a half of the resolution" << std::endl;

        static constexpr const std::size_t K1 = half::K1;
        static constexpr const std::size_t C1 = half::C1;
        static constexpr const std::size_t NF1 = half::NF1;
        static constexpr const std::size_t NV1_1 = half::patch_height;
        static constexpr const std::size_t NV1_2 = half::patch_width;
        static constexpr const std::size_t NH1_1 = NV1_1 - NF1 + 1;
        static constexpr const std::size_t NH1_2 = NV1_2 - NF1 + 1;

        static constexpr const std::size_t K2 = half::K2;
        static constexpr const std::size_t C2 = half::C2;
        static constexpr const std::size_t NF2 = half::NF2;
        static constexpr const std::size_t NV2_1 = NH1_1 / C1;
        static constexpr const std::size_t NV2_2 = NH1_2 / C1;
        static constexpr const std::size_t NH2_1 = NV2_1 - NF2 + 1;
        static constexpr const std::size_t NH2_2 = NV2_2 - NF2 + 1;

        static constexpr const std::size_t K3 = half::K3;
        static constexpr const std::size_t C3 = half::C3;
        static constexpr const std::size_t NF3 = half::NF3;
        static constexpr const std::size_t NV3_1 = NH2_1 / C2;
        static constexpr const std::size_t NV3_2 = NH2_2 / C2;
        static constexpr const std::size_t NH3_1 = NV3_1 - NF3 + 1;
        static constexpr const std::size_t NH3_2 = NV3_2 - NF3 + 1;

#if defined(HALF_CRBM_PMP_1)
        using cdbn_t =
            dll::dbn_desc<
                dll::dbn_layers<
                    dll::conv_rbm_mp_desc<
                        NV1_1, NV1_2, 1, NH1_1 , NH1_2, K1, C1
                        , dll::weight_type<weight>, dll::batch_size<half::B1>
                        , dll::parallel, dll::momentum, dll::weight_decay<half::DT1>
                        , dll::hidden<half::HT1>, dll::sparsity<half::SM1>
                        , dll::dbn_only>::rbm_t
                >
            >::dbn_t;
#elif defined(HALF_CRBM_PMP_2)
        using cdbn_t =
            dll::dbn_desc<
                dll::dbn_layers<
                    dll::conv_rbm_mp_desc<
                        NV1_1, NV1_2, 1, NH1_1 , NH1_2, K1, C1
                        , dll::weight_type<weight>, dll::batch_size<half::B1>
                        , dll::parallel, dll::momentum, dll::weight_decay<half::DT1>
                        , dll::hidden<half::HT1>, dll::sparsity<half::SM1>
                        , dll::dbn_only>::rbm_t
                    , dll::conv_rbm_mp_desc<
                        NV2_1, NV2_2, K1, NH2_1 , NH2_2, K2, C2
                        , dll::weight_type<weight>, dll::batch_size<half::B2>
                        , dll::parallel, dll::momentum, dll::weight_decay<half::DT2>
                        , dll::hidden<half::HT2>, dll::sparsity<half::SM2>
                        , dll::dbn_only>::rbm_t
                >
            >::dbn_t;
#elif defined(HALF_CRBM_PMP_3)
        using cdbn_t =
            dll::dbn_desc<
                dll::dbn_layers<
                    dll::conv_rbm_mp_desc<
                        NV1_1, NV1_2, 1, NH1_1 , NH1_2, K1, C1
                        , dll::weight_type<weight>, dll::batch_size<half::B1>
                        , dll::parallel, dll::momentum, dll::weight_decay<half::DT1>
                        , dll::hidden<half::HT1>, dll::sparsity<half::SM1>
                        , dll::dbn_only>::rbm_t
                    , dll::conv_rbm_mp_desc<
                        NV2_1, NV2_2, K1, NH2_1 , NH2_2, K2, C2
                        , dll::weight_type<weight>, dll::batch_size<half::B2>
                        , dll::parallel, dll::momentum, dll::weight_decay<half::DT2>
                        , dll::hidden<half::HT2>, dll::sparsity<half::SM2>
                        , dll::dbn_only>::rbm_t
                    , dll::conv_rbm_mp_desc<
                        NV3_1, NV3_2, K2, NH3_1 , NH3_2, K3, C3
                        , dll::weight_type<weight>, dll::batch_size<half::B3>
                        , dll::parallel, dll::momentum, dll::weight_decay<half::DT3>
                        , dll::hidden<half::HT3>, dll::sparsity<half::SM3>
                        , dll::dbn_only>::rbm_t
                >
            >::dbn_t;
#elif defined(HALF_CRBM_MP_1)
        using cdbn_t =
            dll::dbn_desc<
                dll::dbn_layers<
                    dll::conv_rbm_desc<
                        NV1_1, NV1_2, 1, NH1_1 , NH1_2, K1
                        , dll::weight_type<weight>, dll::batch_size<half::B1>
                        , dll::parallel, dll::momentum, dll::weight_decay<half::DT1>
                        , dll::hidden<half::HT1>, dll::sparsity<half::SM1>
                        , dll::dbn_only>::rbm_t
                    , dll::mp_layer_3d_desc<K1,NH1_1,NH1_2,1,C1,C1>::layer_t
                >, dll::memory, dll::parallel
            >::dbn_t;
#elif defined(HALF_CRBM_MP_2)
        using cdbn_t =
            dll::dbn_desc<
                dll::dbn_layers<
                    dll::conv_rbm_desc<
                        NV1_1, NV1_2, 1, NH1_1 , NH1_2, K1
                        , dll::weight_type<weight>, dll::batch_size<half::B1>
                        , dll::parallel, dll::momentum, dll::weight_decay<half::DT1>
                        , dll::hidden<half::HT1>, dll::sparsity<half::SM1>
                        , dll::dbn_only>::rbm_t
                    , dll::mp_layer_3d_desc<K1,NH1_1,NH1_2,1,C1,C1>::layer_t
                    , dll::conv_rbm_desc<
                        NV2_1, NV2_2, K1, NH2_1 , NH2_2, K2
                        , dll::weight_type<weight>, dll::batch_size<half::B2>
                        , dll::parallel, dll::momentum, dll::weight_decay<half::DT2>
                        , dll::hidden<half::HT2>, dll::sparsity<half::SM2>
                        , dll::dbn_only>::rbm_t
                    , dll::mp_layer_3d_desc<K2,NH2_1,NH2_2,1,C2,C2>::layer_t
                >, dll::memory, dll::parallel
            >::dbn_t;
#elif defined(HALF_CRBM_MP_3)
        using cdbn_t =
            dll::dbn_desc<
                dll::dbn_layers<
                    dll::conv_rbm_desc<
                        NV1_1, NV1_2, 1, NH1_1 , NH1_2, K1
                        , dll::weight_type<weight>, dll::batch_size<half::B1>
                        , dll::parallel, dll::momentum, dll::weight_decay<half::DT1>
                        , dll::hidden<half::HT1>, dll::sparsity<half::SM1>
                        , dll::dbn_only>::rbm_t
                    , dll::mp_layer_3d_desc<K1,NH1_1,NH1_2,1,C1,C1>::layer_t
                    , dll::conv_rbm_desc<
                        NV2_1, NV2_2, K1, NH2_1 , NH2_2, K2
                        , dll::weight_type<weight>, dll::batch_size<half::B2>
                        , dll::parallel, dll::momentum, dll::weight_decay<half::DT2>
                        , dll::hidden<half::HT2>, dll::sparsity<half::SM2>
                        , dll::dbn_only>::rbm_t
                    , dll::mp_layer_3d_desc<K2,NH2_1,NH2_1,1,C2,C2>::layer_t
                    , dll::conv_rbm_desc<
                        NV3_1, NV3_2, K2, NH3_1 , NH3_2, K3
                        , dll::weight_type<weight>, dll::batch_size<half::B3>
                        , dll::parallel, dll::momentum, dll::weight_decay<half::DT3>
                        , dll::hidden<half::HT3>, dll::sparsity<half::SM3>
                        , dll::dbn_only>::rbm_t
                    , dll::mp_layer_3d_desc<K3,NH3_1,NH3_1,1,C3,C3>::layer_t
                >
            >::dbn_t;
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

        cdbn->display();
        std::cout << cdbn->output_size() << " output features" << std::endl;

        constexpr const auto patch_width = half::patch_width;
        constexpr const auto patch_height = half::patch_height;
        constexpr const auto train_stride = half::train_stride;
        constexpr const auto test_stride = half::test_stride;

        std::cout << "patch_height=" << patch_height << std::endl;
        std::cout << "patch_width=" << patch_width << std::endl;
        std::cout << "train_stride=" << train_stride << std::endl;
        std::cout << "test_stride=" << test_stride << std::endl;

        //Pass information to the next passes (evaluation)
        conf.patch_width = patch_width;
        conf.train_stride = train_stride;
        conf.test_stride = test_stride;

        {
            std::vector<etl::dyn_matrix<weight, 3>> training_patches;
            training_patches.reserve(pretraining_image_names.size() * 5);

            std::cout << "Generate patches ..." << std::endl;

            for(auto& name : pretraining_image_names){
                auto patches = mat_to_patches(conf, dataset.word_images.at(name), true);
                std::move(patches.begin(), patches.end(), std::back_inserter(training_patches));
            }

            std::cout << "... done" << std::endl;

            const std::string file_name("method_2_half.dat");

            cdbn->pretrain(training_patches, half::epochs);
            cdbn->store(file_name);
            //cdbn->load(file_name);
        }

        parameters params;
        params.sc_band = 0.1;

        std::cout << "Evaluate on training set" << std::endl;
        evaluate_patches<false>(dataset, set, conf, *cdbn, train_word_names, train_image_names, true, params);

        std::cout << "Optimize parameters" << std::endl;
        optimize_parameters<false>(dataset, set, conf, *cdbn, train_word_names, valid_image_names, params);

        std::cout << "Evaluate on validation set" << std::endl;
        evaluate_patches<false>(dataset, set, conf, *cdbn, train_word_names, valid_image_names, false, params);

        std::cout << "Evaluate on test set" << std::endl;
        evaluate_patches<false>(dataset, set, conf, *cdbn, train_word_names, test_image_names, false, params);

#if HALF_LEVELS < 2
        //Silence some warnings
        cpp_unused(K2);
        cpp_unused(C2);
        cpp_unused(L2);
        cpp_unused(NH2_1);
        cpp_unused(NH2_2);
#endif

#if HALF_LEVELS < 3
        //Silence some warnings
        cpp_unused(K3);
        cpp_unused(C3);
        cpp_unused(L3);
        cpp_unused(NH3_1);
        cpp_unused(NH3_2);
#endif
    } else if(conf.third){
        std::cout << "Use a third of the resolution" << std::endl;

        static constexpr const std::size_t K1 = third::K1;
        static constexpr const std::size_t C1 = third::C1;
        static constexpr const std::size_t NF1 = third::NF1;
        static constexpr const std::size_t NV1_1 = third::patch_height;
        static constexpr const std::size_t NV1_2 = third::patch_width;
        static constexpr const std::size_t NH1_1 = NV1_1 - NF1 + 1;
        static constexpr const std::size_t NH1_2 = NV1_2 - NF1 + 1;

        static constexpr const std::size_t K2 = third::K2;
        static constexpr const std::size_t C2 = third::C2;
        static constexpr const std::size_t NF2 = third::NF2;
        static constexpr const std::size_t NV2_1 = NH1_1 / C1;
        static constexpr const std::size_t NV2_2 = NH1_2 / C1;
        static constexpr const std::size_t NH2_1 = NV2_1 - NF2 + 1;
        static constexpr const std::size_t NH2_2 = NV2_2 - NF2 + 1;

        static constexpr const std::size_t K3 = third::K3;
        static constexpr const std::size_t C3 = third::C3;
        static constexpr const std::size_t NF3 = third::NF3;
        static constexpr const std::size_t NV3_1 = NH2_1 / C2;
        static constexpr const std::size_t NV3_2 = NH2_2 / C2;
        static constexpr const std::size_t NH3_1 = NV3_1 - NF3 + 1;
        static constexpr const std::size_t NH3_2 = NV3_2 - NF3 + 1;

#if defined(THIRD_CRBM_PMP_1)
        using cdbn_t =
            dll::dbn_desc<
                dll::dbn_layers<
                    dll::conv_rbm_mp_desc<
                        NV1_1, NV1_2, 1, NH1_1 , NH1_2, K1, C1
                        , dll::weight_type<weight>, dll::batch_size<third::B1>
                        , dll::parallel, dll::momentum, dll::weight_decay<third::DT1>
                        , dll::hidden<third::HT1>, dll::sparsity<third::SM1>
                        , dll::dbn_only>::rbm_t
                >
            >::dbn_t;
#elif defined(THIRD_CRBM_PMP_2)
        using cdbn_t =
            dll::dbn_desc<
                dll::dbn_layers<
                    dll::conv_rbm_mp_desc<
                        NV1_1, NV1_2, 1, NH1_1 , NH1_2, K1, C1
                        , dll::weight_type<weight>, dll::batch_size<third::B1>
                        , dll::parallel, dll::momentum, dll::weight_decay<third::DT1>
                        , dll::hidden<third::HT1>, dll::sparsity<third::SM1>
                        , dll::dbn_only>::rbm_t
                    , dll::conv_rbm_mp_desc<
                        NV2_1, NV2_2, K1, NH2_1 , NH2_2, K2, C2
                        , dll::weight_type<weight>, dll::batch_size<third::B2>
                        , dll::parallel, dll::momentum, dll::weight_decay<third::DT2>
                        , dll::hidden<third::HT2>, dll::sparsity<third::SM2>
                        , dll::dbn_only>::rbm_t
                >
            >::dbn_t;
#elif defined(THIRD_CRBM_PMP_3)
        using cdbn_t =
            dll::dbn_desc<
                dll::dbn_layers<
                    dll::conv_rbm_mp_desc<
                        NV1_1, NV1_2, 1, NH1_1 , NH1_2, K1, C1
                        , dll::weight_type<weight>, dll::batch_size<third::B1>
                        , dll::parallel, dll::momentum, dll::weight_decay<third::DT1>
                        , dll::hidden<third::HT1>, dll::sparsity<third::SM1>
                        , dll::dbn_only>::rbm_t
                    , dll::conv_rbm_mp_desc<
                        NV2_1, NV2_2, K1, NH2_1 , NH2_2, K2, C2
                        , dll::weight_type<weight>, dll::batch_size<third::B2>
                        , dll::parallel, dll::momentum, dll::weight_decay<third::DT2>
                        , dll::hidden<third::HT2>, dll::sparsity<third::SM2>
                        , dll::dbn_only>::rbm_t
                    , dll::conv_rbm_mp_desc<
                        NV3_1, NV3_2, K2, NH3_1 , NH3_2, K3, C3
                        , dll::weight_type<weight>, dll::batch_size<third::B3>
                        , dll::parallel, dll::momentum, dll::weight_decay<third::DT3>
                        , dll::hidden<third::HT3>, dll::sparsity<third::SM3>
                        , dll::dbn_only>::rbm_t
                >
            >::dbn_t;
#elif defined(THIRD_CRBM_MP_1)
        using cdbn_t =
            dll::dbn_desc<
                dll::dbn_layers<
                    dll::conv_rbm_desc<
                        NV1_1, NV1_2, 1, NH1_1 , NH1_2, K1
                        , dll::weight_type<weight>, dll::batch_size<third::B1>
                        , dll::parallel, dll::momentum, dll::weight_decay<third::DT1>
                        , dll::hidden<third::HT1>, dll::sparsity<third::SM1>
                        , dll::dbn_only>::rbm_t
                    , dll::mp_layer_3d_desc<K1,NH1_1,NH1_2,1,C1,C1>::layer_t
                >, dll::memory, dll::parallel
            >::dbn_t;
#elif defined(THIRD_CRBM_MP_2)
        using cdbn_t =
            dll::dbn_desc<
                dll::dbn_layers<
                    dll::conv_rbm_desc<
                        NV1_1, NV1_2, 1, NH1_1 , NH1_2, K1
                        , dll::weight_type<weight>, dll::batch_size<third::B1>
                        , dll::parallel, dll::momentum, dll::weight_decay<third::DT1>
                        , dll::hidden<third::HT1>, dll::sparsity<third::SM1>
                        , dll::dbn_only>::rbm_t
                    , dll::mp_layer_3d_desc<K1,NH1_1,NH1_2,1,C1,C1>::layer_t
                    , dll::conv_rbm_desc<
                        NV2_1, NV2_2, K1, NH2_1 , NH2_2, K2
                        , dll::weight_type<weight>, dll::batch_size<third::B2>
                        , dll::parallel, dll::momentum, dll::weight_decay<third::DT2>
                        , dll::hidden<third::HT2>, dll::sparsity<third::SM2>
                        , dll::dbn_only>::rbm_t
                    , dll::mp_layer_3d_desc<K2,NH2_1,NH2_2,1,C2,C2>::layer_t
                >, dll::memory, dll::parallel
            >::dbn_t;
#elif defined(THIRD_CRBM_MP_3)
        using cdbn_t =
            dll::dbn_desc<
                dll::dbn_layers<
                    dll::conv_rbm_desc<
                        NV1_1, NV1_2, 1, NH1_1 , NH1_2, K1
                        , dll::weight_type<weight>, dll::batch_size<third::B1>
                        , dll::parallel, dll::momentum, dll::weight_decay<third::DT1>
                        , dll::hidden<third::HT1>, dll::sparsity<third::SM1>
                        , dll::dbn_only>::rbm_t
                    , dll::mp_layer_3d_desc<K1,NH1_1,NH1_2,1,C1,C1>::layer_t
                    , dll::conv_rbm_desc<
                        NV2_1, NV2_2, K1, NH2_1 , NH2_2, K2
                        , dll::weight_type<weight>, dll::batch_size<third::B2>
                        , dll::parallel, dll::momentum, dll::weight_decay<third::DT2>
                        , dll::hidden<third::HT2>, dll::sparsity<third::SM2>
                        , dll::dbn_only>::rbm_t
                    , dll::mp_layer_3d_desc<K2,NH2_1,NH2_1,1,C2,C2>::layer_t
                    , dll::conv_rbm_desc<
                        NV3_1, NV3_2, K2, NH3_1 , NH3_2, K3
                        , dll::weight_type<weight>, dll::batch_size<third::B3>
                        , dll::parallel, dll::momentum, dll::weight_decay<third::DT3>
                        , dll::hidden<third::HT3>, dll::sparsity<third::SM3>
                        , dll::dbn_only>::rbm_t
                    , dll::mp_layer_3d_desc<K3,NH3_1,NH3_1,1,C3,C3>::layer_t
                >
            >::dbn_t;
#elif defined(THIRD_RBM_1)
        using cdbn_t =
            dll::dbn_desc<
                dll::dbn_layers<
                    dll::rbm_desc<
                        NV1_1 * NV1_2 * 1, NF1
                        , dll::weight_type<weight>, dll::batch_size<third::B1>
                        , dll::momentum, dll::weight_decay<third::DT1>
                        , dll::hidden<third::HT1>, dll::sparsity<third::SM1>
                        , dll::dbn_only>::rbm_t
                >
            >::dbn_t;
#elif defined(THIRD_RBM_2)
        using cdbn_t =
            dll::dbn_desc<
                dll::dbn_layers<
                    dll::rbm_desc<
                        NV1_1 * NV1_2 * 1, NF1
                        , dll::weight_type<weight>, dll::batch_size<third::B1>
                        , dll::momentum, dll::weight_decay<third::DT1>
                        , dll::hidden<third::HT1>, dll::sparsity<third::SM1>
                        , dll::dbn_only>::rbm_t
                    , dll::rbm_desc<
                        NF1, NF2
                        , dll::weight_type<weight>, dll::batch_size<third::B2>
                        , dll::momentum, dll::weight_decay<third::DT2>
                        , dll::hidden<third::HT2>, dll::sparsity<third::SM2>
                        , dll::dbn_only>::rbm_t
                >
            >::dbn_t;
#elif defined(THIRD_RBM_3)
        using cdbn_t =
            dll::dbn_desc<
                dll::dbn_layers<
                    dll::rbm_desc<
                        NV1_1 * NV1_2 * 1, NF1
                        , dll::weight_type<weight>, dll::batch_size<third::B1>
                        , dll::parallel, dll::momentum, dll::weight_decay<third::DT1>
                        , dll::hidden<third::HT1>, dll::sparsity<third::SM1>
                        , dll::dbn_only>::rbm_t
                    , dll::rbm_desc<
                        NF1, NF2
                        , dll::weight_type<weight>, dll::batch_size<third::B2>
                        , dll::parallel, dll::momentum, dll::weight_decay<third::DT2>
                        , dll::hidden<third::HT2>, dll::sparsity<third::SM2>
                        , dll::dbn_only>::rbm_t
                    , dll::rbm_desc<
                        NF2, NF3
                        , dll::weight_type<weight>, dll::batch_size<third::B3>
                        , dll::parallel, dll::momentum, dll::weight_decay<third::DT3>
                        , dll::hidden<third::HT3>, dll::sparsity<third::SM3>
                        , dll::dbn_only>::rbm_t
                >
            >::dbn_t;
#else
        static_assert(false, "No architecture has been selected");
#endif

#if defined(THIRD_CRBM_MP_1) || defined(THIRD_CRBM_MP_2) || defined(THIRD_CRBM_MP_3)
        //Max pooling layers models have less layers
        constexpr const std::size_t L1 = 0;
        constexpr const std::size_t L2 = 2;
        constexpr const std::size_t L3 = 4;
#else
        constexpr const std::size_t L1 = 0;
        constexpr const std::size_t L2 = 1;
        constexpr const std::size_t L3 = 2;
#endif

        auto cdbn = std::make_unique<cdbn_t>();

        // Level 1
        third::rate_0(cdbn->template layer_get<L1>().learning_rate);
        third::momentum_0(cdbn->template layer_get<L1>().initial_momentum, cdbn->template layer_get<L1>().final_momentum);
        third::wd_l1_0(cdbn->template layer_get<L1>().l1_weight_cost);
        third::wd_l2_0(cdbn->template layer_get<L1>().l2_weight_cost);
        third::pbias_0(cdbn->template layer_get<L1>().pbias);
        third::pbias_lambda_0(cdbn->template layer_get<L1>().pbias_lambda);
        third::sparsity_target_0(cdbn->template layer_get<L1>().sparsity_target);

#if THIRD_LEVELS >= 2
        //Level 2
        third::rate_1(cdbn->template layer_get<L2>().learning_rate);
        third::momentum_1(cdbn->template layer_get<L2>().initial_momentum, cdbn->template layer_get<L2>().final_momentum);
        third::wd_l1_1(cdbn->template layer_get<L2>().l1_weight_cost);
        third::wd_l2_1(cdbn->template layer_get<L2>().l2_weight_cost);
        third::pbias_1(cdbn->template layer_get<L2>().pbias);
        third::pbias_lambda_1(cdbn->template layer_get<L2>().pbias_lambda);
        third::sparsity_target_1(cdbn->template layer_get<L1>().sparsity_target);
#endif

#if THIRD_LEVELS >= 3
        //Level 3
        third::rate_2(cdbn->template layer_get<L3>().learning_rate);
        third::momentum_2(cdbn->template layer_get<L3>().initial_momentum, cdbn->template layer_get<L3>().final_momentum);
        third::wd_l1_2(cdbn->template layer_get<L3>().l1_weight_cost);
        third::wd_l2_2(cdbn->template layer_get<L3>().l2_weight_cost);
        third::pbias_2(cdbn->template layer_get<L3>().pbias);
        third::pbias_lambda_2(cdbn->template layer_get<L3>().pbias_lambda);
        third::sparsity_target_2(cdbn->template layer_get<L1>().sparsity_target);
#endif

        cdbn->display();
        std::cout << cdbn->output_size() << " output features" << std::endl;

        constexpr const auto patch_width = third::patch_width;
        constexpr const auto patch_height = third::patch_height;
        constexpr const auto train_stride = third::train_stride;
        constexpr const auto test_stride = third::test_stride;

        std::cout << "patch_height=" << patch_height << std::endl;
        std::cout << "patch_width=" << patch_width << std::endl;
        std::cout << "train_stride=" << train_stride << std::endl;
        std::cout << "test_stride=" << test_stride << std::endl;

        //Pass information to the next passes (evaluation)
        conf.patch_width = patch_width;
        conf.train_stride = train_stride;
        conf.test_stride = test_stride;

        //Train the DBN
        {
            std::vector<etl::dyn_matrix<weight, 3>> training_patches;
            training_patches.reserve(pretraining_image_names.size() * 5);

            std::cout << "Generate patches ..." << std::endl;

            for(auto& name : pretraining_image_names){
                auto patches = mat_to_patches(conf, dataset.word_images.at(name), true);
                std::move(patches.begin(), patches.end(), std::back_inserter(training_patches));
            }

            std::cout << "... done" << std::endl;

            const std::string file_name("method_2_third.dat");

            cdbn->pretrain(training_patches, third::epochs);
            cdbn->store(file_name);
            //cdbn->load(file_name);
        }

        parameters params;
        params.sc_band = 0.1;

        std::cout << "Evaluate on training set" << std::endl;
        evaluate_patches<false>(dataset, set, conf, *cdbn, train_word_names, train_image_names, true, params);

        std::cout << "Optimize parameters" << std::endl;
        optimize_parameters<false>(dataset, set, conf, *cdbn, train_word_names, valid_image_names, params);

        std::cout << "Evaluate on validation set" << std::endl;
        evaluate_patches<false>(dataset, set, conf, *cdbn, train_word_names, valid_image_names, false, params);

        std::cout << "Evaluate on test set" << std::endl;
        evaluate_patches<false>(dataset, set, conf, *cdbn, train_word_names, test_image_names, false, params);

#if defined(THIRD_RBM_1) || defined(THIRD_RBM_2) || defined(THIRD_RBM_3)
        //Silence some warnings
        cpp_unused(K1);
        cpp_unused(K2);
        cpp_unused(K3);
#endif

#if THIRD_LEVELS < 2
        //Silence some warnings
        cpp_unused(K2);
        cpp_unused(C2);
        cpp_unused(L2);
        cpp_unused(NH2_1);
        cpp_unused(NH2_2);
#endif

#if THIRD_LEVELS < 3
        //Silence some warnings
        cpp_unused(K3);
        cpp_unused(C3);
        cpp_unused(L3);
        cpp_unused(NH3_1);
        cpp_unused(NH3_2);
#endif
    } else {
        std::cout << "Use full resolution" << std::endl;

        static constexpr const std::size_t K1 = full::K1;
        static constexpr const std::size_t C1 = full::C1;
        static constexpr const std::size_t NF1 = full::NF1;
        static constexpr const std::size_t NV1_1 = full::patch_height;
        static constexpr const std::size_t NV1_2 = full::patch_width;
        static constexpr const std::size_t NH1_1 = NV1_1 - NF1 + 1;
        static constexpr const std::size_t NH1_2 = NV1_2 - NF1 + 1;

        static constexpr const std::size_t K2 = full::K2;
        static constexpr const std::size_t C2 = full::C2;
        static constexpr const std::size_t NF2 = full::NF2;
        static constexpr const std::size_t NV2_1 = NH1_1 / C1;
        static constexpr const std::size_t NV2_2 = NH1_2 / C1;
        static constexpr const std::size_t NH2_1 = NV2_1 - NF2 + 1;
        static constexpr const std::size_t NH2_2 = NV2_2 - NF2 + 1;

        static constexpr const std::size_t K3 = full::K3;
        static constexpr const std::size_t C3 = full::C3;
        static constexpr const std::size_t NF3 = full::NF3;
        static constexpr const std::size_t NV3_1 = NH2_1 / C2;
        static constexpr const std::size_t NV3_2 = NH2_2 / C2;
        static constexpr const std::size_t NH3_1 = NV3_1 - NF3 + 1;
        static constexpr const std::size_t NH3_2 = NV3_2 - NF3 + 1;

#if defined(FULL_CRBM_PMP_1)
        using cdbn_t =
            dll::dbn_desc<
                dll::dbn_layers<
                    dll::conv_rbm_mp_desc<
                        NV1_1, NV1_2, 1, NH1_1 , NH1_2, K1, C1
                        , dll::weight_type<weight>, dll::batch_size<full::B1>
                        , dll::parallel, dll::momentum, dll::weight_decay<full::DT1>
                        , dll::hidden<full::HT1>, dll::sparsity<full::SM1>
                        , dll::dbn_only>::rbm_t
                >
            >::dbn_t;
#elif defined(FULL_CRBM_PMP_2)
        using cdbn_t =
            dll::dbn_desc<
                dll::dbn_layers<
                      dll::patches_layer_padh_desc<full::patch_width, full::patch_height, 1, full::train_stride, 1>::layer_t
                    , dll::conv_rbm_mp_desc<
                        NV1_1, NV1_2, 1, NH1_1 , NH1_2, K1, C1
                        , dll::weight_type<weight>, dll::batch_size<full::B1>
                        , dll::parallel, dll::momentum, dll::weight_decay<full::DT1>
                        , dll::hidden<full::HT1>, dll::sparsity<full::SM1>
                        , dll::dbn_only>::rbm_t
                    , dll::conv_rbm_mp_desc<
                        NV2_1, NV2_2, K1, NH2_1 , NH2_2, K2, C2
                        , dll::weight_type<weight>, dll::batch_size<full::B2>
                        , dll::parallel, dll::momentum, dll::weight_decay<full::DT2>
                        , dll::hidden<full::HT2>, dll::sparsity<full::SM2>
                        , dll::dbn_only>::rbm_t
                >,
                dll::memory,
                dll::parallel,
                dll::batch_size<5>
            >::dbn_t;
#elif defined(FULL_CRBM_PMP_3)
        using cdbn_t =
            dll::dbn_desc<
                dll::dbn_layers<
                    dll::conv_rbm_mp_desc<
                        NV1_1, NV1_2, 1, NH1_1 , NH1_2, K1, C1
                        , dll::weight_type<weight>, dll::batch_size<full::B1>
                        , dll::parallel, dll::momentum, dll::weight_decay<full::DT1>
                        , dll::hidden<full::HT1>, dll::sparsity<full::SM1>
                        , dll::dbn_only>::rbm_t
                    , dll::conv_rbm_mp_desc<
                        NV2_1, NV2_2, K1, NH2_1 , NH2_2, K2, C2
                        , dll::weight_type<weight>, dll::batch_size<full::B2>
                        , dll::parallel, dll::momentum, dll::weight_decay<full::DT2>
                        , dll::hidden<full::HT2>, dll::sparsity<full::SM2>
                        , dll::dbn_only>::rbm_t
                    , dll::conv_rbm_mp_desc<
                        NV3_1, NV3_2, K2, NH3_1 , NH3_2, K3, C3
                        , dll::weight_type<weight>, dll::batch_size<full::B3>
                        , dll::parallel, dll::momentum, dll::weight_decay<full::DT3>
                        , dll::hidden<full::HT3>, dll::sparsity<full::SM3>
                        , dll::dbn_only>::rbm_t
                >
            >::dbn_t;
#elif defined(FULL_CRBM_MP_1)
        using cdbn_t =
            dll::dbn_desc<
                dll::dbn_layers<
                    dll::conv_rbm_desc<
                        NV1_1, NV1_2, 1, NH1_1 , NH1_2, K1
                        , dll::weight_type<weight>, dll::batch_size<full::B1>
                        , dll::parallel, dll::momentum, dll::weight_decay<full::DT1>
                        , dll::hidden<full::HT1>, dll::sparsity<full::SM1>
                        , dll::dbn_only>::rbm_t
                    , dll::mp_layer_3d_desc<K1,NH1_1,NH1_2,1,C1,C1>::layer_t
                >, dll::memory, dll::parallel
            >::dbn_t;
#elif defined(FULL_CRBM_MP_2)
        using cdbn_t =
            dll::dbn_desc<
                dll::dbn_layers<
                    dll::conv_rbm_desc<
                        NV1_1, NV1_2, 1, NH1_1 , NH1_2, K1
                        , dll::weight_type<weight>, dll::batch_size<full::B1>
                        , dll::parallel, dll::momentum, dll::weight_decay<full::DT1>
                        , dll::hidden<full::HT1>, dll::sparsity<full::SM1>
                        , dll::dbn_only>::rbm_t
                    , dll::mp_layer_3d_desc<K1,NH1_1,NH1_2,1,C1,C1>::layer_t
                    , dll::conv_rbm_desc<
                        NV2_1, NV2_2, K1, NH2_1 , NH2_2, K2
                        , dll::weight_type<weight>, dll::batch_size<full::B2>
                        , dll::parallel, dll::momentum, dll::weight_decay<full::DT2>
                        , dll::hidden<full::HT2>, dll::sparsity<full::SM2>
                        , dll::dbn_only>::rbm_t
                    , dll::mp_layer_3d_desc<K2,NH2_1,NH2_2,1,C2,C2>::layer_t
                >, dll::memory, dll::parallel
            >::dbn_t;
#elif defined(FULL_CRBM_MP_3)
        using cdbn_t =
            dll::dbn_desc<
                dll::dbn_layers<
                    dll::conv_rbm_desc<
                        NV1_1, NV1_2, 1, NH1_1 , NH1_2, K1
                        , dll::weight_type<weight>, dll::batch_size<full::B1>
                        , dll::parallel, dll::momentum, dll::weight_decay<full::DT1>
                        , dll::hidden<full::HT1>, dll::sparsity<full::SM1>
                        , dll::dbn_only>::rbm_t
                    , dll::mp_layer_3d_desc<K1,NH1_1,NH1_2,1,C1,C1>::layer_t
                    , dll::conv_rbm_desc<
                        NV2_1, NV2_2, K1, NH2_1 , NH2_2, K2
                        , dll::weight_type<weight>, dll::batch_size<full::B2>
                        , dll::parallel, dll::momentum, dll::weight_decay<full::DT2>
                        , dll::hidden<full::HT2>, dll::sparsity<full::SM2>
                        , dll::dbn_only>::rbm_t
                    , dll::mp_layer_3d_desc<K2,NH2_1,NH2_1,1,C2,C2>::layer_t
                    , dll::conv_rbm_desc<
                        NV3_1, NV3_2, K2, NH3_1 , NH3_2, K3
                        , dll::weight_type<weight>, dll::batch_size<full::B3>
                        , dll::parallel, dll::momentum, dll::weight_decay<full::DT3>
                        , dll::hidden<full::HT3>, dll::sparsity<full::SM3>
                        , dll::dbn_only>::rbm_t
                    , dll::mp_layer_3d_desc<K3,NH3_1,NH3_1,1,C3,C3>::layer_t
                >
            >::dbn_t;
#else
        static_assert(false, "No architecture has been selected");
#endif

#if defined(FULL_CRBM_PMP_2)
        //Probabilistic max poolin models have less layers
        constexpr const std::size_t L1 = 1;
        constexpr const std::size_t L2 = 2;
        constexpr const std::size_t L3 = 3;
#elif defined(FULL_CRBM_PMP_1) || defined(FULL_CRBM_PMP_3)
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

        cdbn->display();
        std::cout << cdbn->output_size() << " output features" << std::endl;

        constexpr const auto patch_width = full::patch_width;
        constexpr const auto patch_height = full::patch_height;
        constexpr const auto train_stride = full::train_stride;
        constexpr const auto test_stride = full::test_stride;

        std::cout << "patch_height=" << patch_height << std::endl;
        std::cout << "patch_width=" << patch_width << std::endl;
        std::cout << "train_stride=" << train_stride << std::endl;
        std::cout << "test_stride=" << test_stride << std::endl;

        //Pass information to the next passes (evaluation)
        conf.patch_width = patch_width;
        conf.train_stride = train_stride;
        conf.test_stride = test_stride;

        //1. Pretraining
        {
            const std::string file_name("method_2_full.dat");

#ifdef FULL_CRBM_PMP_2
            std::vector<etl::dyn_matrix<weight, 3>> training_images;
            training_images.reserve(pretraining_image_names.size());

            std::cout << "Generate images ..." << std::endl;

            for(auto& name : pretraining_image_names){
                training_images.push_back(mat_for_patches(conf, dataset.word_images.at(name)));
            }

            std::cout << "... done" << std::endl;

            cdbn->pretrain(training_images, full::epochs);
            cdbn->store(file_name);
            //cdbn->load(file_name);
#else
            patch_iterator it(conf, dataset, pretraining_image_names);
            patch_iterator end(conf, dataset, pretraining_image_names, pretraining_image_names.size());

            cdbn->pretrain(it, end, full::epochs);
            cdbn->store(file_name);
            //cdbn->load(file_name);
#endif
        }

        //2. Evaluation

        parameters params;
        params.sc_band = 0.1;

        std::cout << "Evaluate on training set" << std::endl;
        evaluate_patches<true>(dataset, set, conf, *cdbn, train_word_names, train_image_names, true, params);

        std::cout << "Optimize parameters" << std::endl;
        optimize_parameters<true>(dataset, set, conf, *cdbn, train_word_names, valid_image_names, params);

        std::cout << "Evaluate on validation set" << std::endl;
        evaluate_patches<true>(dataset, set, conf, *cdbn, train_word_names, valid_image_names, false, params);

        std::cout << "Evaluate on test set" << std::endl;
        evaluate_patches<true>(dataset, set, conf, *cdbn, train_word_names, test_image_names, false, params);

#if FULL_LEVELS < 2
        //Silence some warnings
        cpp_unused(K2);
        cpp_unused(C2);
        cpp_unused(L2);
        cpp_unused(NH2_1);
        cpp_unused(NH2_2);
#endif

#if FULL_LEVELS < 3
        //Silence some warnings
        cpp_unused(K3);
        cpp_unused(C3);
        cpp_unused(L3);
        cpp_unused(NH3_1);
        cpp_unused(NH3_2);
#endif
    }
}
