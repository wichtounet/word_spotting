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

#if defined(THIRD_CRBM_PMP_1) || defined(THIRD_CRBM_MP_1)
#define THIRD_LEVELS 1
#endif

#if defined(THIRD_CRBM_PMP_2) || defined(THIRD_CRBM_MP_2)
#define THIRD_LEVELS 2
#endif

#if defined(THIRD_CRBM_PMP_3) || defined(THIRD_CRBM_MP_3)
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

struct patch_iterator : std::iterator<std::input_iterator_tag, etl::dyn_matrix<weight>> {
    config& conf;
    const washington_dataset& dataset;
    const std::vector<std::string>& image_names;

    std::size_t current_image = 0;
    std::vector<etl::dyn_matrix<weight>> patches;
    std::size_t current_patch = 0;

    patch_iterator(config& conf, const washington_dataset& dataset, const std::vector<std::string>& image_names, std::size_t i = 0)
            : conf(conf), dataset(dataset), image_names(image_names), current_image(i) {
        if(current_image < image_names.size()){
            patches = mat_to_patches(conf, dataset.word_images.at(image_names[current_image]));
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
    etl::dyn_matrix<weight>& operator*(){
        return patches[current_patch];
    }

    etl::dyn_matrix<weight>* operator->(){
        return &patches[current_patch];
    }

    patch_iterator operator++(){
        if(current_patch == patches.size() - 1){
            ++current_image;
            current_patch = 0;

            if(current_image < image_names.size()){
                patches = mat_to_patches(conf, dataset.word_images.at(image_names[current_image]));
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

template<typename Dataset, typename Set, typename DBN>
void evaluate_patches(const Dataset& dataset, const Set& set, config& conf, const DBN& dbn, const std::vector<std::string>& train_word_names, const std::vector<std::string>& test_image_names, bool training){
    //Get some sizes

    const std::size_t patch_height = HEIGHT / conf.downscale;
    const std::size_t patch_width = conf.patch_width;

    auto result_folder = select_folder("./results/");

    generate_rel_files(result_folder, dataset, set, test_image_names);

    std::cout << "Prepare the outputs ..." << std::endl;

    std::vector<std::vector<etl::dyn_matrix<weight, 3>>> test_features_a(test_image_names.size());

    cpp::default_thread_pool<> pool;

    cpp::parallel_foreach_i(pool, test_image_names.begin(), test_image_names.end(),
        [&,patch_height,patch_width](auto& test_image, std::size_t i){
            auto& vec = test_features_a[i];

            auto patches = mat_to_patches(conf, dataset.word_images.at(test_image));

            for(auto& patch : patches){
                vec.push_back(dbn.prepare_one_output());
                dbn.activation_probabilities(patch, vec.back());
            }

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

    std::cout << "Evaluate performance..." << std::endl;

    std::size_t evaluated = 0;

    std::vector<double> eer(set.keywords.size());
    std::vector<double> ap(set.keywords.size());

    std::ofstream global_top_stream(result_folder + "/global_top_file");
    std::ofstream local_top_stream(result_folder + "/local_top_file");

    for(std::size_t k = 0; k < set.keywords.size(); ++k){
        auto& keyword = set.keywords[k];

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

        auto patches = mat_to_patches(conf, dataset.word_images.at(training_image + ".png"));

        std::vector<etl::dyn_matrix<weight, 3>> ref_a;

        for(auto& patch :patches){
            ref_a.push_back(dbn.prepare_one_output());
            dbn.activation_probabilities(patch, ref_a.back());
        }

#ifdef LOCAL_LINEAR_SCALING
        local_linear_feature_scaling(ref_a);
#endif

#ifdef LOCAL_MEAN_SCALING
        local_mean_feature_scaling(ref_a);
#endif

#if defined(GLOBAL_MEAN_SCALING) || defined(GLOBAL_LINEAR_SCALING)
        for(std::size_t i = 0; i < ref_a.size(); ++i){
            for(std::size_t f = 0; f < ref_a[i].size(); ++f){
                ref_a[i][f] = scale(ref_a[i][f], conf.scale_a[f], conf.scale_b[f]);
            }
        }
#endif

        auto ref_size = dataset.word_images.at(training_image + ".png").size().width;

        std::vector<std::pair<std::string, weight>> diffs_a;

        for(std::size_t t = 0; t < test_image_names.size(); ++t){
            decltype(auto) test_image = test_image_names[t];

            auto t_size = dataset.word_images.at(test_image).size().width;

            double diff_a;

            auto ratio = static_cast<double>(ref_size) / t_size;
            if(ratio > 2.0 || ratio < 0.5){
                diff_a = 100000000.0;
            } else {
                diff_a = dtw_distance(ref_a, test_features_a[t]);
            }

            diffs_a.emplace_back(std::string(test_image.begin(), test_image.end() - 4), diff_a);
        }

        update_stats(k, result_folder, dataset, keyword, diffs_a, eer, ap, global_top_stream, local_top_stream, test_image_names);
    }

    std::cout << "... done" << std::endl;

    std::cout << evaluated << " keywords evaluated" << std::endl;

    double mean_eer = std::accumulate(eer.begin(), eer.end(), 0.0) / eer.size();
    double mean_ap = std::accumulate(ap.begin(), ap.end(), 0.0) / ap.size();

    std::cout << "Mean EER: " << mean_eer << std::endl;
    std::cout << "Mean AP: " << mean_ap << std::endl;
}

} // end of anonymous namespace

void patches_method(const washington_dataset& dataset, const washington_dataset_set& set, config& conf, const std::vector<std::string>& train_word_names, const std::vector<std::string>& train_image_names, const std::vector<std::string>& test_image_names){
    std::cout << "Use method 2 (patches)" << std::endl;

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
        half::rate_0(cdbn->template layer<L1>().learning_rate);
        half::momentum_0(cdbn->template layer<L1>().initial_momentum, cdbn->template layer<L1>().final_momentum);
        half::wd_l1_0(cdbn->template layer<L1>().l1_weight_cost);
        half::wd_l2_0(cdbn->template layer<L1>().l2_weight_cost);
        half::pbias_0(cdbn->template layer<L1>().pbias);
        half::pbias_lambda_0(cdbn->template layer<L1>().pbias_lambda);

#if HALF_LEVELS >= 2
        //Level 2
        half::rate_1(cdbn->template layer<L2>().learning_rate);
        half::momentum_1(cdbn->template layer<L2>().initial_momentum, cdbn->template layer<L2>().final_momentum);
        half::wd_l1_1(cdbn->template layer<L2>().l1_weight_cost);
        half::wd_l2_1(cdbn->template layer<L2>().l2_weight_cost);
        half::pbias_1(cdbn->template layer<L2>().pbias);
        half::pbias_lambda_1(cdbn->template layer<L2>().pbias_lambda);
#endif

#if HALF_LEVELS >= 3
        //Level 3
        half::rate_2(cdbn->template layer<L3>().learning_rate);
        half::momentum_2(cdbn->template layer<L3>().initial_momentum, cdbn->template layer<L3>().final_momentum);
        half::wd_l1_2(cdbn->template layer<L3>().l1_weight_cost);
        half::wd_l2_2(cdbn->template layer<L3>().l2_weight_cost);
        half::pbias_2(cdbn->template layer<L3>().pbias);
        half::pbias_lambda_2(cdbn->template layer<L3>().pbias_lambda);
#endif

        cdbn->display();
        std::cout << cdbn->output_size() << " output features" << std::endl;

        constexpr const auto patch_width = half::patch_width;
        constexpr const auto patch_height = half::patch_height;
        constexpr const auto patch_stride = half::patch_stride;

        std::cout << "patch_height=" << patch_height << std::endl;
        std::cout << "patch_width=" << patch_width << std::endl;
        std::cout << "patch_stride=" << patch_stride << std::endl;

        //Pass information to the next passes (evaluation)
        conf.patch_width = patch_width;
        conf.patch_stride = patch_stride;

        std::vector<etl::dyn_matrix<weight>> training_patches;
        training_patches.reserve(train_image_names.size() * 5);

        std::cout << "Generate patches ..." << std::endl;

        for(auto& name : train_image_names){
            auto patches = mat_to_patches(conf, dataset.word_images.at(name));
            std::move(patches.begin(), patches.end(), std::back_inserter(training_patches));
        }

        std::cout << "... done" << std::endl;

        const std::string file_name("method_2_half.dat");

        cdbn->pretrain(training_patches, half::epochs);
        cdbn->store(file_name);
        //cdbn->load(file_name);

        std::cout << "Evaluate on training set" << std::endl;
        evaluate_patches(dataset, set, conf, *cdbn, train_word_names, train_image_names, true);

        std::cout << "Evaluate on test set" << std::endl;
        evaluate_patches(dataset, set, conf, *cdbn, train_word_names, test_image_names, false);

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
#else
        static_assert(false, "No architecture has been selected");
#endif

#if defined(THIRD_CRBM_PMP_1) || defined(THIRD_CRBM_PMP_2) || defined(THIRD_CRBM_PMP_3)
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
        third::rate_0(cdbn->template layer<L1>().learning_rate);
        third::momentum_0(cdbn->template layer<L1>().initial_momentum, cdbn->template layer<L1>().final_momentum);
        third::wd_l1_0(cdbn->template layer<L1>().l1_weight_cost);
        third::wd_l2_0(cdbn->template layer<L1>().l2_weight_cost);
        third::pbias_0(cdbn->template layer<L1>().pbias);
        third::pbias_lambda_0(cdbn->template layer<L1>().pbias_lambda);

#if THIRD_LEVELS >= 2
        //Level 2
        third::rate_1(cdbn->template layer<L2>().learning_rate);
        third::momentum_1(cdbn->template layer<L2>().initial_momentum, cdbn->template layer<L2>().final_momentum);
        third::wd_l1_1(cdbn->template layer<L2>().l1_weight_cost);
        third::wd_l2_1(cdbn->template layer<L2>().l2_weight_cost);
        third::pbias_1(cdbn->template layer<L2>().pbias);
        third::pbias_lambda_1(cdbn->template layer<L2>().pbias_lambda);
#endif

#if THIRD_LEVELS >= 3
        //Level 3
        third::rate_2(cdbn->template layer<L3>().learning_rate);
        third::momentum_2(cdbn->template layer<L3>().initial_momentum, cdbn->template layer<L3>().final_momentum);
        third::wd_l1_2(cdbn->template layer<L3>().l1_weight_cost);
        third::wd_l2_2(cdbn->template layer<L3>().l2_weight_cost);
        third::pbias_2(cdbn->template layer<L3>().pbias);
        third::pbias_lambda_2(cdbn->template layer<L3>().pbias_lambda);
#endif

        cdbn->display();
        std::cout << cdbn->output_size() << " output features" << std::endl;

        constexpr const auto patch_width = third::patch_width;
        constexpr const auto patch_height = third::patch_height;
        constexpr const auto patch_stride = third::patch_stride;

        std::cout << "patch_height=" << patch_height << std::endl;
        std::cout << "patch_width=" << patch_width << std::endl;
        std::cout << "patch_stride=" << patch_stride << std::endl;

        //Pass information to the next passes (evaluation)
        conf.patch_width = patch_width;
        conf.patch_stride = patch_stride;

        std::vector<etl::dyn_matrix<weight>> training_patches;
        training_patches.reserve(train_image_names.size() * 5);

        std::cout << "Generate patches ..." << std::endl;

        for(auto& name : train_image_names){
            auto patches = mat_to_patches(conf, dataset.word_images.at(name));
            std::move(patches.begin(), patches.end(), std::back_inserter(training_patches));
        }

        std::cout << "... done" << std::endl;

        const std::string file_name("method_2_third.dat");

        cdbn->pretrain(training_patches, third::epochs);
        cdbn->store(file_name);
        //cdbn->load(file_name);

        std::cout << "Evaluate on training set" << std::endl;
        evaluate_patches(dataset, set, conf, *cdbn, train_word_names, train_image_names, true);

        std::cout << "Evaluate on test set" << std::endl;
        evaluate_patches(dataset, set, conf, *cdbn, train_word_names, test_image_names, false);

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
                >,
                dll::memory,
                dll::parallel
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
        full::rate_0(cdbn->template layer<L1>().learning_rate);
        full::momentum_0(cdbn->template layer<L1>().initial_momentum, cdbn->template layer<L1>().final_momentum);
        full::wd_l1_0(cdbn->template layer<L1>().l1_weight_cost);
        full::wd_l2_0(cdbn->template layer<L1>().l2_weight_cost);
        full::pbias_0(cdbn->template layer<L1>().pbias);
        full::pbias_lambda_0(cdbn->template layer<L1>().pbias_lambda);

#if FULL_LEVELS >= 2
        //Level 2
        full::rate_1(cdbn->template layer<L2>().learning_rate);
        full::momentum_1(cdbn->template layer<L2>().initial_momentum, cdbn->template layer<L2>().final_momentum);
        full::wd_l1_1(cdbn->template layer<L2>().l1_weight_cost);
        full::wd_l2_1(cdbn->template layer<L2>().l2_weight_cost);
        full::pbias_1(cdbn->template layer<L2>().pbias);
        full::pbias_lambda_1(cdbn->template layer<L2>().pbias_lambda);
#endif

#if FULL_LEVELS >= 3
        //Level 3
        full::rate_2(cdbn->template layer<L3>().learning_rate);
        full::momentum_2(cdbn->template layer<L3>().initial_momentum, cdbn->template layer<L3>().final_momentum);
        full::wd_l1_2(cdbn->template layer<L3>().l1_weight_cost);
        full::wd_l2_2(cdbn->template layer<L3>().l2_weight_cost);
        full::pbias_2(cdbn->template layer<L3>().pbias);
        full::pbias_lambda_2(cdbn->template layer<L3>().pbias_lambda);
#endif

        cdbn->display();
        std::cout << cdbn->output_size() << " output features" << std::endl;

        constexpr const auto patch_width = full::patch_width;
        constexpr const auto patch_height = full::patch_height;
        constexpr const auto patch_stride = full::patch_stride;

        std::cout << "patch_height=" << patch_height << std::endl;
        std::cout << "patch_width=" << patch_width << std::endl;
        std::cout << "patch_stride=" << patch_stride << std::endl;

        //Pass information to the next passes (evaluation)
        conf.patch_width = patch_width;
        conf.patch_stride = patch_stride;

        //1. Pretraining
        {
            patch_iterator it(conf, dataset, train_image_names);
            patch_iterator end(conf, dataset, train_image_names, train_image_names.size());

            const std::string file_name("method_2_full.dat");

            cdbn->pretrain(it, end, 1);
            cdbn->store(file_name);
            //cdbn->load(file_name);
        }

        //2. Evaluation

        std::cout << "Evaluate on training set" << std::endl;
        evaluate_patches(dataset, set, conf, *cdbn, train_word_names, train_image_names, true);

        std::cout << "Evaluate on test set" << std::endl;
        evaluate_patches(dataset, set, conf, *cdbn, train_word_names, test_image_names, false);

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
