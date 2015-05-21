//=======================================================================
// Copyright Baptiste Wicht 2015.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <fstream>

#include "cpp_utils/parallel.hpp"

#include "dll/conv_rbm.hpp"
#include "dll/conv_rbm_mp.hpp"
#include "dll/dbn.hpp"
#include "dll/avgp_layer.hpp"
#include "dll/mp_layer.hpp"
#include "dll/ocv_visualizer.hpp"

#include "etl/print.hpp"
#include "etl/stop.hpp"

#include "config.hpp"
#include "utils.hpp"
#include "washington.hpp" //Dataset handling
#include "dtw.hpp"        //Dynamic time warping
#include "reports.hpp"    //Reports generation utilities
#include "scaling.hpp"      //Scaling functions

//Include methods
#include "standard.hpp"  //Method 0
#include "holistic.hpp"  //Method 1

//The different configurations
#include "config_third.hpp"

#if defined(THIRD_CRBM_PMP_1) || defined(THIRD_CRBM_MP_1)
#define THIRD_LEVELS 1
#endif

#if defined(THIRD_CRBM_PMP_2) || defined(THIRD_CRBM_MP_2)
#define THIRD_LEVELS 2
#endif

#if defined(THIRD_CRBM_PMP_3) || defined(THIRD_CRBM_MP_3)
#define THIRD_LEVELS 3
#endif

#ifndef THIRD_LEVELS
static_assert(false, "Invalid configuration");
#endif

namespace {

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

int command_train(config& conf){
    if(conf.files.size() < 2){
        std::cout << "Train needs the path to the dataset and the cv set to use" << std::endl;
        return -1;
    }

    auto& dataset_path = conf.files[0];
    auto& cv_set = conf.files[1];

    std::cout << "Dataset: " << dataset_path << std::endl;
    std::cout << "    Set: " << cv_set << std::endl;

    auto dataset = read_dataset(dataset_path);

    std::cout << dataset.line_images.size() << " line images loaded from the dataset" << std::endl;
    std::cout << dataset.word_images.size() << " word images loaded from the dataset" << std::endl;

    if(!dataset.sets.count(cv_set)){
        std::cout << "The subset \"" << cv_set << "\" does not exist" << std::endl;
        return -1;
    }

    auto& set = dataset.sets[cv_set];

    std::cout << set.train_set.size() << " training line images in set" << std::endl;
    std::cout << set.validation_set.size() << " validation line images in set" << std::endl;
    std::cout << set.test_set.size() << " test line images in set" << std::endl;

    std::vector<std::string> train_image_names;
    std::vector<std::string> train_word_names;
    std::vector<std::string> test_image_names;
    std::vector<std::string> valid_image_names;

    for(auto& word_image : dataset.word_images){
        auto& name = word_image.first;
        for(auto& train_image : set.train_set){
            if(name.find(train_image) == 0){
                train_image_names.push_back(name);
                train_word_names.emplace_back(name.begin(), name.end() - 4);
                break;
            }
        }
        for(auto& test_image : set.test_set){
            if(name.find(test_image) == 0){
                test_image_names.push_back(name);
                break;
            }
        }
        for(auto& valid_image : set.validation_set){
            if(name.find(valid_image) == 0){
                valid_image_names.push_back(name);
                break;
            }
        }
    }

    std::cout << train_image_names.size() << " training word images in set" << std::endl;
    std::cout << valid_image_names.size() << " validation word images in set" << std::endl;
    std::cout << test_image_names.size() << " test word images in set" << std::endl;

    if(conf.method_0){
        standard_method(dataset, set, conf, train_word_names, train_image_names, test_image_names);
    } else if(conf.method_1){
        holistic_method(dataset, set, conf, train_word_names, train_image_names, test_image_names);
    } else if(conf.method_2){
        std::cout << "Use method 2 (patches)" << std::endl;

        if(conf.third){
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
                auto patches = mat_to_patches(conf, dataset.word_images[name]);
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
            std::cout << "error: Only -third resolution is supported in method 2 for now" << std::endl;
            print_usage();
            return 1;
        }
    }

    return 0;
}

} //end of anonymous namespace

int main(int argc, char** argv){
    if(argc < 2){
        print_usage();
        return -1;
    }

    auto conf = parse_args(argc, argv);

    if(!conf.method_0 && !conf.method_1 && !conf.method_2){
        std::cout << "error: One method must be selected" << std::endl;
        print_usage();
        return -1;
    }

    if(conf.half){
        conf.downscale = 2;
    } else if(conf.third){
        conf.downscale = 3;
    } else if(conf.quarter){
        conf.downscale = 4;
    }

    if(conf.command == "train"){
        return command_train(conf);
    }

    print_usage();

    return -1;
}
