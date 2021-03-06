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
#include "dll/dbn.hpp"

#ifndef OPENCV_23
#include "dll/ocv_visualizer.hpp"
#endif

#include "nice_svm.hpp"

#include "config.hpp"
#include "standard.hpp"
#include "utils.hpp"
#include "reports.hpp"
#include "dtw.hpp"      //Dynamic time warping
#include "features.hpp" //Features export

#define LOCAL_MEAN_SCALING
#include "scaling.hpp" //Scaling functions

namespace {

using image_type = etl::dyn_matrix<weight>;

} //end of anonymous namespace

template <typename DBN>
std::vector<typename DBN::template layer_type<0>::input_one_t> read_images(const spot_dataset& dataset, config& conf, names train_image_names) {
    std::vector<typename DBN::template layer_type<0>::input_one_t> training_images;

    for (auto& name : train_image_names) {
        if (conf.sub && training_images.size() == 1000) {
            break;
        }

        training_images.emplace_back(holistic_mat<DBN>(conf, dataset.word_images.at(name)));
    }

    return training_images;
}


void holistic_train(
    const spot_dataset& dataset, const spot_dataset_set& set, config& conf,
    names train_word_names, names train_image_names, names valid_image_names, names test_image_names, bool features) {

#ifndef SPOTTER_NO_HOL

    auto evaluate   = [&dataset, &set, &conf](auto& dbn, auto& train_word_names, auto& test_image_names, bool features) {
        using dbn_t = std::decay_t<decltype(*dbn)>;

        std::cout << "Compute features" << std::endl;

        std::vector<decltype(dbn->template prepare_one_output<typename dbn_t::input_t>())> test_features_a;

        for (std::size_t i = 0; i < test_image_names.size(); ++i) {
            test_features_a.emplace_back(dbn->template prepare_one_output<typename dbn_t::input_t>());
        }

        cpp::default_thread_pool<> pool;

        cpp::parallel_foreach_i(pool, test_image_names.begin(), test_image_names.end(),
                                [&test_features_a, &dbn, &dataset, &conf](auto& test_image, std::size_t i) {
                                    auto test_v = holistic_mat<dbn_t>(conf, dataset.word_images.at(test_image));

                                    test_features_a[i] = dbn->features(test_v);
                                });

        std::cout << "... done" << std::endl;

        if (features) {
            export_features_flat(conf, test_image_names, test_features_a, ".1");
        } else {
            std::cout << "Evaluate performance..." << std::endl;

            std::size_t evaluated = 0;

            std::array<double, MAX_N + 1> tp;
            std::array<double, MAX_N + 1> fn;
            std::array<double, MAX_N + 1> maps;

            std::fill(tp.begin(), tp.end(), 0.0);
            std::fill(fn.begin(), fn.end(), 0.0);
            std::fill(maps.begin(), maps.end(), 0.0);

            for (auto& keyword : set.keywords) {
                std::string training_image;
                for (auto& labels : dataset.word_labels) {
                    if (keyword == labels.second && std::find(train_word_names.begin(), train_word_names.end(), labels.first) != train_word_names.end()) {
                        training_image = labels.first;
                        break;
                    }
                }

                //Make sure that there is a sample in the training set
                if (training_image.empty()) {
                    std::cout << "Skipped " << keyword << " since there are no example in the training set" << std::endl;
                    continue;
                }

                auto total_positive = std::count_if(test_image_names.begin(), test_image_names.end(),
                                                    [&dataset, &keyword](auto& i) { return dataset.word_labels.at({i.begin(), i.end() - 4}) == keyword; });

                cpp_assert(total_positive > 0, "No example for one keyword");

                ++evaluated;

                auto ref_v = holistic_mat<dbn_t>(conf, dataset.word_images.at(training_image + ".png"));
                auto ref_a = dbn->template prepare_one_output<typename dbn_t::input_t>();

                ref_a = dbn->features(ref_v);

                std::vector<std::pair<std::string, weight>> diffs_a;

                for (std::size_t t = 0; t < test_image_names.size(); ++t) {
                    decltype(auto) test_image = test_image_names[t];

                    auto diff_a = std::sqrt(etl::sum((ref_a - test_features_a[t]) >> (ref_a - test_features_a[t])));
                    diffs_a.emplace_back(std::string(test_image.begin(), test_image.end() - 4), diff_a);
                }

                std::sort(diffs_a.begin(), diffs_a.end(), [](auto& a, auto& b) { return a.second < b.second; });

                for (std::size_t n = 1; n <= MAX_N; ++n) {
                    int tp_n = 0;

                    for (std::size_t i = 0; i < n && i < diffs_a.size(); ++i) {
                        if (dataset.word_labels.at(diffs_a[i].first) == keyword) {
                            ++tp_n;
                        }
                    }

                    tp[n] += tp_n;
                    fn[n] += total_positive - tp_n;

                    double avep = 0.0;

                    if (tp_n > 0) {
                        for (std::size_t k = 1; k <= n; ++k) {
                            if (dataset.word_labels.at(diffs_a[k - 1].first) == keyword) {
                                int tp_nn = 0;

                                for (std::size_t i = 0; i < k && i < diffs_a.size(); ++i) {
                                    if (dataset.word_labels.at(diffs_a[i].first) == keyword) {
                                        ++tp_nn;
                                    }
                                }

                                avep += static_cast<double>(tp_nn) / k;
                            }
                        }

                        avep /= tp_n;
                    }

                    maps[n] += avep;
                }
            }

            std::cout << "... done" << std::endl;

            std::cout << evaluated << " keywords evaluated" << std::endl;

            for (std::size_t n = 1; n <= MAX_N; ++n) {
                std::cout << "TP(" << n << ") = " << tp[n] << std::endl;
                std::cout << "FP(" << n << ") = " << (n * set.keywords.size() - tp[n]) << std::endl;
                std::cout << "FN(" << n << ") = " << fn[n] << std::endl;
                std::cout << "Precision(" << n << ") = " << (tp[n] / (n * set.keywords.size())) << std::endl;
                std::cout << "Recall(" << n << ") = " << (tp[n] / (tp[n] + fn[n])) << std::endl;
                std::cout << "MAP(" << n << ") = " << (maps[n] / set.keywords.size()) << std::endl;
            }
        }
    };

    if (conf.half) {
        static constexpr const std::size_t NF  = 13;
        static constexpr const std::size_t NF2 = 8;
        static constexpr const std::size_t NF3 = 5;

        using cdbn_t =
            dll::dbn_desc<
                dll::dbn_layers<
                    dll::conv_rbm_desc<
                        1, WIDTH / 2, HEIGHT / 2 //330x60 input image (1 channel)
                        , 30 //Number of feature maps
                        , NF, NF //Configure the size of the filter
                        //, 2                                       //Probabilistic max pooling (2x2)
                        , dll::weight_type<weight>, dll::batch_size<25>, dll::verbose, dll::momentum
                        //, dll::weight_decay<dll::decay_type::L2>
                        , dll::dbn_only
                        //, dll::sparsity<dll::sparsity_method::LEE>
                        >::layer_t,
                    dll::mp_3d_layer_desc<30, 318, 48, 1, 2, 2, dll::weight_type<weight>>::layer_t,
                    dll::conv_rbm_desc<
                        30, 159, 24, 30, NF2, NF2, dll::weight_type<weight>, dll::batch_size<25>, dll::verbose, dll::momentum
                        //, dll::weight_decay<dll::decay_type::L2>
                        , dll::dbn_only
                        //, dll::sparsity<dll::sparsity_method::LEE>
                        >::layer_t,
                    dll::conv_rbm_desc<
                        30, 152, 17, 30, NF3, NF3, dll::weight_type<weight>, dll::batch_size<25>, dll::verbose, dll::momentum
                        //, dll::weight_decay<dll::decay_type::L2>
                        , dll::dbn_only
                        //, dll::sparsity<dll::sparsity_method::LEE>
                        >::layer_t>,
                dll::batch_mode>::dbn_t;

        auto training_images = read_images<cdbn_t>(dataset, conf, train_image_names);

        auto cdbn = std::make_unique<cdbn_t>();

        cdbn->display();

        std::cout << cdbn->output_size() << " output features" << std::endl;

        auto mini_div = (1024.0 * 1024.0);
        auto div      = sizeof(std::size_t) / (mini_div);
        auto mul      = training_images.size() * div;

        std::cout << "DBN size: " << sizeof(cdbn_t) * div << "MB" << std::endl;

        std::cout << "Layer 0 input storage: " << cdbn->template layer_get<0>().input_size() * mul << "MB necessary" << std::endl;
        std::cout << "Layer 0 tmp storage: " << cdbn->template layer_get<0>().output_size() * mul << "MB necessary" << std::endl;
        std::cout << "Layer 1 tmp storage: " << cdbn->template layer_get<1>().output_size() * mul << "MB necessary" << std::endl;
        std::cout << "Layer 2 tmp storage: " << cdbn->template layer_get<2>().output_size() * mul << "MB necessary" << std::endl;
        std::cout << "Layer 3 tmp storage: " << cdbn->template layer_get<3>().output_size() * mul << "MB necessary" << std::endl;

        std::cout << "Trainer 0 storage: " << sizeof(dll::base_cd_trainer<1, cdbn_t::layer_type<0>, false>) / mini_div << "MB necessary" << std::endl;
        std::cout << "Trainer 2 storage: " << sizeof(dll::base_cd_trainer<1, cdbn_t::layer_type<2>, false>) / mini_div << "MB necessary" << std::endl;

        std::cout << "Final test features:" << test_image_names.size() * cdbn->output_size() * div << "MB" << std::endl;

        if (conf.load || features) {
            cdbn->load("method_1_half.dat");
        } else {
            cdbn->pretrain(training_images, 2);
            cdbn->store("method_1_half.dat");
        }

        std::cout << "Evaluate on training set" << std::endl;
        evaluate(cdbn, train_word_names, train_image_names, features);

        std::cout << "Evaluate on validation set" << std::endl;
        evaluate(cdbn, train_word_names, valid_image_names, features);

        std::cout << "Evaluate on test set" << std::endl;
        evaluate(cdbn, train_word_names, test_image_names, features);
    } else if (conf.third) {
        static constexpr const std::size_t NF  = 13;
        static constexpr const std::size_t NF2 = 7;
        static constexpr const std::size_t NF3 = 3;

        using cdbn_t =
            dll::dbn_desc<
                dll::dbn_layers<
                    dll::conv_rbm_desc<
                        1, WIDTH / 3, HEIGHT / 3 //165x30 input image (1 channel)
                        , 30 //Number of feature maps
                        , NF, NF //Configure the size of the filter
                        //, 2                                       //Probabilistic max pooling (2x2)
                        , dll::weight_type<weight>, dll::batch_size<25>, dll::verbose, dll::momentum, dll::weight_decay<dll::decay_type::L2>, dll::dbn_only, dll::hidden<dll::unit_type::RELU6>
                        //, dll::sparsity<dll::sparsity_method::LEE>
                        >::layer_t,
                    dll::mp_3d_layer_desc<30, 208, 28, 1, 2, 2, dll::weight_type<weight>>::layer_t,
                    dll::conv_rbm_desc<30, 104, 14, 30, NF2, NF2, dll::weight_type<weight>, dll::batch_size<25>, dll::verbose, dll::momentum, dll::weight_decay<dll::decay_type::L2>, dll::dbn_only, dll::sparsity<dll::sparsity_method::LEE>>::layer_t,
                    dll::mp_3d_layer_desc<30, 98, 8, 1, 2, 2, dll::weight_type<weight>>::layer_t,
                    dll::conv_rbm_desc<30, 49, 4, 30, NF3, NF3, dll::weight_type<weight>, dll::batch_size<25>, dll::verbose, dll::momentum, dll::weight_decay<dll::decay_type::L2>, dll::dbn_only, dll::sparsity<dll::sparsity_method::LEE>>::layer_t>,
                dll::batch_mode>::dbn_t;

        auto training_images = read_images<cdbn_t>(dataset, conf, train_image_names);

        auto cdbn = std::make_unique<cdbn_t>();

        cdbn->display();

        std::cout << cdbn->output_size() << " output features" << std::endl;

        cdbn->template layer_get<0>().learning_rate /= 100;
        cdbn->template layer_get<2>().learning_rate /= 10;
        cdbn->template layer_get<4>().learning_rate /= 10;
        cdbn->template layer_get<4>().pbias_lambda *= 2;

        std::string file_name = "method_1_third.dat";

        if (conf.view) {
            cdbn->load(file_name);

#ifndef OPENCV_23
            dll::visualize_rbm(cdbn->template layer_get<0>());
#endif
        } else {
            if (conf.load || features) {
                cdbn->load(file_name);
            } else {
                cdbn->pretrain(training_images, 10);
                cdbn->store(file_name);
            }

            std::cout << "Evaluate on training set" << std::endl;
            evaluate(cdbn, train_word_names, train_image_names, features);

            std::cout << "Evaluate on validation set" << std::endl;
            evaluate(cdbn, train_word_names, valid_image_names, features);

            std::cout << "Evaluate on test set" << std::endl;
            evaluate(cdbn, train_word_names, test_image_names, features);

            if (conf.svm) {
                std::vector<std::vector<double>> training_samples(train_image_names.size());
                std::vector<decltype(cdbn->template prepare_one_output<typename cdbn_t::input_t>())> training_features;
                std::vector<std::size_t> training_labels(train_image_names.size());

                for (auto& train_image_name : train_image_names) {
                    cpp_unused(train_image_name);
                    training_features.emplace_back(cdbn->template prepare_one_output<typename cdbn_t::input_t>());
                }

                cpp::default_thread_pool<> pool;

                std::map<std::vector<std::string>, std::size_t> ids;
                std::size_t next = 0;

                for (std::size_t i = 0; i < train_image_names.size(); ++i) {
                    auto test_image = train_image_names[i];

                    auto test_v = holistic_mat<cdbn_t>(conf, dataset.word_images.at(test_image));

                    training_features[i] = cdbn->features(test_v);

                    std::copy(training_features[i].begin(), training_features[i].end(), std::back_inserter(training_samples[i]));

                    test_image = std::string(test_image.begin(), test_image.end() - 4);

                    std::cout << "test_image=\"" << test_image << "\"" << std::endl;
                    std::cout << "word_label.first=" << dataset.word_labels.begin()->first << std::endl;

                    if (dataset.word_labels.count(test_image)) {
                        std::cout << "WTF" << std::endl;
                    }

                    auto label = dataset.word_labels.at(test_image);
                    std::cout << label << std::endl;
                    if (!ids.count(label)) {
                        ids[label] = next++;
                    }

                    training_labels[i] = ids[label];

                    std::cout << training_labels[i] << std::endl;
                }

                std::cout << "... done" << std::endl;

                auto training_problem = svm::make_problem(training_labels, training_samples);

                auto mnist_parameters = svm::default_parameters();

                mnist_parameters.svm_type    = C_SVC;
                mnist_parameters.kernel_type = RBF;
                mnist_parameters.probability = 1;
                mnist_parameters.C           = 2.8;
                mnist_parameters.gamma       = 0.0073;

                //Make it quiet
                svm::make_quiet();

                //Make sure parameters are not too messed up
                if (!svm::check(training_problem, mnist_parameters)) {
                    return;
                }

                svm::model model;
                model = svm::train(training_problem, mnist_parameters);

                std::cout << "Number of classes: " << model.classes() << std::endl;

                std::cout << "Test on training set" << std::endl;
                svm::test_model(training_problem, model);
            }
        }
    } else if (conf.quarter) {
        static constexpr const std::size_t NF  = 7;
        static constexpr const std::size_t NF2 = 3;
        static constexpr const std::size_t NF3 = 3;

        using cdbn_t =
            dll::dbn_desc<
                dll::dbn_layers<
                    dll::conv_rbm_desc<
                        1, WIDTH / 4, HEIGHT / 4 //165x30 input image (1 channel)
                        ,
                        30 //Number of feature maps
                        ,
                        NF, NF //Configure the size of the filter
                        //, 2                                       //Probabilistic max pooling (2x2)
                        ,
                        dll::weight_type<weight>, dll::batch_size<25>, dll::verbose, dll::momentum
                        //, dll::weight_decay<dll::decay_type::L2>
                        ,
                        dll::dbn_only, dll::sparsity<dll::sparsity_method::LEE>>::layer_t
                    //, dll::mp_3d_layer_desc<30,318,48,1,2,2, dll::weight_type<weight>>::layer_t
                    , dll::conv_rbm_desc<
                        30, 159, 24, 30, NF2, NF2, dll::weight_type<weight>, dll::batch_size<25>, dll::verbose, dll::momentum
                        //, dll::weight_decay<dll::decay_type::L2>
                        , dll::dbn_only, dll::sparsity<dll::sparsity_method::LEE>>::layer_t
                    , dll::conv_rbm_desc<
                        30, 157, 22, 30, NF3, NF3, dll::weight_type<weight>, dll::batch_size<25>, dll::verbose, dll::momentum
                        //, dll::weight_decay<dll::decay_type::L2>
                        , dll::dbn_only, dll::sparsity<dll::sparsity_method::LEE>>::layer_t
                >
                , dll::batch_mode>::dbn_t;

        auto training_images = read_images<cdbn_t>(dataset, conf, train_image_names);

        auto cdbn = std::make_unique<cdbn_t>();

        cdbn->display();

        std::cout << cdbn->output_size() << " output features" << std::endl;

        cdbn->template layer_get<0>().learning_rate /= 10;
        cdbn->template layer_get<1>().learning_rate /= 10;
        cdbn->template layer_get<2>().learning_rate /= 10;
        cdbn->template layer_get<2>().pbias_lambda *= 2;

        if (conf.load || features) {
            cdbn->load("method_1_quarter.dat");
        } else {
            cdbn->pretrain(training_images, 10);
            cdbn->store("method_1_quarter.dat");
        }

        std::cout << "Evaluate on training set" << std::endl;
        evaluate(cdbn, train_word_names, train_image_names, features);

        std::cout << "Evaluate on validation set" << std::endl;
        evaluate(cdbn, train_word_names, valid_image_names, features);

        std::cout << "Evaluate on test set" << std::endl;
        evaluate(cdbn, train_word_names, test_image_names, features);
    } else {
        using cdbn_t =
            dll::dbn_desc<
                dll::dbn_layers<
                    dll::conv_rbm_desc<
                        1, WIDTH, HEIGHT //660x120 input image (1 channel)
                        ,
                        12 //Number of feature maps
                        ,
                        WIDTH + 1 - 19, HEIGHT + 1 - 19 //Configure the size of the filter
                        //, 2                                //Probabilistic max pooling (2x2)
                        ,
                        dll::batch_size<25>, dll::verbose, dll::momentum, dll::weight_type<weight>, dll::weight_decay<dll::decay_type::L2>>::layer_t>,
                dll::batch_mode>::dbn_t;

        auto training_images = read_images<cdbn_t>(dataset, conf, train_image_names);

        auto cdbn = std::make_unique<cdbn_t>();

        std::cout << cdbn->output_size() << " output features" << std::endl;

        if (conf.load || features) {
            cdbn->load("method_1.dat");
        } else {
            cdbn->pretrain(training_images, 10);
            cdbn->store("method_1.dat");
        }

        std::cout << "Evaluate on training set" << std::endl;
        evaluate(cdbn, train_word_names, train_image_names, features);

        std::cout << "Evaluate on validation set" << std::endl;
        evaluate(cdbn, train_word_names, valid_image_names, features);

        std::cout << "Evaluate on test set" << std::endl;
        evaluate(cdbn, train_word_names, test_image_names, features);
    }
#else
    cpp_unused(dataset);
    cpp_unused(set);
    cpp_unused(conf);
    cpp_unused(train_word_names);
    cpp_unused(train_image_names);
    cpp_unused(valid_image_names);
    cpp_unused(test_image_names);
    cpp_unused(features);
#endif //SPOTTER_NO_HOL
}

void holistic_features(
    const spot_dataset& dataset, const spot_dataset_set& set, config& conf,
    names train_word_names, names train_image_names, names valid_image_names, names test_image_names) {
    //Export features
    holistic_train(dataset, set, conf, train_word_names, train_image_names, valid_image_names, test_image_names, true);

    return;
}
