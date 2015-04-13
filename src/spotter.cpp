//=======================================================================
// Copyright Baptiste Wicht 2015.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================


#include <iostream>

#include "dll/conv_rbm.hpp"
#include "dll/conv_rbm_mp.hpp"
#include "dll/dbn.hpp"
#include "dll/avgp_layer.hpp"
#include "dll/mp_layer.hpp"
#include "dll/ocv_visualizer.hpp"

#include "etl/print.hpp"

#include "nice_svm.hpp"

#include "cpp_utils/parallel.hpp"

#include "config.hpp"
#include "washington.hpp"

using weight = double;

static constexpr const std::size_t WIDTH = 660;
static constexpr const std::size_t HEIGHT = 120;

constexpr const std::size_t MAX_N = 25;

static_assert(WIDTH % 2 == 0, "Width must be divisible by 2");
static_assert(HEIGHT % 2 == 0, "Height must be divisible by 2");

static_assert(WIDTH % 3 == 0, "Width must be divisible by 4");
static_assert(HEIGHT % 3 == 0, "Height must be divisible by 4");

static_assert(WIDTH % 4 == 0, "Width must be divisible by 4");
static_assert(HEIGHT % 4 == 0, "Height must be divisible by 4");

template<typename T>
std::ostream& operator<<(std::ostream& stream, const std::vector<T>& vec){
    std::string comma = "";
    stream << "[";
    for(auto& v : vec){
        stream << comma << v;
        comma = ", ";
    }
    stream << "]";

    return stream;
}

namespace {

etl::dyn_matrix<weight> mat_to_dyn(const config& conf, cv::Mat& image){
    cv::Mat normalized(cv::Size(WIDTH, HEIGHT), CV_8U);
    normalized = cv::Scalar(255);

    image.copyTo(normalized(cv::Rect((WIDTH - image.size().width) / 2, 0, image.size().width, HEIGHT)));

    if(conf.half){
        cv::Mat scaled_normalized(cv::Size(WIDTH / 2, HEIGHT / 2), CV_8U);
        cv::resize(normalized, scaled_normalized, scaled_normalized.size(), cv::INTER_AREA);
        cv::adaptiveThreshold(scaled_normalized, normalized, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 7, 2);
    } else if(conf.quarter) {
        cv::Mat scaled_normalized(cv::Size(WIDTH / 4, HEIGHT / 4), CV_8U);
        cv::resize(normalized, scaled_normalized, scaled_normalized.size(), cv::INTER_AREA);
        cv::adaptiveThreshold(scaled_normalized, normalized, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 7, 2);
    } else if(conf.third) {
        cv::Mat scaled_normalized(cv::Size(WIDTH / 3, HEIGHT / 3), CV_8U);
        cv::resize(normalized, scaled_normalized, scaled_normalized.size(), cv::INTER_AREA);
        cv::adaptiveThreshold(scaled_normalized, normalized, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 7, 2);
    }

    etl::dyn_matrix<weight> training_image(normalized.size().width, normalized.size().height);

    for(std::size_t y = 0; y < static_cast<std::size_t>(normalized.size().height); ++y){
        for(std::size_t x = 0; x < static_cast<std::size_t>(normalized.size().width); ++x){
            auto pixel = normalized.at<uint8_t>(cv::Point(x, y));

            training_image(x, y) = pixel == 0 ? 0.0 : 1.0;

            if(pixel != 0 && pixel != 255){
                std::cout << "The normalized input image is not binary! pixel:" << static_cast<int>(pixel) << std::endl;
            }
        }
    }

    return training_image;
}

int command_train(const config& conf){
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

    std::vector<etl::dyn_matrix<weight>> training_images;

    for(auto& name : train_image_names){
        if(conf.sub && training_images.size() == 1000){
            break;
        }

        training_images.emplace_back(mat_to_dyn(conf, dataset.word_images[name]));
    }

    if(conf.method_1){
        std::cout << "Use method 1 (holistic)" << std::endl;

        auto evaluate = [&dataset,&set,&conf](auto& crbm, auto& train_word_names, auto& test_image_names){
            std::vector<etl::dyn_matrix<weight, 3>> test_features_a;

            for(std::size_t i = 0; i < test_image_names.size(); ++i){
                test_features_a.emplace_back(crbm->prepare_one_output());
            }

            cpp::default_thread_pool<> pool;

            cpp::parallel_foreach_i(pool, test_image_names.begin(), test_image_names.end(),
                [&test_features_a, &crbm, &dataset, &conf](auto& test_image, std::size_t i){
                    auto test_v = mat_to_dyn(conf, dataset.word_images[test_image]);

                    crbm->activation_probabilities(test_v, test_features_a[i]);
                });

            std::cout << "... done" << std::endl;

            std::cout << "Evaluate performance..." << std::endl;

            std::size_t evaluated = 0;

            std::array<double, MAX_N + 1> tp;
            std::array<double, MAX_N + 1> fn;
            std::array<double, MAX_N + 1> maps;

            std::fill(tp.begin(), tp.end(), 0.0);
            std::fill(fn.begin(), fn.end(), 0.0);
            std::fill(maps.begin(), maps.end(), 0.0);

            for(auto& keyword : set.keywords){
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

                auto total_positive = std::count_if(test_image_names.begin(), test_image_names.end(),
                    [&dataset, &keyword](auto& i){ return dataset.word_labels[{i.begin(), i.end() - 4}] == keyword; });

                //Make sure that there is a sample in the test set
                if(total_positive == 0){
                    std::cout << "Skipped " << keyword << " since there are no example in the test set" << std::endl;
                    continue;
                }

                ++evaluated;

                auto ref_v = mat_to_dyn(conf, dataset.word_images[training_image + ".png"]);
                auto ref_a = crbm->prepare_one_output();

                crbm->activation_probabilities(ref_v, ref_a);

                std::vector<std::pair<std::string, weight>> diffs_a;

                for(std::size_t t = 0; t < test_image_names.size(); ++t){
                    decltype(auto) test_image = test_image_names[t];

                    auto diff_a = std::sqrt(etl::sum((ref_a - test_features_a[t]) * (ref_a - test_features_a[t])));
                    diffs_a.emplace_back(std::string(test_image.begin(), test_image.end() - 4), diff_a);
                }

                std::sort(diffs_a.begin(), diffs_a.end(), [](auto& a, auto& b){ return a.second < b.second; });

                for(std::size_t n = 1; n <= MAX_N; ++n){
                    int tp_n = 0;

                    for(std::size_t i = 0; i < n && i < diffs_a.size(); ++i){
                        if(dataset.word_labels[diffs_a[i].first] == keyword){
                            ++tp_n;
                        }
                    }

                    tp[n] += tp_n;
                    fn[n] += total_positive - tp_n;

                    double avep = 0.0;

                    if(tp_n > 0){
                        for(std::size_t k = 1; k <= n; ++k){
                            if(dataset.word_labels[diffs_a[k-1].first] == keyword){
                                int tp_nn = 0;

                                for(std::size_t i = 0; i < k && i < diffs_a.size(); ++i){
                                    if(dataset.word_labels[diffs_a[i].first] == keyword){
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

            for(std::size_t n = 1; n <= MAX_N; ++n){
                std::cout << "TP(" << n << ") = " << tp[n] << std::endl;
                std::cout << "FP(" << n << ") = " << (n * set.keywords.size() - tp[n]) << std::endl;
                std::cout << "FN(" << n << ") = " << fn[n] << std::endl;
                std::cout << "Precision(" << n << ") = " << (tp[n] / (n * set.keywords.size())) << std::endl;
                std::cout << "Recall(" << n << ") = " << (tp[n] / (tp[n] + fn[n])) << std::endl;
                std::cout << "MAP(" << n << ") = " << (maps[n] / set.keywords.size()) << std::endl;
            }
        };

        if(conf.half){
            static constexpr const std::size_t NF = 13;
            static constexpr const std::size_t NF2 = 8;
            static constexpr const std::size_t NF3 = 5;

            using cdbn_t =
                dll::dbn_desc<
                    dll::dbn_layers<
                        dll::conv_rbm_desc<
                            WIDTH / 2, HEIGHT / 2, 1                    //330x60 input image (1 channel)
                            , WIDTH / 2 + 1 - NF , HEIGHT / 2 + 1 - NF  //Configure the size of the filter
                            , 30                                       //Number of feature maps
                            //, 2                                       //Probabilistic max pooling (2x2)
                            , dll::weight_type<weight>
                            , dll::batch_size<25>
                            , dll::parallel
                            , dll::verbose
                            , dll::momentum
                            //, dll::weight_decay<dll::decay_type::L2>
                            , dll::dbn_only
                            //, dll::sparsity<dll::sparsity_method::LEE>
                        >::rbm_t
                        , dll::mp_layer_3d_desc<30,318,48,1,2,2>::layer_t
                        , dll::conv_rbm_desc<
                            159, 24, 30
                            , 159 + 1 - NF2 , 24 + 1 - NF2
                            , 30
                            , dll::weight_type<weight>
                            , dll::batch_size<25>
                            , dll::parallel
                            , dll::verbose
                            , dll::momentum
                            //, dll::weight_decay<dll::decay_type::L2>
                            , dll::dbn_only
                            //, dll::sparsity<dll::sparsity_method::LEE>
                        >::rbm_t
                        , dll::conv_rbm_desc<
                            152, 17, 30
                            , 152 + 1 - NF3 , 17 + 1 - NF3
                            , 30
                            , dll::weight_type<weight>
                            , dll::batch_size<25>
                            , dll::parallel
                            , dll::verbose
                            , dll::momentum
                            //, dll::weight_decay<dll::decay_type::L2>
                            , dll::dbn_only
                            //, dll::sparsity<dll::sparsity_method::LEE>
                        >::rbm_t
                    >
                    , dll::memory
                >::dbn_t;

            auto cdbn = std::make_unique<cdbn_t>();

            cdbn->display();

            std::cout << cdbn->output_size() << " output features" << std::endl;

            auto mini_div = (1024.0 * 1024.0);
            auto div = sizeof(std::size_t) / (mini_div);
            auto mul = training_images.size() * div;

            std::cout << "DBN size: " << sizeof(cdbn_t) * div << "MB" << std::endl;

            std::cout << "Layer 0 input storage: " << cdbn->template layer<0>().input_size() * mul << "MB necessary" << std::endl;
            std::cout << "Layer 0 tmp storage: " << cdbn->template layer<0>().output_size() * mul << "MB necessary" << std::endl;
            std::cout << "Layer 1 tmp storage: " << cdbn->template layer<1>().output_size() * mul << "MB necessary" << std::endl;
            std::cout << "Layer 2 tmp storage: " << cdbn->template layer<2>().output_size() * mul << "MB necessary" << std::endl;
            std::cout << "Layer 3 tmp storage: " << cdbn->template layer<3>().output_size() * mul << "MB necessary" << std::endl;

            std::cout << "Trainer 0 storage: " << sizeof(dll::base_cd_trainer<1, cdbn_t::rbm_type<0>, false, false>) / mini_div << "MB necessary" << std::endl;
            std::cout << "Trainer 2 storage: " << sizeof(dll::base_cd_trainer<1, cdbn_t::rbm_type<2>, false, false>) / mini_div << "MB necessary" << std::endl;

            std::cout << "Final test features:" << test_image_names.size() * cdbn->output_size() * div << "MB" << std::endl;

            //cdbn->template layer<0>().learning_rate /= 10;
            //cdbn->template layer<1>()->learning_rate *= 10;

            cdbn->pretrain(training_images, 2);
            cdbn->store("method_1_half.dat");
            //cdbn->load("method_1_half.dat");

            std::cout << "Evaluate on training set" << std::endl;
            evaluate(cdbn, train_word_names, train_image_names);

            std::cout << "Evaluate on test set" << std::endl;
            evaluate(cdbn, train_word_names, test_image_names);
        } else if(conf.third){
            static constexpr const std::size_t NF = 13;
            static constexpr const std::size_t NF2 = 7;
            static constexpr const std::size_t NF3 = 3;

            using cdbn_t =
                dll::dbn_desc<
                    dll::dbn_layers<
                        dll::conv_rbm_desc<
                            WIDTH / 3, HEIGHT / 3, 1                    //165x30 input image (1 channel)
                            , WIDTH / 3 + 1 - NF , HEIGHT / 3 + 1 - NF  //Configure the size of the filter
                            , 30                                        //Number of feature maps
                            //, 2                                       //Probabilistic max pooling (2x2)
                            , dll::weight_type<weight>
                            , dll::batch_size<25>
                            , dll::parallel
                            , dll::verbose
                            , dll::momentum
                            , dll::weight_decay<dll::decay_type::L2>
                            , dll::dbn_only
                            , dll::hidden<dll::unit_type::RELU6>
                            //, dll::sparsity<dll::sparsity_method::LEE>
                            , dll::watcher<dll::opencv_rbm_visualizer>
                        >::rbm_t
                        , dll::mp_layer_3d_desc<30,208,28,1,2,2>::layer_t
                        , dll::conv_rbm_desc<
                            104, 14, 30
                            , 104 + 1 - NF2 , 14 + 1 - NF2
                            , 30
                            , dll::weight_type<weight>
                            , dll::batch_size<25>
                            , dll::parallel
                            , dll::verbose
                            , dll::momentum
                            , dll::weight_decay<dll::decay_type::L2>
                            , dll::dbn_only
                            , dll::sparsity<dll::sparsity_method::LEE>
                        >::rbm_t
                        , dll::mp_layer_3d_desc<30,98,8,1,2,2>::layer_t
                        , dll::conv_rbm_desc<
                            49, 4, 30
                            , 49 + 1 - NF3 , 4 + 1 - NF3
                            , 30
                            , dll::weight_type<weight>
                            , dll::batch_size<25>
                            , dll::parallel
                            , dll::verbose
                            , dll::momentum
                            , dll::weight_decay<dll::decay_type::L2>
                            , dll::dbn_only
                            , dll::sparsity<dll::sparsity_method::LEE>
                        >::rbm_t
                    >
                    , dll::memory
                >::dbn_t;

            auto cdbn = std::make_unique<cdbn_t>();

            cdbn->display();

            std::cout << cdbn->output_size() << " output features" << std::endl;

            cdbn->template layer<0>().learning_rate /= 100;
            //cdbn->template layer<0>().pbias_lambda *= 4;

            cdbn->template layer<2>().learning_rate /= 10;
            cdbn->template layer<4>().learning_rate /= 10;
            cdbn->template layer<4>().pbias_lambda *= 2;

            std::string file_name = "method_1_third.dat";

            if(conf.view){
                cdbn->load(file_name);

                dll::visualize_rbm(cdbn->template layer<0>());
            } else {
                //cdbn->pretrain(training_images, 10);
                //cdbn->store(file_name);
                cdbn->load(file_name);

                std::cout << "Evaluate on training set" << std::endl;
                evaluate(cdbn, train_word_names, train_image_names);

                std::cout << "Evaluate on test set" << std::endl;
                //evaluate(cdbn, train_word_names, test_image_names);

                //for(std::size_t i = 0; i < 4; ++i){
                    //auto features = cdbn->prepare_one_output();

                    //cdbn->activation_probabilities(
                        //training_images[i],
                        //features);

                    //std::cout << features << std::endl;
                    //std::cout << etl::sum(features) << std::endl;
                    //std::cout << etl::to_string(features) << std::endl;
                //}

                if(conf.svm){
                    std::vector<std::vector<double>> training_samples(train_image_names.size());
                    std::vector<etl::dyn_matrix<weight, 3>> training_features;
                    std::vector<std::size_t> training_labels(train_image_names.size());

                    for(std::size_t i = 0; i < train_image_names.size(); ++i){
                        training_features.emplace_back(cdbn->prepare_one_output());
                    }

                    cpp::default_thread_pool<> pool;

                    std::map<std::vector<std::string>, std::size_t> ids;
                    std::size_t next =  0;

                    for(std::size_t i = 0; i < train_image_names.size(); ++i){
                        auto test_image = train_image_names[i];

                        auto test_v = mat_to_dyn(conf, dataset.word_images[test_image]);

                        cdbn->activation_probabilities(test_v, training_features[i]);

                        std::copy(training_features[i].begin(), training_features[i].end(), std::back_inserter(training_samples[i]));

                        test_image = std::string(test_image.begin(), test_image.end() - 4);

                        std::cout << "test_image=\"" << test_image << "\"" << std::endl;
                        std::cout << "word_label.first=" << dataset.word_labels.begin()->first << std::endl;

                        if(dataset.word_labels.count(test_image)){
                            std::cout << "WTF" << std::endl;
                        }

                        auto label = dataset.word_labels[test_image];
                        std::cout << label << std::endl;
                        if(!ids.count(label)){
                            ids[label] = next++;
                        }

                        training_labels[i] = ids[label];

                        std::cout << training_labels[i] << std::endl;
                    }

                    std::cout << "... done" << std::endl;

                    auto training_problem = svm::make_problem(training_labels, training_samples);
                    //auto test_problem = svm::make_problem(dataset.test_labels, dataset.test_images, 0, false);

                    auto mnist_parameters = svm::default_parameters();

                    mnist_parameters.svm_type = C_SVC;
                    mnist_parameters.kernel_type = RBF;
                    mnist_parameters.probability = 1;
                    mnist_parameters.C = 2.8;
                    mnist_parameters.gamma = 0.0073;

                    //Make it quiet
                    svm::make_quiet();

                    //Make sure parameters are not too messed up
                    if(!svm::check(training_problem, mnist_parameters)){
                        return 1;
                    }

                    svm::model model;
                    model = svm::train(training_problem, mnist_parameters);

                    std::cout << "Number of classes: " << model.classes() << std::endl;

                    std::cout << "Test on training set" << std::endl;
                    svm::test_model(training_problem, model);
                }
            }
        } else if(conf.quarter){
            static constexpr const std::size_t NF = 7;
            static constexpr const std::size_t NF2 = 3;
            static constexpr const std::size_t NF3 = 3;

            using cdbn_t =
                dll::dbn_desc<
                    dll::dbn_layers<
                        dll::conv_rbm_desc<
                            WIDTH / 4, HEIGHT / 4, 1                    //165x30 input image (1 channel)
                            , WIDTH / 4 + 1 - NF , HEIGHT / 4 + 1 - NF  //Configure the size of the filter
                            , 30                                        //Number of feature maps
                            //, 2                                       //Probabilistic max pooling (2x2)
                            , dll::weight_type<weight>
                            , dll::batch_size<25>
                            , dll::parallel
                            , dll::verbose
                            , dll::momentum
                            //, dll::weight_decay<dll::decay_type::L2>
                            , dll::dbn_only
                            , dll::sparsity<dll::sparsity_method::LEE>
                        >::rbm_t
                        //, dll::mp_layer_3d_desc<30,318,48,1,2,2>::layer_t
                        , dll::conv_rbm_desc<
                            159, 24, 30
                            , 159 + 1 - NF2 , 24 + 1 - NF2
                            , 30
                            , dll::weight_type<weight>
                            , dll::batch_size<25>
                            , dll::parallel
                            , dll::verbose
                            , dll::momentum
                            //, dll::weight_decay<dll::decay_type::L2>
                            , dll::dbn_only
                            , dll::sparsity<dll::sparsity_method::LEE>
                        >::rbm_t
                        , dll::conv_rbm_desc<
                            157, 22, 30
                            , 157 + 1 - NF3 , 22 + 1 - NF3
                            , 30
                            , dll::weight_type<weight>
                            , dll::batch_size<25>
                            , dll::parallel
                            , dll::verbose
                            , dll::momentum
                            //, dll::weight_decay<dll::decay_type::L2>
                            , dll::dbn_only
                            , dll::sparsity<dll::sparsity_method::LEE>
                        >::rbm_t
                    >
                    , dll::memory
                >::dbn_t;

            auto cdbn = std::make_unique<cdbn_t>();

            cdbn->display();

            std::cout << cdbn->output_size() << " output features" << std::endl;

            cdbn->template layer<0>().learning_rate /= 10;
            cdbn->template layer<1>().learning_rate /= 10;
            cdbn->template layer<2>().learning_rate /= 10;
            cdbn->template layer<2>().pbias_lambda *= 2;

            cdbn->pretrain(training_images, 10);
            cdbn->store("method_1_quarter.dat");
            //cdbn->load("method_1_quarter.dat");
            std::cout << "Evaluate on training set" << std::endl;
            evaluate(cdbn, train_word_names, train_image_names);

            std::cout << "Evaluate on test set" << std::endl;
            evaluate(cdbn, train_word_names, test_image_names);
        } else {
            using cdbn_t =
                dll::dbn_desc<
                    dll::dbn_layers<
                        dll::conv_rbm_desc<
                        WIDTH, HEIGHT, 1                     //660x120 input image (1 channel)
                        , WIDTH + 1 - 19 , HEIGHT + 1 - 19   //Configure the size of the filter
                        , 12                                 //Number of feature maps
                        //, 2                                //Probabilistic max pooling (2x2)
                        , dll::batch_size<25>
                        , dll::parallel
                        , dll::verbose
                        , dll::momentum
                        , dll::weight_type<weight>
                        , dll::weight_decay<dll::decay_type::L2>
                        >::rbm_t
                    >
                    , dll::memory
                >::dbn_t;

            auto cdbn = std::make_unique<cdbn_t>();

            std::cout << cdbn->output_size() << " output features" << std::endl;

            cdbn->pretrain(training_images, 10);
            cdbn->store("method_1.dat");
            //cdbn->load("method_1.dat");

            std::cout << "Evaluate on training set" << std::endl;
            evaluate(cdbn, train_word_names, train_image_names);

            std::cout << "Evaluate on test set" << std::endl;
            evaluate(cdbn, train_word_names, test_image_names);
        }
    } else if(conf.method_2){
        std::cout << "Use method 2 (patches)" << std::endl;

        if(conf.third){
            static constexpr const std::size_t NF = 17;
            static constexpr const std::size_t NF2 = 11;
            //static constexpr const std::size_t NF3 = 3;

            static constexpr const std::size_t NV = HEIGHT / 3;

            using cdbn_t =
                dll::dbn_desc<
                    dll::dbn_layers<
                        dll::conv_rbm_desc<
                            NV, NV, 1                                   //40x40 input image (1 channel)
                            , NV + 1 - NF , NV + 1 - NF                 //Configure the size of the filter
                            , 30                                        //Number of feature maps
                            //, 2                                       //Probabilistic max pooling (2x2)
                            , dll::weight_type<weight>
                            , dll::batch_size<25>
                            , dll::parallel
                            , dll::verbose
                            , dll::momentum
                            , dll::weight_decay<dll::decay_type::L2>
                            , dll::dbn_only
                            //, dll::hidden<dll::unit_type::RELU6>
                            //, dll::sparsity<dll::sparsity_method::LEE>
                            //, dll::watcher<dll::opencv_rbm_visualizer>
                        >::rbm_t
                        , dll::conv_rbm_desc<
                            24, 24, 30
                            , 24 + 1 - NF2 , 24 + 1 - NF2
                            , 30
                            , dll::weight_type<weight>
                            , dll::batch_size<25>
                            , dll::parallel
                            , dll::verbose
                            , dll::momentum
                            , dll::weight_decay<dll::decay_type::L2>
                            , dll::dbn_only
                            //, dll::sparsity<dll::sparsity_method::LEE>
                        >::rbm_t
                    >
                    , dll::memory
                >::dbn_t;

            auto cdbn = std::make_unique<cdbn_t>();

            cdbn->template layer<0>().learning_rate /= 10;
            //cdbn->template layer<1>().learning_rate /= 10;

            std::cout << cdbn->output_size() << " output features" << std::endl;

            constexpr const std::size_t stride = 5;
            constexpr const auto patches = ((WIDTH / 3) - NV) / stride + 1;

            //TODO Normally every image should be in the correct matrix dimensions

            std::cout << "Generate images (reverse) ..." << std::endl;

            std::vector<etl::dyn_matrix<weight>> training_images_reverse;
            training_images_reverse.reserve(training_images.size());

            for(auto& image : training_images){
                training_images_reverse.emplace_back(HEIGHT / 3, WIDTH / 3);

                auto& image_reverse = training_images_reverse.back();

                for(std::size_t y = 0; y < HEIGHT / 3; ++y){
                    for(std::size_t x = 0; x < WIDTH / 3; ++x){
                        image_reverse(y, x) = image(x, y);
                    }
                }
            }

            std::cout << "... done" << std::endl;

            std::vector<etl::dyn_matrix<weight>> training_patches;
            training_patches.reserve(training_images.size() * patches);

            std::cout << "Generate patches ..." << std::endl;

            for(auto& image : training_images_reverse){
                for(std::size_t i = 0; i < patches; ++i){
                    training_patches.emplace_back(NV, NV);

                    auto& patch = training_patches.back();

                    for(std::size_t y = 0; y < NV; ++y){
                        for(std::size_t x = 0; x < NV; ++x){
                            patch(y, x) = image(y, x + i * stride);
                        }
                    }
                }
            }

            std::cout << "... done" << std::endl;

            const std::string file_name("method_2_third.dat");

            cdbn->pretrain(training_patches, 10);
            cdbn->store(file_name);
            //cdbn->load(file_name);
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

    if(!conf.method_1 && !conf.method_2){
        std::cout << "error: One method must be selected" << std::endl;
        print_usage();
        return -1;
    }

    if(conf.command == "train"){
        return command_train(conf);
    }

    print_usage();

    return -1;
}
