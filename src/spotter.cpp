//=======================================================================
// Copyright Baptiste Wicht 2015.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================


#include <iostream>

#include "dll/conv_rbm.hpp"
#include "dll/conv_rbm_mp.hpp"

#include "cpp_utils/parallel.hpp"

#include "config.hpp"
#include "washington.hpp"

static constexpr const std::size_t WIDTH = 660;
static constexpr const std::size_t HEIGHT = 120;

constexpr const std::size_t MAX_N = 50;

static_assert(WIDTH % 2 == 0, "Width must be divisible by 2");
static_assert(HEIGHT % 2 == 0, "Height must be divisible by 2");

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

etl::dyn_matrix<double> mat_to_dyn(const config& conf, cv::Mat& image){
    cv::Mat normalized(cv::Size(WIDTH, HEIGHT), CV_8U);
    normalized = cv::Scalar(255);

    image.copyTo(normalized(cv::Rect((WIDTH - image.size().width) / 2, 0, image.size().width, HEIGHT)));

    if(conf.half){
        cv::Mat scaled_normalized(cv::Size(WIDTH / 2, HEIGHT / 2), CV_8U);
        cv::resize(normalized, scaled_normalized, scaled_normalized.size(), cv::INTER_AREA);
        cv::adaptiveThreshold(scaled_normalized, normalized, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 7, 2);
    }

    etl::dyn_matrix<double> training_image(normalized.size().width, normalized.size().height);

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

    std::vector<etl::dyn_matrix<double>> training_images;

    for(auto& name : train_image_names){
        if(training_images.size() < 2400){
            training_images.emplace_back(mat_to_dyn(conf, dataset.word_images[name]));
        }
    }

    auto evaluate = [&](auto& crbm){
        std::array<double, MAX_N + 1> recalls;
        std::array<double, MAX_N + 1> precisions;
        std::array<double, MAX_N + 1> fn;

        std::fill(recalls.begin(), recalls.end(), 0);
        std::fill(precisions.begin(), precisions.end(), 0);
        std::fill(fn.begin(), fn.end(), 0);

        std::cout << "Compute features on the test set..." << std::endl;

        std::vector<etl::dyn_matrix<double, 3>> test_features_a;
        std::vector<etl::dyn_matrix<double, 3>> test_features_s;

        for(std::size_t i = 0; i < test_image_names.size(); ++i){
            test_features_a.push_back(crbm->prepare_one_output());
            test_features_s.push_back(crbm->prepare_one_output());
        }

        cpp::default_thread_pool<> pool;

        cpp::parallel_foreach_i(pool, test_image_names.begin(), test_image_names.end(),
            [&test_features_a, &test_features_s, &crbm, &dataset, &conf](auto& test_image, std::size_t i){
                auto test_v = mat_to_dyn(conf, dataset.word_images[test_image]);

                crbm->activate_one(test_v, test_features_a[i], test_features_s[i]);
            });

        std::cout << "... done" << std::endl;

        std::cout << "Evaluate performance..." << std::endl;

        std::size_t evaluated = 0;

        for(auto& keyword : set.keywords){
            std::string training_image;
            for(auto& labels : dataset.word_labels){
                if(keyword == labels.second && std::find(train_word_names.begin(), train_word_names.end(), labels.first) != train_word_names.end()){
                    training_image = labels.first;
                    break;
                }
            }

            if(training_image.empty()){
                continue;
            }

            ++evaluated;

            auto ref_v = mat_to_dyn(conf, dataset.word_images[training_image + ".png"]);
            auto ref_a = crbm->prepare_one_output();
            auto ref_s = crbm->prepare_one_output();

            crbm->activate_one(ref_v, ref_a, ref_s);

            std::vector<std::pair<std::string, double>> diffs_a;
            std::vector<std::pair<std::string, double>> diffs_s;

            for(std::size_t t = 0; t < test_image_names.size(); ++t){
                decltype(auto) test_image = test_image_names[t];

                double diff_a = etl::sum(etl::abs(ref_a - test_features_a[t]));
                diffs_a.emplace_back(std::string(test_image.begin(), test_image.end() - 4), diff_a);

                double diff_s = etl::sum(etl::abs(ref_s - test_features_s[t]));
                diffs_s.emplace_back(std::string(test_image.begin(), test_image.end() - 4), diff_s);
            }

            std::sort(diffs_a.begin(), diffs_a.end(), [](auto& a, auto& b){ return a.second < b.second; });
            std::sort(diffs_s.begin(), diffs_s.end(), [](auto& a, auto& b){ return a.second < b.second; });

            auto total_positive = std::count_if(diffs_a.begin(), diffs_a.end(),
                [&dataset, &keyword](auto& d){ return dataset.word_labels[d.first] == keyword; });

            std::cout << "Best diff(a):" << diffs_a.front().second << std::endl;
            std::cout << "Worst diff(a):" << diffs_a.back().second << std::endl;
            std::cout << "Best diff(s):" << diffs_s.front().second << std::endl;
            std::cout << "Worst diff(s):" << diffs_s.back().second << std::endl;

            for(std::size_t n = 1; n <= MAX_N; ++n){
                int tp = 0;

                for(std::size_t i = 0; i < n && i < diffs_a.size(); ++i){
                    if(dataset.word_labels[diffs_a[i].first] == keyword){
                        ++tp;
                    }
                }

                recalls[n] += tp;
                precisions[n] += tp;
                fn[n] += total_positive - tp;
            }
        }

        std::cout << "... done" << std::endl;

        std::cout << evaluated << " keywords evaluated" << std::endl;

        for(std::size_t n = 1; n <= MAX_N; ++n){
            std::cout << "Precision(" << n << ") = " << (precisions[n] / (n * set.keywords.size())) << std::endl;
            std::cout << "Recall(" << n << ") = " << (recalls[n] / fn[n]) << std::endl;
        }
    };

    if(conf.half){
        using crbm_t =
            dll::conv_rbm_desc<
            WIDTH / 2, HEIGHT / 2, 1                    //330x60 input image (1 channel)
            , WIDTH / 2 + 1 - 19 , HEIGHT / 2 + 1 - 19  //Configure the size of the filter
            , 12                                        //Number of feature maps
            //, 2                                       //Probabilistic max pooling (2x2)
            , dll::batch_size<25>
            , dll::parallel
            , dll::verbose
            , dll::momentum
            , dll::weight_decay<dll::decay_type::L2>
            >::rbm_t;

        auto crbm = std::make_unique<crbm_t>();

        std::cout << crbm->output_size() << " output features" << std::endl;

        crbm->train(training_images, 5);
        crbm->store("method_1_half.dat");
        //crbm->load("method_1.dat");

        evaluate(crbm);
    } else {
        using crbm_t =
            dll::conv_rbm_desc<
            WIDTH, HEIGHT, 1                     //660x120 input image (1 channel)
            , WIDTH + 1 - 19 , HEIGHT + 1 - 19   //Configure the size of the filter
            , 12                                 //Number of feature maps
            //, 2                                //Probabilistic max pooling (2x2)
            , dll::batch_size<25>
            , dll::parallel
            , dll::verbose
            , dll::momentum
            , dll::weight_decay<dll::decay_type::L2>
            >::rbm_t;

        auto crbm = std::make_unique<crbm_t>();

        std::cout << crbm->output_size() << " output features" << std::endl;

        crbm->train(training_images, 5);
        crbm->store("method_1.dat");
        //crbm->load("method_1.dat");

        evaluate(crbm);
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

    if(!conf.method_1){
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
