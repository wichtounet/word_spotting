//=======================================================================
// Copyright Baptiste Wicht 2015-2017.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>

#include "config.hpp"
#include "dataset.hpp" //Dataset handling

//Include methods
#include "standard.hpp" // Method 0
#include "holistic.hpp" // Method 1
#include "patches.hpp"  // Method 2
#include "ae.hpp"       // Comparison of auto encoders

//Method-Agnostic commands
#include "evaluate_features.hpp" //evaluate_patches command

#include <atomic>
#include <mutex>
#include "dll/util/timers.hpp"

namespace {

using string_vector = std::vector<std::string>;

template <typename Set>
void extract_names(const config& conf, spot_dataset& dataset, Set& set, string_vector& train_image_names, string_vector& train_word_names, string_vector& test_image_names, string_vector& valid_image_names) {
    if(conf.ak){
        test_image_names  = set.test_set;
        train_image_names = set.train_set;
        valid_image_names = set.validation_set;

        for(auto& name : set.train_set){
            train_word_names.emplace_back(name.begin(), name.end() - 4);
        }
    } else {
        std::cout << set.train_set.size() << " training line images in set" << std::endl;
        std::cout << set.validation_set.size() << " validation line images in set" << std::endl;
        std::cout << set.test_set.size() << " test line images in set" << std::endl;

        for (auto& word_image : dataset.word_images) {
            auto& name = word_image.first;

            for (auto& train_image : set.train_set) {
                if (name.find(train_image) == 0) {
                    train_image_names.push_back(name);
                    train_word_names.emplace_back(name.begin(), name.end() - 4);
                    break;
                }
            }

            for (auto& test_image : set.test_set) {
                if (name.find(test_image) == 0) {
                    test_image_names.push_back(name);
                    break;
                }
            }

            for (auto& valid_image : set.validation_set) {
                if (name.find(valid_image) == 0) {
                    valid_image_names.push_back(name);
                    break;
                }
            }
        }
    }

    std::cout << train_image_names.size() << " training word images in set" << std::endl;
    std::cout << valid_image_names.size() << " validation word images in set" << std::endl;
    std::cout << test_image_names.size() << " test word images in set" << std::endl;
}

spot_dataset read_dataset(config& conf) {
    decltype(auto) dataset_path = conf.files[0];
    decltype(auto) cv_set = conf.files[1];

    std::cout << " Dataset: " << dataset_path << std::endl;
    std::cout << "     Set: " << cv_set << std::endl;

    spot_dataset dataset;

    if(conf.washington){
        dataset = read_washington(conf, dataset_path);
    } else if(conf.parzival){
        dataset = read_parzival(conf, dataset_path);
    } else if(conf.iam){
        dataset = read_iam(conf, dataset_path);
    } else if(conf.manmatha){
        dataset = read_manmatha(conf, dataset_path);
    } else if(conf.ak){
        dataset = read_ak(conf, dataset_path);
    } else if(conf.botany){
        dataset = read_botany(conf, dataset_path);
    } else {
        std::cerr << "Invalid configuration of the dataset" << std::endl;
    }

    std::cout << dataset.line_images.size() << " line images loaded from the dataset" << std::endl;
    std::cout << dataset.word_images.size() << " word images loaded from the dataset" << std::endl;

    if (conf.washington || conf.manmatha) {
        conf.cv_full_path = dataset_path + "/sets/" + cv_set + "/";

        if (conf.gray) {
            conf.data_full_path = dataset_path + "/data/word_gray/";
        } else if (conf.binary) {
            conf.data_full_path = dataset_path + "/data/word_binary/";
        } else {
            conf.data_full_path = dataset_path + "/data/word_images_normalized/";
        }
    } else if (conf.parzival) {
        conf.cv_full_path   = dataset_path + "/sets1/";
        conf.data_full_path = dataset_path + "/data/word_images_normalized/";
    } else if (conf.iam) {
        conf.cv_full_path   = dataset_path + "/sets/";
        conf.data_full_path = dataset_path + "/data/word_images_normalized/";
    } else if (conf.ak || conf.botany) {
        conf.cv_full_path   = dataset_path + "/sets/";
        conf.data_full_path = dataset_path + "/data/word_images_normalized/";
    }

    return dataset;
}

int command_train(config& conf) {
    if (conf.files.size() < 2) {
        std::cout << "Train needs the path to the dataset and the cv set to use" << std::endl;
        return -1;
    }

    auto dataset = read_dataset(conf);

    decltype(auto) cv_set = conf.files[1];

    if (!dataset.sets.count(cv_set)) {
        std::cout << "The subset \"" << cv_set << "\" does not exist" << std::endl;
        return -1;
    }

    auto& set = dataset.sets[cv_set];

    string_vector train_image_names, train_word_names, test_image_names, valid_image_names;

    extract_names(conf, dataset, set, train_image_names, train_word_names, test_image_names, valid_image_names);

    switch (conf.method) {
        case Method::Marti2001:
        case Method::Rath2007:
        case Method::Rath2003:
        case Method::Rodriguez2008:
        case Method::Vinciarelli2004:
        case Method::Terasawa2009:
            standard_train(dataset, set, conf, train_word_names, train_image_names, valid_image_names, test_image_names);
            break;
        case Method::Holistic:
            holistic_train(dataset, set, conf, train_word_names, train_image_names, valid_image_names, test_image_names);
            break;
        case Method::Patches:
            patches_train(dataset, set, conf, train_word_names, train_image_names, valid_image_names, test_image_names);
            break;
        case Method::AE:
            ae_train(dataset, set, conf, train_word_names, train_image_names, test_image_names);
            break;
    }

    dll::dump_timers();

    return 0;
}

int command_features(config& conf) {
    if (conf.files.size() < 2) {
        std::cout << "features needs the path to the dataset and the cv set to use" << std::endl;
        return -1;
    }

    auto dataset = read_dataset(conf);

    decltype(auto) cv_set = conf.files[1];

    if (!dataset.sets.count(cv_set)) {
        std::cout << "The subset \"" << cv_set << "\" does not exist" << std::endl;
        return -1;
    }

    auto& set = dataset.sets[cv_set];

    string_vector train_image_names, train_word_names, test_image_names, valid_image_names;

    extract_names(conf, dataset, set, train_image_names, train_word_names, test_image_names, valid_image_names);

    switch (conf.method) {
        case Method::Marti2001:
        case Method::Rath2007:
        case Method::Rath2003:
        case Method::Rodriguez2008:
        case Method::Vinciarelli2004:
        case Method::Terasawa2009:
            standard_features(dataset, set, conf, train_word_names, train_image_names, valid_image_names, test_image_names);
            break;
        case Method::Holistic:
            holistic_features(dataset, set, conf, train_word_names, train_image_names, valid_image_names, test_image_names);
            break;
        case Method::Patches:
            patches_features(dataset, set, conf, train_word_names, train_image_names, valid_image_names, test_image_names);
            break;
        case Method::AE:
            std::cerr << "AE method does not support feature extraction" << std::endl;
            break;
    }

    return 0;
}

std::size_t bench_runtime_std(config& conf, Method method, const spot_dataset& dataset, names image_names){
    auto start = std::chrono::steady_clock::now();

    conf.method = method;
    standard_runtime(dataset, conf, image_names);

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    std::cout << "   Total: " << (double(duration) / 1000 / 1000) << "ms" << std::endl;
    std::cout << "   Image: " << (double(duration) / image_names.size() / 1000) << "us" << std::endl;

    return duration;
}

std::size_t bench_runtime_patches(config& conf, const spot_dataset& dataset, const spot_dataset_set& set, names train_word_names, names image_names){
    auto start = std::chrono::steady_clock::now();

    patches_runtime(dataset, set, conf, train_word_names, image_names);

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    std::cout << "   Total: " << (double(duration) / 1000 / 1000) << "ms" << std::endl;
    std::cout << "   Image: " << (double(duration) / image_names.size() / 1000) << "us" << std::endl;

    return duration;
}

int command_runtime(config& conf) {
    if (conf.files.size() < 2) {
        std::cout << "runtime needs the path to the dataset and the cv set to use" << std::endl;
        return -1;
    }

    auto dataset = read_dataset(conf);

    decltype(auto) cv_set = conf.files[1];

    if (!dataset.sets.count(cv_set)) {
        std::cout << "The subset \"" << cv_set << "\" does not exist" << std::endl;
        return -1;
    }

    auto& set = dataset.sets[cv_set];

    string_vector train_image_names, train_word_names, test_image_names, valid_image_names;

    extract_names(conf, dataset, set, train_image_names, train_word_names, test_image_names, valid_image_names);

    std::cout << std::endl << "Benchmark the running time of feature extraction" << std::endl;

    std::cout << std::endl << "Marti2001" << std::endl;
    bench_runtime_std(conf, Method::Marti2001, dataset, train_image_names);

    std::cout << std::endl << "Rodriguez2008" << std::endl;
    bench_runtime_std(conf, Method::Rodriguez2008, dataset, train_image_names);

    std::cout << std::endl << "Terasawa2009" << std::endl;
    bench_runtime_std(conf, Method::Terasawa2009, dataset, train_image_names);

    std::cout << std::endl << "Patches" << std::endl;
    bench_runtime_patches(conf, dataset, set, train_word_names, train_image_names);

    return 0;
}

int command_evaluate_features(config& conf) {
    if (conf.files.size() < 2) {
        std::cout << "evaluate_features needs the path to the dataset and the cv set to use" << std::endl;
        return -1;
    }

    auto dataset = read_dataset(conf);

    decltype(auto) cv_set = conf.files[1];

    if (!dataset.sets.count(cv_set)) {
        std::cout << "The subset \"" << cv_set << "\" does not exist" << std::endl;
        return -1;
    }

    auto& set = dataset.sets[cv_set];

    string_vector train_image_names, train_word_names, test_image_names, valid_image_names;

    extract_names(conf, dataset, set, train_image_names, train_word_names, test_image_names, valid_image_names);

    //TODO At this point, we need to pass the features to DTW but we ned to match them to images first for evaluation
    //     Format them just like evaluation does before DTW

    evaluate_features(dataset, set, conf, train_word_names, train_image_names, valid_image_names, test_image_names);

    return 0;
}


} //end of anonymous namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage();
        return -1;
    }

    auto conf = parse_args(argc, argv);

    if (conf.half) {
        conf.downscale = 2;
    } else if (conf.third) {
        conf.downscale = 3;
    } else if (conf.quarter) {
        conf.downscale = 4;
    }

    std::cout << std::string(11, '#') << " Keyword spotting " << std::string(11, '#') << std::endl;
    std::cout << "     Method: " << method_to_string(conf.method) << std::endl;
    std::cout << "  Grayscale: " << (conf.grayscale ? "true" : "false") << std::endl;

    if (conf.hmm && conf.htk) {
        std::cout << "       Eval: HMM (HTK)" << std::endl;
    } else if (conf.hmm) {
        std::cout << "       Eval: HMM (mlpack)" << std::endl;
    } else {
        std::cout << "       Eval: DTW" << std::endl;
    }

    bool ok = true;
    int ret = 0;

    if (conf.command == "evaluate") {
        conf.load = true;
        ret = command_train(conf);
    } else if (conf.command == "train") {
        ret = command_train(conf);
    } else if (conf.command == "features") {
        ret = command_features(conf);
    } else if (conf.command == "runtime") {
        ret = command_runtime(conf);
    } else if (conf.command == "evaluate_features") {
        ret = command_evaluate_features(conf);
    } else {
        ok = false;
    }

    std::cout << std::string(40, '#') << std::endl;

    if(!ok){
        print_usage();

        return -1;
    }

    return ret;
}
