//=======================================================================
// Copyright Baptiste Wicht 2015.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>

#include "config.hpp"
#include "washington.hpp"   //Dataset handling

//Include methods
#include "standard.hpp"     //Method 0
#include "holistic.hpp"     //Method 1
#include "patches.hpp"      //Method 2

namespace {

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
        standard_method(dataset, set, conf, train_word_names, train_image_names, valid_image_names, test_image_names);
    } else if(conf.method_1){
        holistic_method(dataset, set, conf, train_word_names, train_image_names, valid_image_names, test_image_names);
    } else if(conf.method_2){
        patches_method(dataset, set, conf, train_word_names, train_image_names, valid_image_names, test_image_names);
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
