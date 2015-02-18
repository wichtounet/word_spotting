//=======================================================================
// Copyright Baptiste Wicht 2015.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================


#include <iostream>

#include "config.hpp"
#include "washington.hpp"

namespace {

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
    std::vector<std::string> test_image_names;
    std::vector<std::string> valid_image_names;

    for(auto& word_image : dataset.word_images){
        auto& name = word_image.first;
        for(auto& train_image : set.train_set){
            if(name.find(std::string(train_image.begin(), train_image.end() - 4)) == 0){
                train_image_names.push_back(name);
                break;
            }
        }
        for(auto& test_image : set.test_set){
            if(name.find(std::string(test_image.begin(), test_image.end() - 4)) == 0){
                test_image_names.push_back(name);
                break;
            }
        }
        for(auto& valid_image : set.validation_set){
            if(name.find(std::string(valid_image.begin(), valid_image.end() - 4)) == 0){
                valid_image_names.push_back(name);
                break;
            }
        }
    }

    std::cout << train_image_names.size() << " training word images in set" << std::endl;
    std::cout << valid_image_names.size() << " validation word images in set" << std::endl;
    std::cout << test_image_names.size() << " test word images in set" << std::endl;

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
