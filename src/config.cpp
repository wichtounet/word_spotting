//=======================================================================
// Copyright Baptiste Wicht 2015.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>

#include "config.hpp"

void print_usage(){
    std::cout << "Usage: spotter [options] <command> file [file...]" << std::endl;
    std::cout << "Supported commands: " << std::endl;
    std::cout << " * train" << std::endl;
    std::cout << "Supported options: " << std::endl;
    std::cout << " -1 : Method 1" << std::endl;
    std::cout << " -half : Takes half resolution images only" << std::endl;
    std::cout << " -quarter : Takes quarter resolution images only" << std::endl;
    std::cout << " -third : Takes third resolution images only" << std::endl;
    std::cout << " -svm : Use a SVM" << std::endl;
    std::cout << " -view : Load the DBN and visualize its weights" << std::endl;
    std::cout << " -sub : Takes only a subset of the dataset to train" << std::endl;
}

config parse_args(int argc, char** argv){
    config conf;

    for(std::size_t i = 1; i < static_cast<size_t>(argc); ++i){
        conf.args.emplace_back(argv[i]);
    }

    std::size_t i = 0;
    for(; i < conf.args.size(); ++i){
        if(conf.args[i] == "-1"){
            conf.method_1 = true;
        } else if(conf.args[i] == "-half"){
            conf.half = true;
        } else if(conf.args[i] == "-quarter"){
            conf.quarter = true;
        } else if(conf.args[i] == "-third"){
            conf.third = true;
        } else if(conf.args[i] == "-svm"){
            conf.svm = true;
        } else if(conf.args[i] == "-view"){
            conf.view = true;
        } else if(conf.args[i] == "-sub"){
            conf.sub = true;
        } else {
            break;
        }
    }

    //TODO Ensure that methods are mutually exclusive

    conf.command = conf.args[i++];

    for(; i < conf.args.size(); ++i){
        conf.files.push_back(conf.args[i]);
    }

    return conf;
}
