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
    if(conf.files.empty()){
        std::cout << "Train needs the path to the dataset" << std::endl;
        return -1;
    }

    auto dataset = read_dataset(conf.files.front());

    std::cout << dataset.line_images.size() << " line images loaded" << std::endl;
    std::cout << dataset.word_images.size() << " word images loaded" << std::endl;

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
