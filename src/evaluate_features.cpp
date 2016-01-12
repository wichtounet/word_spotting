//=======================================================================
// Copyright Baptiste Wicht 2015.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <string>

#include "memory.hpp"            //First for debug reasons
#include "evaluate_features.hpp" //The header of this source file
#include "patches_config.hpp"    //The different configurations

#define SUFFIX .
#define SUFFIX_CAT_I(A, B) A ## B
#define SUFFIX_CAT(A, B) SUFFIX_CAT_I(A, B)
#define STRINGIFY_I(A) #A
#define STRINGIFY(A) STRINGIFY_I(A)

namespace {

std::string get_suffix(config& conf){
    if(conf.method_0){
        return ".0";
    } else if(conf.method_1){
        return ".1";
    } else if(conf.method_2){
        if(conf.half){
            return STRINGIFY(SUFFIX_CAT(SUFFIX, HALF_LEVELS));
        } else if(conf.third){
            return STRINGIFY(SUFFIX_CAT(SUFFIX, THIRD_LEVELS));
        } else {
            return STRINGIFY(SUFFIX_CAT(SUFFIX, FULL_LEVELS));
        }
    }

    return ".invalid";
}

} // end of anonymous namespace


void evaluate_features(const spot_dataset& dataset, const spot_dataset_set& set, config& conf,
                      names train_word_names,
                      names train_image_names, names valid_image_names, names test_image_names){
    //TODO
}
