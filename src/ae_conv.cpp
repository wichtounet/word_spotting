//=======================================================================
// Copyright Baptiste Wicht 2015-2016.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <string>

#include "memory.hpp" //First for debug reasons

#include "cpp_utils/parallel.hpp"

#include "dll/neural/conv_layer.hpp"
#include "dll/neural/deconv_layer.hpp"
#include "dll/dbn.hpp"
#include "dll/trainer/stochastic_gradient_descent.hpp"

#include "ae_config.hpp" // Must be first

#include "ae_conv.hpp"
#include "config.hpp"
#include "evaluation.hpp"    //evaluation utilities
#include "features.hpp"      //Features exporting
#include "normalization.hpp" //Normalization functions
#include "reports.hpp"
#include "standard.hpp"
#include "utils.hpp"

#include "ae_evaluation.hpp"

namespace {

// TODO

} // end of anonymous namespace

void conv_evaluate_all(const spot_dataset& dataset, const spot_dataset_set& set, config& conf, names train_word_names, names test_image_names, parameters params, const std::vector<image_t>& training_patches){
    if (conf.conv) {
        //TODO
    }
}
