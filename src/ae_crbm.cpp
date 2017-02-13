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

#include "dll/rbm/conv_rbm.hpp"
#include "dll/dbn.hpp"

#include "ae_config.hpp" // Must be first

#include "ae_crbm.hpp"
#include "evaluation.hpp"    //Evaluation utilities
#include "features.hpp"      //Features exporting
#include "normalization.hpp" //Normalization functions
#include "reports.hpp"
#include "reports.hpp"
#include "standard.hpp"
#include "utils.hpp"

#include "ae_evaluation.hpp"

namespace {

//TODO

} // end of anonymous namespace

void crbm_evaluate_all(const spot_dataset& dataset, const spot_dataset_set& set, config& conf, names train_word_names, names test_image_names, parameters params, const std::vector<image_t>& training_patches){
    if (conf.crbm) {
        //TODO
    }
}
