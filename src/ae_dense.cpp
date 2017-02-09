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

#include "dll/neural/dense_layer.hpp"
#include "dll/dbn.hpp"
#include "dll/trainer/stochastic_gradient_descent.hpp"

#include "ae_config.hpp" // Must be first

#include "ae_dense.hpp"
#include "config.hpp"
#include "evaluation.hpp"    //evaluation utilities
#include "features.hpp"      //Features exporting
#include "normalization.hpp" //Normalization functions
#include "reports.hpp"
#include "standard.hpp"
#include "utils.hpp"

#include "ae_evaluation.hpp"

namespace {

template<size_t N>
void dense_evaluate(const spot_dataset& dataset, const spot_dataset_set& set, config& conf, names train_word_names, names test_image_names, parameters params, const std::vector<image_t>& training_patches) {
    using network_t = typename dll::dbn_desc<
        dll::dbn_layers<
            typename dll::dense_desc<patch_height * patch_width, N>::layer_t,
            typename dll::dense_desc<N, patch_height * patch_width>::layer_t
        >,
        dll::momentum, dll::trainer<dll::sgd_trainer>,
        dll::batch_size<batch_size>
    >::dbn_t;

    auto net = std::make_unique<network_t>();

    net->display();

    // Configure the network
    net->learning_rate    = 0.1;
    net->initial_momentum = 0.9;
    net->momentum         = 0.9;

    // Train as autoencoder
    net->fine_tune_ae(training_patches, epochs);

    auto folder = spot::evaluate_patches_ae<0, image_t>(dataset, set, conf, *net, train_word_names, test_image_names, false, params);
    std::cout << "AE-Result: Dense(" << N << "):" << folder << std::endl;
}

} // end of anonymous namespace

void dense_evaluate_all(const spot_dataset& dataset, const spot_dataset_set& set, config& conf, names train_word_names, names test_image_names, parameters params, const std::vector<image_t>& training_patches){
    if (conf.dense) {
        dense_evaluate<10>(dataset, set, conf, train_word_names, test_image_names, params, training_patches);
        dense_evaluate<50>(dataset, set, conf, train_word_names, test_image_names, params, training_patches);
        dense_evaluate<100>(dataset, set, conf, train_word_names, test_image_names, params, training_patches);
        dense_evaluate<200>(dataset, set, conf, train_word_names, test_image_names, params, training_patches);
        dense_evaluate<300>(dataset, set, conf, train_word_names, test_image_names, params, training_patches);
        dense_evaluate<400>(dataset, set, conf, train_word_names, test_image_names, params, training_patches);
        dense_evaluate<500>(dataset, set, conf, train_word_names, test_image_names, params, training_patches);
    }
}
