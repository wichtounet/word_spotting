//=======================================================================
// Copyright Baptiste Wicht 2015-2017.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "dll/neural/dense_layer.hpp"
#include "dll/dbn.hpp"
#include "dll/trainer/stochastic_gradient_descent.hpp"

#include "ae_config.hpp" // Must be first

#include "ae_dense.hpp"

#include "ae_evaluation.hpp"

namespace {

template<size_t N>
void deep_dense_evaluate(const spot_dataset& dataset, const spot_dataset_set& set, config& conf, names train_word_names, names test_image_names, parameters params, const std::vector<image_t>& training_patches, float learning_rate, size_t epochs) {
    static constexpr const size_t M = 500;

    using network_t = typename dll::dbn_desc<
        dll::dbn_layers<
            typename dll::dense_desc<patch_height * patch_width, M>::layer_t,
            typename dll::dense_desc<M, N>::layer_t,
            typename dll::dense_desc<N, M>::layer_t,
            typename dll::dense_desc<M, patch_height * patch_width>::layer_t
        >,
        dll::momentum,
        dll::weight_decay<dll::decay_type::L2>,
        dll::trainer<dll::sgd_trainer>,
        dll::batch_size<batch_size>,
        dll::shuffle
    >::dbn_t;

    auto net = std::make_unique<network_t>();

    net->display();

    // Configure the network
    net->learning_rate    = learning_rate;
    net->initial_momentum = 0.9;
    net->momentum         = 0.9;

    // Train as autoencoder
    net->fine_tune_ae(training_patches, epochs);

    auto folder = spot::evaluate_patches_ae<1, image_t>(dataset, set, conf, *net, train_word_names, test_image_names, false, params);
    std::cout << "AE-Result: Deep-Dense(" << N << "):" << folder << std::endl;
}

} // end of anonymous namespace

void deep_dense_evaluate_all(const spot_dataset& dataset, const spot_dataset_set& set, config& conf, names train_word_names, names test_image_names, parameters params, const std::vector<image_t>& training_patches){
    if (conf.dense && !conf.deep) {
        deep_dense_evaluate<10>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, 1e-3, epochs);
        deep_dense_evaluate<50>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, 1e-3, epochs);
        deep_dense_evaluate<100>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, 1e-4, epochs);
        deep_dense_evaluate<200>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, 1e-4, epochs);
        deep_dense_evaluate<300>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, 1e-5, epochs);
        deep_dense_evaluate<400>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, 1e-5, epochs);
        deep_dense_evaluate<500>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, 1e-5, epochs);
    }
}
