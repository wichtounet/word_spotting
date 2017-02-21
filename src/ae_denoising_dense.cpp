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

#include "ae_evaluation.hpp"

namespace {

using network_t = dll::dbn_desc<
    dll::dbn_layers<
        dll::dense_desc<patch_height * patch_width, 50>::layer_t,
        dll::dense_desc<50, patch_height * patch_width>::layer_t
    >,
    dll::momentum,
    dll::weight_decay<dll::decay_type::L2>,
    dll::trainer<dll::sgd_trainer>,
    dll::batch_size<batch_size>,
    dll::shuffle
>::dbn_t;

void denoising_dense_evaluate(double noise, const spot_dataset& dataset, const spot_dataset_set& set, config& conf, names train_word_names, names test_image_names, parameters params, const std::vector<image_t>& training_patches, float learning_rate, size_t epochs) {
    auto net = std::make_unique<network_t>();

    net->display();

    // Configure the network
    net->learning_rate    = learning_rate;
    net->initial_momentum = 0.9;
    net->momentum         = 0.9;

    // Train as autoencoder
    if (noise == 0.0) {
        net->fine_tune_ae(training_patches, epochs);
    } else {
        net->fine_tune_dae(training_patches, epochs, noise);
    }

    auto folder = spot::evaluate_patches_ae<0, image_t>(dataset, set, conf, *net, train_word_names, test_image_names, false, params);
    std::cout << "AE-Result: Denoising-Dense(" << noise << "):" << folder << std::endl;
}

} // end of anonymous namespace

void denoising_dense_evaluate_all(const spot_dataset& dataset, const spot_dataset_set& set, config& conf, names train_word_names, names test_image_names, parameters params, const std::vector<image_t>& training_patches){
    if (conf.denoising && !conf.rbm) {
        auto lr = 1e-3;

        denoising_dense_evaluate(0.0, dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        denoising_dense_evaluate(0.05, dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        denoising_dense_evaluate(0.10, dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        denoising_dense_evaluate(0.15, dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        denoising_dense_evaluate(0.20, dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        denoising_dense_evaluate(0.25, dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        denoising_dense_evaluate(0.30, dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        denoising_dense_evaluate(0.35, dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        denoising_dense_evaluate(0.40, dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        denoising_dense_evaluate(0.45, dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        denoising_dense_evaluate(0.50, dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
    }
}
