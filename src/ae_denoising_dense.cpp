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

template<size_t Noise>
void denoising_dense_evaluate(const spot_dataset& dataset, const spot_dataset_set& set, config& conf, names train_word_names, names test_image_names, parameters params, const std::vector<image_t>& training_patches, float learning_rate, size_t epochs) {
    using network_t = typename dll::dbn_desc<
        dll::dbn_layers<
            dll::dense_layer<patch_height * patch_width, 50>,
            dll::dense_layer<50, patch_height * patch_width>
        >,
        dll::updater<dll::updater_type::MOMENTUM>,
        dll::weight_decay<dll::decay_type::L2>,
        dll::trainer<dll::sgd_trainer>,
        dll::batch_size<batch_size>,
        dll::shuffle,
        dll::noise<Noise>
    >::dbn_t;

    auto net = std::make_unique<network_t>();

    net->display();

    // Configure the network
    net->learning_rate    = learning_rate;
    net->initial_momentum = 0.9;
    net->momentum         = 0.9;

    // Train as autoencoder
    net->fine_tune_ae(training_patches, epochs);

    auto folder = spot::evaluate_patches_ae<0, image_t>(dataset, set, conf, *net, train_word_names, test_image_names, false, params);
    std::cout << "AE-Result: Denoising-Dense(" << Noise << "):" << folder << std::endl;
}

} // end of anonymous namespace

void denoising_dense_evaluate_all(const spot_dataset& dataset, const spot_dataset_set& set, config& conf, names train_word_names, names test_image_names, parameters params, const std::vector<image_t>& training_patches){
    if (conf.denoising && !conf.rbm) {
        auto lr = 1e-3;

        denoising_dense_evaluate<0>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        denoising_dense_evaluate<5>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        denoising_dense_evaluate<10>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        denoising_dense_evaluate<15>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        denoising_dense_evaluate<20>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        denoising_dense_evaluate<25>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        denoising_dense_evaluate<30>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        denoising_dense_evaluate<35>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        denoising_dense_evaluate<40>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        denoising_dense_evaluate<45>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        denoising_dense_evaluate<50>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
    }
}
