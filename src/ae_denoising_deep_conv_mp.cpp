//=======================================================================
// Copyright Baptiste Wicht 2015-2017.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "dll/neural/conv_layer.hpp"
#include "dll/neural/deconv_layer.hpp"
#include "dll/pooling/mp_layer.hpp"
#include "dll/pooling/upsample_layer.hpp"
#include "dll/dbn.hpp"
#include "dll/trainer/stochastic_gradient_descent.hpp"

#include "ae_config.hpp" // Must be first

#include "ae_conv.hpp"

#include "ae_evaluation.hpp"

namespace {

void denoising_deep_conv_mp_evaluate(double noise, const spot_dataset& dataset, const spot_dataset_set& set, config& conf, names train_word_names, names test_image_names, parameters params, const std::vector<image_t>& training_patches, float learning_rate, size_t epochs) {
    static constexpr size_t KK = 6;
    static constexpr size_t K = 5;

    static constexpr size_t K1 = 9;
    static constexpr size_t K2 = 3;

    static constexpr size_t NH1_1 = patch_height - K1 + 1;
    static constexpr size_t NH1_2 = patch_width - K1 + 1;

    static constexpr size_t NH2_1 = (NH1_1 / 2) - K2 + 1;
    static constexpr size_t NH2_2 = (NH1_2 / 2) - K2 + 1;

    using network_t = dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_desc<1, patch_height, patch_width, KK, NH1_1, NH1_2>::layer_t,
            dll::mp_layer_3d_desc<KK, NH1_1, NH1_2, 1, 2, 2>::layer_t,

            dll::conv_desc<KK, NH1_1 / 2, NH1_2 / 2, K, NH2_1, NH2_2>::layer_t,
            dll::mp_layer_3d_desc<K, NH2_1, NH2_2, 1, 2, 2>::layer_t,

            dll::upsample_layer_3d_desc<K, NH2_1 / 2, NH2_2 / 2, 1, 2, 2>::layer_t,
            dll::deconv_desc<K, NH2_1, NH2_2, KK, K2, K2>::layer_t,

            dll::upsample_layer_3d_desc<KK, NH1_1 / 2, NH1_2 / 2, 1, 2, 2>::layer_t,
            dll::deconv_desc<KK, NH1_1, NH1_2, 1, K1, K1>::layer_t
        >,
        dll::momentum,
        dll::weight_decay<dll::decay_type::L2>,
        dll::trainer<dll::sgd_trainer>,
        dll::batch_size<batch_size>,
        dll::shuffle,
        dll::batch_mode
    >::dbn_t;

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

    // Results before the MP
    auto folder = spot::evaluate_patches_ae<3, image_t>(dataset, set, conf, *net, train_word_names, test_image_names, false, params);
    std::cout << "AE-Result: Denoising-Deep-Conv-MP(" << noise << "):" << folder << std::endl;
}

} // end of anonymous namespace

void denoising_deep_conv_mp_evaluate_all(const spot_dataset& dataset, const spot_dataset_set& set, config& conf, names train_word_names, names test_image_names, parameters params, const std::vector<image_t>& training_patches){
    if (conf.denoising && !conf.rbm) {
        const auto lr = 1e-3;

        denoising_deep_conv_mp_evaluate(0.0, dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        denoising_deep_conv_mp_evaluate(0.05, dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        denoising_deep_conv_mp_evaluate(0.10, dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        denoising_deep_conv_mp_evaluate(0.15, dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        denoising_deep_conv_mp_evaluate(0.20, dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        denoising_deep_conv_mp_evaluate(0.25, dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        denoising_deep_conv_mp_evaluate(0.30, dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        denoising_deep_conv_mp_evaluate(0.35, dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        denoising_deep_conv_mp_evaluate(0.40, dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        denoising_deep_conv_mp_evaluate(0.45, dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        denoising_deep_conv_mp_evaluate(0.50, dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
    }
}