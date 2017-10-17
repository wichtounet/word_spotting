//=======================================================================
// Copyright Baptiste Wicht 2015-2017.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef SPOTTER_NO_AE

#include "dll/neural/conv_layer.hpp"
#include "dll/neural/deconv_layer.hpp"
#include "dll/pooling/mp_layer.hpp"
#include "dll/pooling/upsample_layer.hpp"
#include "dll/dbn.hpp"
#include "dll/trainer/stochastic_gradient_descent.hpp"
#include "dll/util/flatten.hpp"

#include "ae_config.hpp" // Must be first

#include "ae_conv.hpp"

#include "ae_evaluation.hpp"

namespace {

template<size_t Noise>
void denoising_stacked_conv_mp_evaluate(const spot_dataset& dataset, const spot_dataset_set& set, config& conf, names train_word_names, names test_image_names, parameters params, const std::vector<image_t>& training_patches, float learning_rate, size_t epochs) {
    static constexpr size_t KK = 6;
    static constexpr size_t K = 5;

    static constexpr size_t K1 = 9;
    static constexpr size_t K2 = 3;

    static constexpr size_t NH1_1 = patch_height - K1 + 1;
    static constexpr size_t NH1_2 = patch_width - K1 + 1;

    static constexpr size_t NH2_1 = (NH1_1 / 2) - K2 + 1;
    static constexpr size_t NH2_2 = (NH1_2 / 2) - K2 + 1;

    using network1_t = typename dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_layer<1, patch_height, patch_width, KK, K1, K1>,
            dll::mp_3d_layer<KK, NH1_1, NH1_2, 1, 2, 2>,
            dll::upsample_3d_layer<KK, NH1_1 / 2, NH1_2 / 2, 1, 2, 2>,
            dll::deconv_layer<KK, NH1_1, NH1_2, 1, K1, K1>
        >,
        dll::updater<dll::updater_type::MOMENTUM>,
        dll::weight_decay<dll::decay_type::L2>,
        dll::trainer<dll::sgd_trainer>,
        dll::batch_size<batch_size>,
        dll::shuffle,
        dll::noise<Noise>
    >::dbn_t;

    using network2_t = typename dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_layer<KK, NH1_1 / 2, NH1_2 / 2, K, K2, K2>,
            dll::mp_3d_layer<K, NH2_1, NH2_2, 1, 2, 2>,
            dll::upsample_3d_layer<K, NH2_1 / 2, NH2_2 / 2, 1, 2, 2>,
            dll::deconv_layer<K, NH2_1, NH2_2, KK, K2, K2>
        >,
        dll::updater<dll::updater_type::MOMENTUM>,
        dll::weight_decay<dll::decay_type::L2>,
        dll::trainer<dll::sgd_trainer>,
        dll::batch_size<batch_size>,
        dll::shuffle,
        dll::batch_mode,
        dll::noise<Noise>
    >::dbn_t;

    auto net1 = std::make_unique<network1_t>();

    net1->display();

    // Configure the network
    net1->learning_rate    = learning_rate;
    net1->initial_momentum = 0.9;
    net1->momentum         = 0.9;

    // Train as autoencoder
    net1->fine_tune_ae(training_patches, epochs);

    // Extract intermediate features

    thread_pool pool;
    auto int_features = spot::prepare_outputs_ae<1, image_t>(pool, dataset, *net1, conf, test_image_names, false, false);
    auto int_features_flat = dll::flatten(int_features);

    // Configure the second network

    auto net2 = std::make_unique<network2_t>();

    net2->display();

    // Configure the network
    net2->learning_rate    = learning_rate;
    net2->initial_momentum = 0.9;
    net2->momentum         = 0.9;

    // Train as autoencoder
    net2->fine_tune_ae(int_features_flat, epochs);

    auto folder = spot::evaluate_patches_ae_stacked_2<1, 1, image_t>(dataset, set, conf, *net1, *net2, train_word_names, test_image_names, false, params);
    std::cout << "AE-Result: Denoising-Stacked-Conv-MP(" << Noise << "):" << folder << std::endl;
}

} // end of anonymous namespace

void denoising_stacked_conv_mp_evaluate_all(const spot_dataset& dataset, const spot_dataset_set& set, config& conf, names train_word_names, names test_image_names, parameters params, const std::vector<image_t>& training_patches){
    if (conf.denoising && !conf.rbm) {
        const auto lr = 1e-3;

        denoising_stacked_conv_mp_evaluate<0>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        denoising_stacked_conv_mp_evaluate<5>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        denoising_stacked_conv_mp_evaluate<10>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        denoising_stacked_conv_mp_evaluate<15>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        denoising_stacked_conv_mp_evaluate<20>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        denoising_stacked_conv_mp_evaluate<25>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        denoising_stacked_conv_mp_evaluate<30>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        denoising_stacked_conv_mp_evaluate<35>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        denoising_stacked_conv_mp_evaluate<40>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        denoising_stacked_conv_mp_evaluate<45>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        denoising_stacked_conv_mp_evaluate<50>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
    }
}

#endif // SPOTTER_NO_AE
