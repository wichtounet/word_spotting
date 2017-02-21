//=======================================================================
// Copyright Baptiste Wicht 2015-2017.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "dll/neural/conv_layer.hpp"
#include "dll/neural/dense_layer.hpp"
#include "dll/neural/deconv_layer.hpp"
#include "dll/dbn.hpp"
#include "dll/trainer/stochastic_gradient_descent.hpp"

#include "ae_config.hpp" // Must be first

#include "ae_conv.hpp"

#include "ae_evaluation.hpp"

namespace {

template<size_t N>
void hybrid_stacked_evaluate(const spot_dataset& dataset, const spot_dataset_set& set, config& conf, names train_word_names, names test_image_names, parameters params, const std::vector<image_t>& training_patches, float learning_rate, size_t epochs) {
    static constexpr size_t KK = 6;
    static constexpr size_t K1 = 9;

    static constexpr size_t NH1_1 = patch_height - K1 + 1;
    static constexpr size_t NH1_2 = patch_width - K1 + 1;

    using network1_t = dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_desc<1, patch_height, patch_width, KK, NH1_1, NH1_2>::layer_t,
            dll::deconv_desc<KK, NH1_1, NH1_2, 1, K1, K1>::layer_t
        >,
        dll::momentum,
        dll::weight_decay<dll::decay_type::L2>,
        dll::trainer<dll::sgd_trainer>,
        dll::batch_size<batch_size>,
        dll::shuffle,
        dll::batch_mode
    >::dbn_t;

    using network2_t = typename dll::dbn_desc<
        dll::dbn_layers<
            typename dll::dense_desc<KK * NH1_1 * NH1_2, N>::layer_t,
            typename dll::dense_desc<N, KK * NH1_1 * NH1_2>::layer_t
        >,
        dll::momentum,
        dll::weight_decay<dll::decay_type::L2>,
        dll::trainer<dll::sgd_trainer>,
        dll::batch_size<batch_size>,
        dll::shuffle,
        dll::batch_mode
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
    auto int_features = spot::prepare_outputs_ae<0, image_t>(pool, dataset, *net1, conf, test_image_names, false, false);
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

    auto folder = spot::evaluate_patches_ae_stacked_2<0, 0, image_t>(dataset, set, conf, *net1, *net2, train_word_names, test_image_names, false, params);
    std::cout << "AE-Result: Hybrid-Stacked(" << N << "):" << folder << std::endl;
}

} // end of anonymous namespace

void hybrid_stacked_evaluate_all(const spot_dataset& dataset, const spot_dataset_set& set, config& conf, names train_word_names, names test_image_names, parameters params, const std::vector<image_t>& training_patches){
    //TODO Enable later again
    return false;

    if (conf.hybrid && !conf.rbm) {
        const auto lr = 1e-3;

        hybrid_stacked_evaluate<10>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        hybrid_stacked_evaluate<20>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        hybrid_stacked_evaluate<30>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        hybrid_stacked_evaluate<40>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        hybrid_stacked_evaluate<50>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        hybrid_stacked_evaluate<60>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        hybrid_stacked_evaluate<70>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        hybrid_stacked_evaluate<80>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        hybrid_stacked_evaluate<90>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        hybrid_stacked_evaluate<100>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
    }
}
