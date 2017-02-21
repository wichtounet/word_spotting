//=======================================================================
// Copyright Baptiste Wicht 2015-2017.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "dll/neural/dense_layer.hpp"
#include "dll/neural/conv_layer.hpp"
#include "dll/neural/deconv_layer.hpp"
#include "dll/dbn.hpp"
#include "dll/trainer/stochastic_gradient_descent.hpp"

#include "ae_config.hpp" // Must be first

#include "ae_conv.hpp"

#include "ae_evaluation.hpp"

namespace {

template<size_t N>
void hybrid_deep_evaluate(const spot_dataset& dataset, const spot_dataset_set& set, config& conf, names train_word_names, names test_image_names, parameters params, const std::vector<image_t>& training_patches, float learning_rate, size_t epochs) {
    static constexpr size_t KK = 6;
    static constexpr size_t K1 = 9;

    static constexpr size_t NH1_1 = patch_height - K1 + 1;
    static constexpr size_t NH1_2 = patch_width - K1 + 1;

    using network_t = typename dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_desc<1, patch_height, patch_width, KK, NH1_1, NH1_2>::layer_t,
            typename dll::dense_desc<KK * NH1_1 * NH1_2, N>::layer_t,
            typename dll::dense_desc<N, KK * NH1_1 * NH1_2>::layer_t,
            dll::deconv_desc<KK, NH1_1, NH1_2, 1, K1, K1>::layer_t
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
    std::cout << "AE-Result: Hybrid-Deep(" << N << "):" << folder << std::endl;
}

} // end of anonymous namespace

void hybrid_deep_evaluate_all(const spot_dataset& dataset, const spot_dataset_set& set, config& conf, names train_word_names, names test_image_names, parameters params, const std::vector<image_t>& training_patches){
    //TODO Enable later again
    return false;

    if (conf.hybrid && !conf.rbm) {
        const auto lr = 1e-3;

        hybrid_deep_evaluate<10>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        hybrid_deep_evaluate<20>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        hybrid_deep_evaluate<30>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        hybrid_deep_evaluate<40>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        hybrid_deep_evaluate<50>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        hybrid_deep_evaluate<60>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        hybrid_deep_evaluate<70>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        hybrid_deep_evaluate<80>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        hybrid_deep_evaluate<90>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        hybrid_deep_evaluate<100>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
    }
}
