//=======================================================================
// Copyright Baptiste Wicht 2015-2017.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "dll/neural/conv_layer.hpp"
#include "dll/neural/deconv_layer.hpp"
#include "dll/dbn.hpp"
#include "dll/trainer/stochastic_gradient_descent.hpp"

#include "ae_config.hpp" // Must be first

#include "ae_conv.hpp"

#include "ae_evaluation.hpp"

namespace {

template<size_t K>
void deep_conv_evaluate(const spot_dataset& dataset, const spot_dataset_set& set, config& conf, names train_word_names, names test_image_names, parameters params, const std::vector<image_t>& training_patches, float learning_rate, size_t epochs) {
    static constexpr size_t KK = 6;

    static constexpr size_t K1 = 9;
    static constexpr size_t K2 = 9;

    static constexpr size_t NH1_1 = patch_height - K1 + 1;
    static constexpr size_t NH1_2 = patch_width - K1 + 1;

    static constexpr size_t NH2_1 = NH1_1 - K2 + 1;
    static constexpr size_t NH2_2 = NH1_2 - K2 + 1;

    using network_t = typename dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_desc<1, patch_height, patch_width, KK, NH1_1, NH1_2>::layer_t,
            typename dll::conv_desc<KK, NH1_1, NH1_2, K, NH2_1, NH2_2>::layer_t,
            typename dll::deconv_desc<K, NH2_1, NH2_2, KK, K2, K2>::layer_t,
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
    net->fine_tune_ae(training_patches, epochs);

    // Results before the MP
    auto folder = spot::evaluate_patches_ae<1, image_t>(dataset, set, conf, *net, train_word_names, test_image_names, false, params);
    std::cout << "AE-Result: Deep-Conv(" << K << "):" << folder << std::endl;
}

} // end of anonymous namespace

void deep_conv_evaluate_all(const spot_dataset& dataset, const spot_dataset_set& set, config& conf, names train_word_names, names test_image_names, parameters params, const std::vector<image_t>& training_patches){
    if (conf.conv && conf.deep) {
        const auto lr = 1e-3;

        deep_conv_evaluate<1>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        deep_conv_evaluate<2>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        deep_conv_evaluate<3>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        deep_conv_evaluate<4>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        deep_conv_evaluate<5>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        deep_conv_evaluate<6>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        deep_conv_evaluate<7>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        deep_conv_evaluate<8>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        deep_conv_evaluate<9>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        deep_conv_evaluate<10>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
    }
}
