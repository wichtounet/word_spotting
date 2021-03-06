//=======================================================================
// Copyright Baptiste Wicht 2015-2017.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef SPOTTER_NO_AE

#include "dll/rbm/conv_rbm.hpp"
#include "dll/pooling/mp_layer.hpp"
#include "dll/dbn.hpp"

#include "ae_config.hpp" // Must be first

#include "ae_crbm.hpp"

#include "ae_evaluation.hpp"

namespace {

template<size_t K>
void deep_crbm_mp_evaluate(const spot_dataset& dataset, const spot_dataset_set& set, config& conf, names train_word_names, names test_image_names, parameters params, const std::vector<image_t>& training_patches, float learning_rate, size_t epochs) {
    static constexpr size_t KK = 6;

    static constexpr size_t K1 = 9;
    static constexpr size_t K2 = 3;

    static constexpr size_t NH1_1 = patch_height - K1 + 1;
    static constexpr size_t NH1_2 = patch_width - K1 + 1;

    static constexpr size_t NH2_1 = (NH1_1 / 2) - K2 + 1;
    static constexpr size_t NH2_2 = (NH1_2 / 2) - K2 + 1;

    using network_t = typename dll::dbn_desc<
        dll::dbn_layers<
            typename dll::conv_rbm_desc<
                1, patch_height, patch_width,
                KK, K1, K1,
                dll::batch_size<batch_size>,
                dll::weight_decay<dll::decay_type::L2>,
                dll::momentum
            >::layer_t,
            dll::mp_3d_layer<KK, NH1_1, NH1_2, 1, 2, 2>,
            typename dll::conv_rbm_desc<
                KK, NH1_1 / 2, NH1_2 / 2,
                K, K2, K2,
                dll::batch_size<batch_size>,
                dll::weight_decay<dll::decay_type::L2>,
                dll::momentum
            >::layer_t,
            dll::mp_3d_layer<K, NH2_1, NH2_2, 1, 2, 2>
        >,
        dll::shuffle,
        dll::batch_mode
    >::dbn_t;

    auto net = std::make_unique<network_t>();

    net->display();
    std::cout << net->output_size() << " features" << std::endl;

    // Configure the network
    net->template layer_get<0>().learning_rate    = 1e-6;
    net->template layer_get<0>().initial_momentum = 0.9;
    net->template layer_get<0>().momentum         = 0.9;

    net->template layer_get<2>().learning_rate    = learning_rate;
    net->template layer_get<2>().initial_momentum = 0.9;
    net->template layer_get<2>().momentum         = 0.9;

    // Train as RBM
    net->pretrain(training_patches, epochs);

    auto folder = spot::evaluate_patches_ae<2, image_t>(dataset, set, conf, *net, train_word_names, test_image_names, false, params);
    std::cout << "AE-Result: Deep-CRBM-MP(2)(" << K << "):" << folder << std::endl;

    folder = spot::evaluate_patches_ae<3, image_t>(dataset, set, conf, *net, train_word_names, test_image_names, false, params);
    std::cout << "AE-Result: Deep-CRBM-MP(3)(" << K << "):" << folder << std::endl;
}

} // end of anonymous namespace

void deep_crbm_mp_evaluate_all(const spot_dataset& dataset, const spot_dataset_set& set, config& conf, names train_word_names, names test_image_names, parameters params, const std::vector<image_t>& training_patches){
    if (conf.crbm && conf.deep) {
        const auto lr = 1e-3;

        deep_crbm_mp_evaluate<1>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        deep_crbm_mp_evaluate<2>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        deep_crbm_mp_evaluate<3>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        deep_crbm_mp_evaluate<4>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        deep_crbm_mp_evaluate<5>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        deep_crbm_mp_evaluate<6>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        deep_crbm_mp_evaluate<7>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        deep_crbm_mp_evaluate<8>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        deep_crbm_mp_evaluate<9>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        deep_crbm_mp_evaluate<10>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
    }
}

#endif // SPOTTER_NO_AE
