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
void crbm_mp_evaluate(const spot_dataset& dataset, const spot_dataset_set& set, config& conf, names train_word_names, names test_image_names, parameters params, const std::vector<image_t>& training_patches, float learning_rate, size_t epochs) {
    static constexpr size_t K1 = 17;

    static constexpr size_t NH1_1 = patch_height - K1 + 1;
    static constexpr size_t NH1_2 = patch_width - K1 + 1;

    using network_t = typename dll::dbn_desc<
        dll::dbn_layers<
            typename dll::conv_rbm_desc<
                1, patch_height, patch_width,
                K, K1, K1,
                dll::batch_size<batch_size>,
                dll::weight_decay<dll::decay_type::L2>,
                dll::momentum,
                dll::shuffle
            >::layer_t
            , dll::mp_3d_layer<K, NH1_1, NH1_2, 1, 2, 2>
    >>::dbn_t;

    auto net = std::make_unique<network_t>();

    net->display();
    std::cout << net->output_size() << " features" << std::endl;

    // Configure the network
    net->template layer_get<0>().learning_rate    = learning_rate;
    net->template layer_get<0>().initial_momentum = 0.9;
    net->template layer_get<0>().momentum         = 0.9;

    // Train as RBM
    net->pretrain(training_patches, epochs);

    auto folder = spot::evaluate_patches_ae<1, image_t>(dataset, set, conf, *net, train_word_names, test_image_names, false, params);
    std::cout << "AE-Result: CRBM_MP(" << K << "):" << folder << std::endl;
}

} // end of anonymous namespace

void crbm_mp_evaluate_all(const spot_dataset& dataset, const spot_dataset_set& set, config& conf, names train_word_names, names test_image_names, parameters params, const std::vector<image_t>& training_patches){
    if (conf.crbm && !conf.deep) {
        crbm_mp_evaluate<1>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, 1e-3, epochs);
        crbm_mp_evaluate<2>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, 1e-3, epochs);
        crbm_mp_evaluate<3>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, 1e-4, epochs);
        crbm_mp_evaluate<4>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, 1e-5, epochs);
        crbm_mp_evaluate<5>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, 1e-5, epochs);
        crbm_mp_evaluate<6>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, 1e-6, epochs);
        crbm_mp_evaluate<7>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, 1e-6, epochs);
        crbm_mp_evaluate<8>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, 1e-6, epochs);
        crbm_mp_evaluate<9>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, 1e-6, epochs);
        crbm_mp_evaluate<10>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, 1e-6, epochs);
    }
}

#endif // SPOTTER_NO_AE
