//=======================================================================
// Copyright Baptiste Wicht 2015-2017.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "dll/rbm/rbm.hpp"
#include "dll/rbm/conv_rbm_mp.hpp"
#include "dll/dbn.hpp"

#include "ae_config.hpp" // Must be first

#include "ae_crbm.hpp"

#include "ae_evaluation.hpp"

namespace {

template<size_t N>
void hybrid_rbm_pmp_evaluate(const spot_dataset& dataset, const spot_dataset_set& set, config& conf, names train_word_names, names test_image_names, parameters params, const std::vector<image_t>& training_patches, float learning_rate, size_t epochs) {
    static constexpr size_t K = 6;
    static constexpr size_t K1 = 9;

    static constexpr size_t NH1_1 = patch_height - K1 + 1;
    static constexpr size_t NH1_2 = patch_width - K1 + 1;

    using network_t = typename dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_rbm_mp<
                1, patch_height, patch_width,
                K, K1, K1, 2,
                dll::batch_size<batch_size>,
                dll::weight_decay<dll::decay_type::L2>,
                dll::momentum
            >,
            dll::rbm<
                K * (NH1_1 / 2) * (NH1_2 / 2), N,
                dll::batch_size<batch_size>,
                dll::weight_decay<dll::decay_type::L2>,
                dll::momentum
            >
        >,
        dll::batch_mode
    >::dbn_t;

    auto net = std::make_unique<network_t>();

    net->display();
    std::cout << net->output_size() << " features" << std::endl;

    // Configure the network
    net->template layer_get<0>().learning_rate    = 1e-6;
    net->template layer_get<0>().initial_momentum = 0.9;
    net->template layer_get<0>().momentum         = 0.9;

    net->template layer_get<1>().learning_rate    = learning_rate;
    net->template layer_get<1>().initial_momentum = 0.9;
    net->template layer_get<1>().momentum         = 0.9;

    // Train as RBM
    net->pretrain(training_patches, epochs);

    auto folder = spot::evaluate_patches_ae<1, image_t>(dataset, set, conf, *net, train_word_names, test_image_names, false, params);
    std::cout << "AE-Result: Hybrid-RBM-PMP(" << N << "):" << folder << std::endl;
}

} // end of anonymous namespace

void hybrid_rbm_pmp_evaluate_all(const spot_dataset& dataset, const spot_dataset_set& set, config& conf, names train_word_names, names test_image_names, parameters params, const std::vector<image_t>& training_patches){
    if (conf.hybrid && conf.rbm) {
        const auto lr = 1e-3;

        hybrid_rbm_pmp_evaluate<10>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        hybrid_rbm_pmp_evaluate<20>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        hybrid_rbm_pmp_evaluate<30>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        hybrid_rbm_pmp_evaluate<40>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        hybrid_rbm_pmp_evaluate<50>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        hybrid_rbm_pmp_evaluate<60>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        hybrid_rbm_pmp_evaluate<70>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        hybrid_rbm_pmp_evaluate<80>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        hybrid_rbm_pmp_evaluate<90>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
        hybrid_rbm_pmp_evaluate<100>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, lr, epochs);
    }
}
