//=======================================================================
// Copyright Baptiste Wicht 2015-2016.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "dll/rbm/rbm.hpp"
#include "dll/dbn.hpp"

#include "ae_config.hpp" // Must be first

#include "ae_rbm.hpp"

#include "ae_evaluation.hpp"

namespace {

template<size_t N>
void deep_rbm_evaluate(const spot_dataset& dataset, const spot_dataset_set& set, config& conf, names train_word_names, names test_image_names, parameters params, const std::vector<image_t>& training_patches, float learning_rate, size_t epochs) {
    static constexpr const size_t M = 500;

    using network_t = typename dll::dbn_desc<
        dll::dbn_layers<
            typename dll::rbm_desc<
                patch_height * patch_width, M,
                dll::batch_size<batch_size>,
                dll::weight_decay<dll::decay_type::L2>,
                dll::momentum,
                dll::shuffle
            >::layer_t,
            typename dll::rbm_desc<
                M, N,
                dll::batch_size<batch_size>,
                dll::weight_decay<dll::decay_type::L2>,
                dll::momentum,
                dll::shuffle
            >::layer_t
    >>::dbn_t;

    auto net = std::make_unique<network_t>();

    net->display();

    // Configure the first
    net->template layer_get<0>().learning_rate    = learning_rate;
    net->template layer_get<0>().initial_momentum = 0.9;
    net->template layer_get<0>().momentum         = 0.9;

    // Configure the second layer
    net->template layer_get<1>().learning_rate    = learning_rate;
    net->template layer_get<1>().initial_momentum = 0.9;
    net->template layer_get<1>().momentum         = 0.9;

    // Train as RBM
    net->pretrain(training_patches, epochs);

    auto folder = spot::evaluate_patches_ae<0, image_t>(dataset, set, conf, *net, train_word_names, test_image_names, false, params);
    std::cout << "AE-Result: Deep-RBM(" << N << "):" << folder << std::endl;
}

} // end of anonymous namespace

void deep_rbm_evaluate_all(const spot_dataset& dataset, const spot_dataset_set& set, config& conf, names train_word_names, names test_image_names, parameters params, const std::vector<image_t>& training_patches){
    if (conf.rbm && conf.deep) {
        deep_rbm_evaluate<10>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, 1e-3, epochs);
        deep_rbm_evaluate<50>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, 1e-3, epochs);
        deep_rbm_evaluate<100>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, 1e-4, epochs);
        deep_rbm_evaluate<200>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, 1e-4, epochs);
        deep_rbm_evaluate<300>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, 1e-5, epochs);
        deep_rbm_evaluate<400>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, 1e-5, epochs);
        deep_rbm_evaluate<500>(dataset, set, conf, train_word_names, test_image_names, params, training_patches, 1e-5, epochs);
    }
}
