//=======================================================================
// Copyright Baptiste Wicht 2015-2016.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <iostream>
#include <atomic>
#include <mutex>
#include <algorithm>

#include "dll/util/timers.hpp"

#include "hmm_htk.hpp"
#include "hmm_mlpack.hpp"

using thread_pool = cpp::default_thread_pool<>;

struct parameters {
    double sc_band;
};

/*!
 * \brief Select all the training images for the given keyword
 * \param dataset The current dataset
 * \param keyword The search keyword
 * \param train_names The used train names
 * \return A vector of all the relevant training images
 */
std::vector<std::string> select_training_images(const spot_dataset& dataset, names keyword, names train_names);

template <typename Ref, typename Features, typename RefFunctor>
std::vector<std::pair<std::string, weight>> compute_distances(const config& conf,
    thread_pool& pool, const spot_dataset& dataset, Features& test_features_a, Ref& ref_a,
    names training_images, names test_image_names, names train_word_names, parameters parameters, RefFunctor functor) {
    std::vector<std::pair<std::string, weight>> diffs_a(test_image_names.size());

    if(conf.hmm && conf.htk){
        static hmm_htk::hmm_p global_hmm;
        static std::vector<double> global_likelihoods;

        //Either frakking compiler or me is too stupid, so we need this workaround
        auto& global_l = global_likelihoods;

        if(global_hmm.empty()){
            std::cout << "Prepare global HMM" << std::endl;

            std::cout << "Prepare features" << std::endl;

            hmm_htk::prepare_train_features(train_word_names, functor);
            hmm_htk::prepare_test_features(test_image_names, test_features_a);

            global_hmm = hmm_htk::train_global_hmm(conf, dataset, train_word_names);

            hmm_htk::global_likelihood_all(conf, pool, global_hmm, test_image_names, global_likelihoods);

            std::cout << ".... global done" << std::endl;
        }

        auto hmm = hmm_htk::prepare_test_keywords(dataset, training_images);

        // Compute the keywords likelihoods

        std::vector<double> keyword_likelihoods;

        hmm_htk::keyword_likelihood_all(conf, pool, global_hmm, hmm, test_image_names, keyword_likelihoods);

        // Compute the final distances

        cpp::parallel_foreach_i(pool, test_image_names.begin(), test_image_names.end(), [&](auto& test_image, std::size_t t) {
            auto best_diff_a = hmm_htk::hmm_distance(dataset, test_image, training_images, global_l[t], keyword_likelihoods[t]);

            diffs_a[t] = std::make_pair(std::string(test_image.begin(), test_image.end() - 4), best_diff_a);
        });
    } else if(conf.hmm){
        static hmm_mlpack::gmm_p global_hmm;

        if(!global_hmm){
            global_hmm = hmm_mlpack::train_global_hmm(train_word_names, functor);
        }

        auto hmm = hmm_mlpack::train_ref_hmm(dataset, ref_a, training_images);

        //Either frakking compiler or me is too stupid, so we need this
        //workaround
        auto& gmm = global_hmm;

        cpp::parallel_foreach_i(pool, test_image_names.begin(), test_image_names.end(), [&](auto& test_image, std::size_t t) {
            auto best_diff_a = hmm_mlpack::hmm_distance(dataset, gmm, hmm, test_image, test_features_a[t], training_images);

            diffs_a[t] = std::make_pair(std::string(test_image.begin(), test_image.end() - 4), best_diff_a);
        });
    } else {
        cpp::parallel_foreach_i(pool, test_image_names.begin(), test_image_names.end(), [&](auto& test_image, std::size_t t) {
            auto t_size = dataset.word_images.at(test_image).size().width;

            double best_diff_a = 100000000.0;

            for (std::size_t i = 0; i < ref_a.size(); ++i) {
                auto ref_size = dataset.word_images.at(training_images[i] + ".png").size().width;

                double diff_a;
                auto ratio = static_cast<double>(ref_size) / t_size;
                if (ratio > 2.0 || ratio < 0.5) {
                    diff_a = 100000000.0;
                } else {
                    diff_a = dtw_distance(ref_a[i], test_features_a[t], true, parameters.sc_band);
                }

                best_diff_a = std::min(best_diff_a, diff_a);
            }

            diffs_a[t] = std::make_pair(std::string(test_image.begin(), test_image.end() - 4), best_diff_a);
        });
    }

    return diffs_a;
}
