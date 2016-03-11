//=======================================================================
// Copyright Baptiste Wicht 2015-2016.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef WORD_SPOTTER_HMM_HPP
#define WORD_SPOTTER_HMM_HPP

#ifndef SPOTTER_NO_HMM

#include <random>

#include <mlpack/core.hpp>
#include <mlpack/methods/hmm/hmm.hpp>
#include <mlpack/methods/gmm/gmm.hpp>

namespace hmm_mlpack {

using GMM = mlpack::gmm::GMM;

template<typename Distribution>
using HMM = mlpack::hmm::HMM<Distribution>;

using gmm_p = std::unique_ptr<GMM>;
using hmm_p = std::unique_ptr<HMM<GMM>>;

//Number of gaussians for the HMM
static constexpr const std::size_t n_hmm_gaussians = 1;

//Number of gaussians for the GMM
static constexpr const std::size_t n_gmm_gaussians = 16;

//Number of states per character
static constexpr const auto n_states_per_char = 2;

static constexpr const std::size_t salts = 5;
static constexpr const double salt = 0.05;

template <typename RefFunctor>
gmm_p train_global_hmm(names train_word_names, RefFunctor functor) {
    dll::auto_timer timer("gmm_train");

    auto ref_a = functor(train_word_names);

    const auto n_features = ref_a[0][0].size();

    auto gmm = std::make_unique<GMM>(n_gmm_gaussians, n_features);

    //TODO Better Configure how the subset if selected
    std::size_t step = 5;

    //Collect information on the dataset

    std::size_t n_observations = 0;
    std::size_t n_images = 0;

    for(std::size_t image = 0; image < ref_a.size(); image += step){
        n_observations += ref_a[image].size();
        ++n_images;
    }

    //Flatten all the images

    arma::mat flatten_mat(n_features, n_observations);
    std::size_t o = 0;

    for(std::size_t image = 0; image < ref_a.size(); image += step){
        auto& ref_image = ref_a[image];

        auto width = ref_image.size();

        for(std::size_t i = 0; i < width; ++i){
            auto od = o++;
            for(std::size_t j = 0; j < n_features; ++j){
                flatten_mat(j, od) = ref_image[i][j];
            }
        }
    }

    try {
        std::cout << "Start training the GMM" << std::endl;
        std::cout << "\tn_images: " << n_images << std::endl;
        std::cout << "\tn_observations=" << n_observations << std::endl;
        std::cout << "\tn_total_features=" << n_observations * 9 << std::endl;

        gmm->Train(flatten_mat);

        std::cout << "GMM succesfully converged (with " << n_images << " images)" << std::endl;
    } catch (const std::logic_error& e){
        std::cout << "frakking GMM failed: " << e.what() << std::endl;
    } catch (const std::runtime_error& e){
        std::cout << "frakking GMM failed to converge: " << e.what() << std::endl;
    }

    return gmm;
}

template <typename Dataset, typename Ref>
hmm_p train_ref_hmm(const Dataset& dataset, Ref& ref_a, names training_images) {
    dll::auto_timer timer("hmm_train");

    auto characters = dataset.word_labels.at(training_images[0]).size();

    const auto n_states = characters * n_states_per_char;
    const auto n_features = ref_a[0][0].size();

    auto hmm = std::make_unique<HMM<GMM>>(n_states, GMM(n_hmm_gaussians, n_features));

    //Collect the standard deviations

    std::vector<double> deviations(n_features);

    for(std::size_t f = 0; f < n_features; ++f){
        double mean = 0.0;
        std::size_t n = 0;

        for(auto& image : ref_a){
            for(auto& column : image){
                mean += column[f];
            }
            n += image.size();
        }

        mean /= n;

        double stddev = 0.0;

        for(auto& image : ref_a){
            for(auto& column : image){
                stddev += (column[f] - mean) * (column[f] - mean);
            }
        }

        deviations[f] = std::sqrt(stddev / n);
    }

    static std::default_random_engine rand_engine(std::time(nullptr));

    //Generate the Salt distributions

    std::vector<std::normal_distribution<double>> distributions;

    for(std::size_t f = 0; f < n_features; ++f){
        distributions.emplace_back(0.0, salt * deviations[f]);
    }

    // Copy the input (for salting)
    auto reference = ref_a;
    reference.clear();

    for(auto& image : ref_a){
        for (std::size_t s = 0; s < salts; ++s) {
            auto copy = image;

            if(salt > 0.0){
                for (auto& column : copy) {
                    for (std::size_t f = 0; f < n_features; ++f) {
                        column[f] += distributions[f](rand_engine);
                    }
                }
            }

            reference.push_back(copy);
        }
    }

    std::vector<arma::mat> images(reference.size());
    std::vector<arma::Row<size_t>> labels(reference.size());

    for(std::size_t image = 0; image < reference.size(); ++image){
        auto& ref_image = reference[image];

        auto width = ref_image.size();

        images[image] = arma::mat(n_features, width);
        labels[image] = arma::Row<size_t>(width);

        std::size_t current_label = 0;
        std::size_t label_distance = width / n_states;

        for(std::size_t i = 0; i < width; ++i){
            for(std::size_t j = 0; j < ref_image[i].size(); ++j){
                images[image](j, i) = ref_image[i][j];
            }

            labels[image](i) = std::min(current_label, n_states - 1);

            if(i > 0 && i % label_distance == 0){
                ++current_label;
            }
        }
    }

    try {
        hmm->Train(images, labels);

        std::cout << "HMM succesfully converged (with " << images.size() << " images)" << std::endl;
    } catch (const std::logic_error& e){
        std::cout << "frakking HMM failed: " << e.what() << std::endl;
        std::cout << "\tn_images: " << images.size() << std::endl;
        std::cout << "\tn_features: " << n_features << std::endl;
        std::cout << "\tn_states: " << n_states << std::endl;

        for(std::size_t i = 0; i < images.size(); ++i){
            auto& image = images[i];
            auto& label = labels[i];

            image.print("Image");
            label.print("Label");
        }
    } catch (const std::runtime_error& e){
        std::cout << "frakking HMM failed to converge: " << e.what() << std::endl;
        std::cout << "\tn_images: " << images.size() << std::endl;
        std::cout << "\tn_features: " << n_features << std::endl;
        std::cout << "\tn_states: " << n_states << std::endl;
    }

    return hmm;
}

template <typename Dataset, typename V1>
double hmm_distance(const Dataset& dataset, const gmm_p& gmm, const hmm_p& hmm, std::size_t pixel_width, const V1& test_image, names training_images) {
    double ref_width = 0;

    for(auto& image : training_images){
        ref_width += dataset.word_images.at(image + ".png").size().width;
    }

    ref_width /= training_images.size();

    auto ratio = ref_width / pixel_width;

    if (ratio > 2.0 || ratio < 0.5) {
        return 1e8;
    }

    const auto n_features = test_image[0].size();
    const auto width = test_image.size();

    arma::mat image(n_features, width);

    for(std::size_t i = 0; i < width; ++i){
        for(std::size_t j = 0; j < test_image[i].size(); ++j){
            image(j, i) = test_image[i][j];
        }
    }

    auto gmm_likelihood = [&gmm](auto& image){
        auto likelihood = 0.0;

        for(std::size_t i = 0; i < image.n_cols; ++i){
            double p = gmm->Probability(image.col(i));

            if(!std::isfinite(p)){
                std::cerr << "WARNING: p(x|GMM) not finite: " << p << std::endl;
            }

            double logp;

            if(p == 0.0){
                std::cerr << "WARNING: p(x|GMM) == 0 col: " << i << std::endl;
                logp = -100;
            } else {
                logp = std::log(p);
            }

            if(!std::isfinite(logp)){
                std::cerr << "WARNING: log(p(x|GMM)) not finite: " << logp << std::endl;
            }

            likelihood += logp;
        }

        return likelihood;
    };

    auto p_hmm = hmm->LogLikelihood(image);
    auto p_gmm = gmm_likelihood(image);

    if(!std::isfinite(p_hmm)){
        std::cerr << "WARNING: p(X|HMM) not finite: " << p_hmm << std::endl;
        return 1e8; //TODO WHY
    }

    if(!std::isfinite(p_gmm)){
        std::cerr << "WARNING: p(X|GMM) not finite: " << p_gmm << std::endl;
    }

    return -(p_hmm / p_gmm);
}

} //end of namespace hmm_mlpack

#else

namespace hmm_mlpack {

using gmm_p = int;
using hmm_p = int;

template <typename RefFunctor>
gmm_p train_global_hmm(names /*train_word_names*/, RefFunctor /*functor*/) {
    //Disabled HMM
    std::cerr << "HMM has been disabled, -hmm should not be used" << std::endl;
}

template <typename Dataset, typename Ref>
hmm_p train_ref_hmm(const Dataset& /*dataset*/, Ref& /*ref_a*/, names /*training_images*/) {
    //Disabled HMM
    std::cerr << "HMM has been disabled, -hmm should not be used" << std::endl;
}

template <typename Dataset, typename V1>
double hmm_distance(const Dataset& /*dataset*/, const gmm_p& /*global_hmm*/, const hmm_p& /*hmm*/, std::size_t /*pixel_width*/, const V1& /*test_image*/, names /*training_images*/) {
    //Disabled HMM
    std::cerr << "HMM has been disabled, -hmm should not be used" << std::endl;
}

} //end of namespace hmm_mlpack

#endif

#endif
