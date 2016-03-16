//=======================================================================
// Copyright Baptiste Wicht 2015-2016.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-function"

#pragma once

#ifndef SPOTTER_NO_HMM

#define FULL_DEBUG

#include <random>

#include <sys/stat.h> //mkdir

#include "dll/util/io.hpp" //TODO This should be extracted to cpp_utils

namespace hmm_htk {

using gmm_p = std::string;
using hmm_p = std::string;

//Number of gaussians for the HMM
constexpr const std::size_t n_hmm_gaussians = 2;

//Number of training iterations for the HMM
constexpr const std::size_t n_hmm_iterations = 2;

//Number of states per character
constexpr const auto n_states_per_char = 10;

const std::string bin_hmm_init = "scripts/hmm-init.pl";
const std::string bin_hhed   = "HHEd";
const std::string bin_herest = "HERest";
const std::string bin_hvite = "HVite";

inline auto exec_command(const std::string& command) {
    std::stringstream output;

    char buffer[1024];

    FILE* stream = popen(command.c_str(), "r");

    while (fgets(buffer, 1024, stream) != NULL) {
        output << buffer;
    }

    auto status = pclose(stream);
    auto exit_code = WEXITSTATUS(status);

    return std::make_pair(exit_code, output.str());
}

template <typename RefFunctor>
gmm_p train_global_hmm(names train_word_names, RefFunctor functor) {
    dll::auto_timer timer("htk_gmm_train");

    auto ref_a = functor(train_word_names);

    const auto n_features = ref_a[0][0].size();

    //TODO Better Configure how the subset if selected
    std::size_t step = 5;

    //Collect information on the dataset

    std::size_t n_observations = 0;
    std::size_t n_images = 0;

    for(std::size_t image = 0; image < ref_a.size(); image += step){
        n_observations += ref_a[image].size();
        ++n_images;
    }

    //TODO

    return "frakking_gmm";
}

template <typename Dataset, typename Ref>
hmm_p train_ref_hmm(const Dataset& dataset, Ref& ref_a, names training_images) {
    dll::auto_timer timer("htk_hmm_train");

    auto label = dataset.word_labels.at(training_images[0]);
    auto characters = label.size();

    const auto n_states = characters * n_states_per_char;
    const auto n_features = ref_a[0][0].size();

    const std::string base_folder = ".hmm";
    const std::string folder = base_folder + "/" + keyword_to_short_string(label);

    const std::string features_file    = folder + "/train_features.lst";
    const std::string hmm_info_file    = folder + "/hmm_info";
    const std::string means_file       = folder + "/means";
    const std::string variances_file   = folder + "/variances";
    const std::string covariances_file = folder + "/covariances";
    const std::string init_mmf_file    = folder + "/init_mmf";
    const std::string htk_config_file  = folder + "/htk_config";
    const std::string letters_file     = folder + "/letters";
    const std::string mlf_file         = folder + "/train.mlf";

    mkdir(base_folder.c_str(), 0777);
    mkdir(folder.c_str(), 0777);

    std::vector<std::string> features_files;

    // Generate the features files (at least, it makes sense, but why binary...)

    for(std::size_t i = 0; i < training_images.size(); ++i){
        auto& image = training_images[i];
        auto& ref   = ref_a[i];

        const std::string file_path = folder + "/" + image + ".htk";

        features_files.push_back(file_path);

        std::ofstream os(file_path, std::ofstream::binary);

        dll::binary_write(os, static_cast<int>(ref_a.size()));                 //Number of samples
        dll::binary_write(os, static_cast<int>(1));                            //Dummy HTK_SAMPLE_RATE
        dll::binary_write(os, static_cast<short>(n_features * sizeof(float))); //Sample size
        dll::binary_write(os, static_cast<short>(9));                          //Used defined sample kind = 9 ?

        //Write all the values
        for(auto feature_vector : ref){
            for(auto v : feature_vector){
                dll::binary_write(os, v);
            }
        }
    }

    // Generate a file with the list of feature files (silly...)

    {
        std::ofstream os(features_file);

        for(auto& file : features_files){
            os << file << "\n";
        }
    }

    // Generate the HTK config file (no idea what is in it)

    {
        std::ofstream os(htk_config_file);

        os << "NATURALREADORDER    =   T\n";
        os << "NATURALWRITEORDER   =   T";
    }

    // Generate a file with the states

    {
        std::ofstream os(hmm_info_file);

        for(std::size_t i = 1; i <= characters; ++i){
            os << "s" << i << " " << n_states_per_char << " nocov noinit\n";
        }
    }

    // Generate a file with the letters

    {
        std::ofstream os(letters_file);

        for(std::size_t i = 1; i <= characters; ++i){
            os << "s" << i << "\n";
        }
    }

    // Generate a file with the labels (I think)

    {
        std::ofstream os(mlf_file);

        os << "#!MLF!#" << std::endl;

        for(std::size_t i = 0; i < training_images.size(); ++i){
            auto& image = training_images[i];
            auto& ref   = ref_a[i];

            const std::string lab_name = folder + "/" + image + ".lab";

            os << "\"" << lab_name << "\"" << '\n';

            for(std::size_t i = 1; i <= characters; ++i){
                os << "s" << i << '\n';
            }

            os << "." << '\n';
        }
    }

    // Generate the means/variances/covariances/init_mmf files

    {
        std::string init_command =
            bin_hmm_init
            + " --hmmmacro " + init_mmf_file
            + " -e " + means_file
            + " -v " + variances_file
            + " -c " + covariances_file
            + " --infofile " + hmm_info_file
            + " " + features_file;

        auto result = exec_command(init_command);

        if(result.first){
            std::cout << "init scripts failed with result code: " << result.first << std::endl;
            std::cout << "Command: " << init_command << std::endl;
            std::cout << result.second << std::endl;
        }
    }

    for(std::size_t g = 1; g <= n_hmm_gaussians; ++g){
        const std::string mmf_file           = folder + "/trained_" + std::to_string(g) + ".mmf";
        const std::string stats_file         = folder + "/stats_" + std::to_string(g) + ".txt";
        const std::string multigaussian_file = folder + "/mu_" + std::to_string(g) + ".hhed";

        std::string previous_mmf_file;
        if(g == 1){
            previous_mmf_file = init_mmf_file;
        } else {
            previous_mmf_file = folder + "/trained_" + std::to_string(g - 1) + ".mmf";
        }

        std::string cp_command = "cp " + previous_mmf_file + " " + mmf_file;
        auto cp_result = exec_command(cp_command);

        if(cp_result.first){
            std::cout << "cp failed with result code: " << cp_result.first << std::endl;
            std::cout << "Command: " << cp_command << std::endl;
            std::cout << cp_result.second << std::endl;

            //If HHEd fails, HERest will fail anyway
            break;
        }

        {
            std::ofstream os(multigaussian_file);
            os << "MU " << g << " {*.state[2-10000].mix}" << std::endl;
        }

        std::string hhed_command =
            bin_hhed +
            " -C " + htk_config_file +
            " -M " + folder +
            " -H " + mmf_file +
            " " + multigaussian_file + " " + letters_file;

        auto hhed_result = exec_command(hhed_command);

        if(hhed_result.first){
            std::cout << "HHEd failed with result code: " << hhed_result.first << std::endl;
            std::cout << "Command: " << hhed_command << std::endl;
            std::cout << hhed_result.second << std::endl;

            //If HHEd fails, HERest will fail anyway
            break;
        }

        for(std::size_t i = 0; i < n_hmm_iterations; ++i){
            const double herest_min_variance = 0.000001;

            std::string herest_command =
                bin_herest +
                " -C " + htk_config_file +
                " -v " + std::to_string(herest_min_variance) +
                " -M " + folder +
                " -I " + mlf_file +
                " -H " + mmf_file +
                " -s " + stats_file +
                " -S " + features_file +
                " " + letters_file;

            auto herest_result = exec_command(herest_command);

            if(herest_result.first){
                std::cout << "HERest failed with result code: " << herest_result.first << std::endl;
                std::cout << "Command: " << herest_command << std::endl;
                std::cout << herest_result.second << std::endl;
            }

        }
    }

    return folder;
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

    const std::string folder = hmm;

    const std::string features_file    = folder + "/test_features.lst";
    const std::string hmm_info_file    = folder + "/hmm_info";
    const std::string means_file       = folder + "/means";
    const std::string variances_file   = folder + "/variances";
    const std::string covariances_file = folder + "/covariances";
    const std::string init_mmf_file    = folder + "/init_mmf";
    const std::string htk_config_file  = folder + "/htk_config";
    const std::string letters_file     = folder + "/letters";
    const std::string mlf_file         = folder + "/train.mlf";
    const std::string log_file    = folder + "/vite.log";

    const std::string file_path = folder + "/test_file.htk";

    {
        std::ofstream os(features_file);
        os << file_path << "\n";
    }

    {
        std::ofstream os(file_path, std::ofstream::binary);

        dll::binary_write(os, static_cast<int>(1));                            //Number of samples
        dll::binary_write(os, static_cast<int>(1));                            //Dummy HTK_SAMPLE_RATE
        dll::binary_write(os, static_cast<short>(n_features * sizeof(float))); //Sample size
        dll::binary_write(os, static_cast<short>(9));                          //Used defined sample kind = 9 ?

        //Write all the values
        for(auto feature_vector : test_image){
            for(auto v : feature_vector){
                dll::binary_write(os, v);
            }
        }
    }

    std::string hvite_command =
        bin_hvite +
        " -C " + htk_config_file +
        " -i " + log_file +
        " -H " + hmm_info_file +
        " -S " + features_file +
        " " + letters_file;

    auto hvite_result = exec_command(hvite_command);

    if(hvite_result.first){
        std::cout << "HVite failed with result code: " << hvite_result.first << std::endl;
        std::cout << "Command: " << hvite_command << std::endl;
        std::cout << hvite_result.second << std::endl;
    }

  //`bin/HVite -C #{file_htk_config} -i #{file_recognition_log} -H #{file_mmf} -S #{file_featurelist} #{file_spelling} #{file_letters}`




    //TODO

    return 1e8;
}

} //end of namespace hmm_mlpack

#else

namespace hmm_htk {

using gmm_p = std::string;
using hmm_p = std::string;

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
#pragma GCC diagnostic pop
