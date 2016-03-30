//=======================================================================
// Copyright Baptiste Wicht 2015-2016.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#define FULL_DEBUG

#include <random>
#include <set>

#include <sys/stat.h> //mkdir

#include "cpp_utils/io.hpp"

//#define SPACE_MODEL
//#define HMM_VERBOSE
//#define WRITE_LOG

namespace hmm_htk {

using hmm_p = std::string;

// Number of gaussians for the HMM
constexpr const std::size_t n_hmm_gaussians = 7;

// Number of training iterations for the HMM
constexpr const std::size_t n_hmm_iterations = 4;

// Number of states per character
constexpr const auto n_states_per_char_std     = 20;
constexpr const auto n_states_per_char_patches = 10;

// Number of states per space character
constexpr const auto n_states_per_space = 10;

// Minimum variance for training
constexpr const double herest_min_variance = 0.000001;

const std::string bin_hmm_init   = "scripts/hmm-init.pl";
const std::string bin_hhed       = "HHEd";
const std::string bin_herest     = "HERest";
const std::string bin_hvite      = "HVite";
const std::string bin_hparse     = "HParse";
const std::string bin_debug_args = " -A -D -V -T 1 ";

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

inline void write_log(const std::string& result, const std::string& file){
#ifdef WRITE_LOG
    std::ofstream os(file);
    os << result;
#else
    cpp_unused(result);
    cpp_unused(file);
#endif
}

template <typename Dataset>
hmm_p train_global_hmm(const config& conf, const Dataset& dataset, names train_word_names) {
    dll::auto_timer timer("htk_global_hmm_train");

    // The folders
    const std::string base_folder = ".hmm";
    const std::string folder = base_folder + "/global/";
    const std::string logs_folder = base_folder + "/global/logs/";

    mkdir(base_folder.c_str(), 0777);
    mkdir(folder.c_str(), 0777);
    mkdir(logs_folder.c_str(), 0777);

    // Generated files
    const std::string htk_config_file     = folder + "/htk_config";
    const std::string hmm_info_file       = folder + "/hmm_info";
    const std::string letters_file        = folder + "/letters";
    const std::string features_file       = folder + "/train_features.lst";
    const std::string means_file          = folder + "/means";
    const std::string variances_file      = folder + "/variances";
    const std::string covariances_file    = folder + "/covariances";
    const std::string init_mmf_file       = folder + "/init_mmf";
    const std::string mlf_file            = folder + "/train.mlf";
    const std::string spelling_file       = folder + "/spelling";
    const std::string global_grammar_file = folder + "/grammar.bnf";
    const std::string global_wordnet_file = folder + "/grammar.wnet";

    // Get number of states per char from the configuration
    const std::size_t n_states_per_char =
            conf.method == Method::Patches
        ?   n_states_per_char_patches
        :   n_states_per_char_std;

    // Collect the characters

    std::set<std::string> characters;

    for(const auto& training_image : train_word_names){
        decltype(auto) label = dataset.word_labels.at(training_image);

        for(const auto& character : label){
            characters.insert(character);
        }
    }

    // Generate the HTK config file (no idea what is in it)

    {
        std::ofstream os(htk_config_file);

        os << "NATURALREADORDER    =   T\n"
           << "NATURALWRITEORDER   =   T";
    }

    // Generate a file with the states

    {
        std::ofstream os(hmm_info_file);

        for(const auto& character : characters){
            os << character << " " << n_states_per_char << " nocov noinit\n";
        }

#ifdef SPACE_MODEL
        os << "sp " << n_states_per_space << " nocov noinit\n";
#endif
    }

    // Generate a file with the letters

    {
        std::ofstream os(letters_file);

        for(const auto& character : characters){
            os << character << "\n";
        }

#ifdef SPACE_MODEL
        os << "sp\n";
#endif
    }

    // Generate a file with the list of feature files (silly...)

    {
        std::ofstream os(features_file);

        for(auto& image : train_word_names){
            os << base_folder + "/train/" + image << ".htk\n";
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

    // Generate a file with the labels

    {
        std::ofstream os(mlf_file);

        os << "#!MLF!#" << std::endl;

        for(const auto& image : train_word_names){
            const std::string lab_name = base_folder + "/train/" + image + ".lab";

            os << "\"" << lab_name << "\"\n";

#ifdef SPACE_MODEL
            os << "sp\n";
#endif

            decltype(auto) label = dataset.word_labels.at(image);

            for(const auto& character : label){
                os << character << '\n';
            }

#ifdef SPACE_MODEL
            os << "sp\n";
#endif

            os << ".\n";
        }
    }

    // Generate the spelling file (used for testing)

    {
        std::ofstream os(spelling_file);

        for(auto& character : characters){
            os << character << " " << character << '\n';
        }

#ifdef SPACE_MODEL
        os << "sp sp\n";
#endif
    }

    // Generate the global grammar (used for testing)

    {
        std::ofstream os(global_grammar_file);

#ifdef SPACE_MODEL
        os << "( sp < ";
#else
        os << "( < ";
#endif

        std::string sep = " ";
        for(auto& character : characters){
            os << sep << character;
            sep = " | ";
        }

#ifdef SPACE_MODEL
        os << " > sp )";
#else
        os << " > )";
#endif
    }

    // Generate the global wordnet (used for testing)

    {
        std::string hparse_command =
            bin_hparse +
            bin_debug_args +
            " " + global_grammar_file +
            " " + global_wordnet_file;

        auto hparse_result = exec_command(hparse_command);

        if(hparse_result.first){
            std::cout << "HParse failed with result code: " << hparse_result.first << std::endl;
            std::cout << "Command: " << hparse_command << std::endl;
            std::cout << hparse_result.second << std::endl;
        }
    }

    // Train each gaussian

    std::cout << "Start training HMM with " << train_word_names.size() << " word images" << std::endl;

    for(std::size_t g = 1; g <= n_hmm_gaussians; ++g){
        const std::string mmf_file           = folder + "/trained_" + std::to_string(g) + ".mmf";
        const std::string stats_file         = folder + "/stats_" + std::to_string(g) + ".txt";
        const std::string multigaussian_file = folder + "/mu_" + std::to_string(g) + ".hhed";
        const std::string hhed_log_file      = folder + "/hhed_" + std::to_string(g) + ".log";

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
            bin_debug_args +
            " -C " + htk_config_file +
            " -M " + folder +
            " -H " + mmf_file +
            " " + multigaussian_file + " " + letters_file;

        auto hhed_result = exec_command(hhed_command);

        write_log(hhed_result.second, hhed_log_file);

        if(hhed_result.first){
            std::cout << "HHEd failed with result code: " << hhed_result.first << std::endl;
            std::cout << "Command: " << hhed_command << std::endl;
            std::cout << hhed_result.second << std::endl;

            //If HHEd fails, HERest will fail anyway
            break;
        }

        for(std::size_t i = 0; i < n_hmm_iterations; ++i){
            const std::string herest_log_file      = folder + "/herest_" + std::to_string(g) + "_" + std::to_string(i) + ".log";

            std::string herest_command =
                bin_herest +
                bin_debug_args +
                " -C " + htk_config_file +
                " -v " + std::to_string(herest_min_variance) +
                " -M " + folder +
                " -I " + mlf_file +
                " -H " + mmf_file +
                " -s " + stats_file +
                " -S " + features_file +
                " " + letters_file;

            auto herest_result = exec_command(herest_command);

            write_log(herest_result.second, herest_log_file);

            if(herest_result.first){
                std::cout << "HERest failed with result code: " << herest_result.first << std::endl;
                std::cout << "Command: " << herest_command << std::endl;
                std::cout << herest_result.second << std::endl;
            }
        }

        std::cout << '.';
    }

    std::cout << " done" << std::endl;

    return base_folder;
}

template <typename Dataset>
hmm_p prepare_test_keywords(const Dataset& dataset, names training_images) {
    dll::auto_timer timer("htk_prepare_test_keywords");

    const decltype(auto) label = dataset.word_labels.at(training_images[0]);

    // Folders
    const std::string base_folder = ".hmm";
    const std::string folder      = base_folder + "/" + keyword_to_short_string(label);

    mkdir(folder.c_str(), 0777);

    // Generated files
    const std::string keyword_grammar_file = folder + "/grammar.bnf";
    const std::string keyword_wordnet_file = folder + "/grammar.wnet";

    // Generate the global grammar (used for testing)

    {
        std::ofstream os(keyword_grammar_file);

#ifdef SPACE_MODEL
        os << "( sp ";
#else
        os << "( ";
#endif

        for(auto& character : label){
            os << " " << character;
        }

#ifdef SPACE_MODEL
        os << " sp )";
#else
        os << " )";
#endif
    }

    // Generate the global wordnet (used for testing)

    {
        std::string hparse_command =
            bin_hparse +
            bin_debug_args +
            " " + keyword_grammar_file +
            " " + keyword_wordnet_file;

        auto hparse_result = exec_command(hparse_command);

        if(hparse_result.first){
            std::cout << "HParse failed with result code: " << hparse_result.first << std::endl;
            std::cout << "Command: " << hparse_command << std::endl;
            std::cout << hparse_result.second << std::endl;
        }
    }

    return folder;
}

template <typename V1>
void prepare_features(const std::string& folder_name, names test_image_names, const V1& test_features_a, bool lst_file) {
    const std::string base_folder = ".hmm";
    const std::string folder = base_folder + "/" + folder_name;

    mkdir(base_folder.c_str(), 0777);
    mkdir(folder.c_str(), 0777);

    for(std::size_t t = 0; t < test_image_names.size(); ++t){
        auto& test_image = test_image_names[t];
        auto& test_features = test_features_a[t];

        const auto n_features = test_features[0].size();

        // Local files
        const std::string features_file = folder + "/" + test_image + ".lst";
        const std::string file_path     = folder + "/" + test_image + ".htk";

        // Generate the file with the list of feature files

        if(lst_file) {
            std::ofstream os(features_file);
            os << file_path << "\n";
        }

        // Generate the feature file

        {
            std::ofstream os(file_path, std::ofstream::binary);

            cpp::binary_write(os, static_cast<int32_t>(test_features.size()));       //Number of observations
            cpp::binary_write(os, static_cast<int32_t>(1));                          //Dummy HTK_SAMPLE_RATE
            cpp::binary_write(os, static_cast<int16_t>(n_features * sizeof(float))); //Observation size
            cpp::binary_write(os, static_cast<int16_t>(9));                          //Used defined sample kind = 9

            //Write all the values
            for (decltype(auto) feature_vector : test_features) {
                for (auto feature : feature_vector) {
                    cpp::binary_write(os, static_cast<float>(feature));
                }
            }
        }
    }
}

template <typename V1>
void prepare_test_features(names test_image_names, const V1& test_features_a) {
    dll::auto_timer timer("htk_train_features");
    prepare_features("test", test_image_names, test_features_a, false);
}

template <typename V1>
void prepare_train_features(names test_image_names, const V1& test_features_a) {
    dll::auto_timer timer("htk_test_features");
    prepare_features("train", test_image_names, test_features_a, false);
}

inline void global_likelihood_many(const hmm_p& base_folder, names test_image_names, std::vector<double>& global_likelihoods, std::size_t t, std::size_t start, std::size_t end) {
    // Global files
    const std::string hmm_info_file       = base_folder + "/global/trained_" + std::to_string(n_hmm_gaussians) + ".mmf";
    const std::string htk_config_file     = base_folder + "/global/htk_config";
    const std::string letters_file        = base_folder + "/global/letters";
    const std::string global_wordnet_file = base_folder + "/global/grammar.wnet";
    const std::string spelling_file       = base_folder + "/global/spelling";

    // Generated files
    const std::string global_trans_file = base_folder + "/global/logs/thread_" + std::to_string(t) + ".trans";
    const std::string global_log_file   = base_folder + "/global/logs/thread_" + std::to_string(t) + ".log";
    const std::string global_lst_file   = base_folder + "/test/thread_" + std::to_string(t) + ".lst";

    // Generate the listing file

    {
        std::ofstream os(global_lst_file);

        for (std::size_t i = start; i < end; ++i) {
            os << base_folder << "/test/" << test_image_names[i] << ".htk\n";
        }
    }

    std::string hvite_command =
        bin_hvite +
        bin_debug_args +
        " -C " + htk_config_file +
        " -w " + global_wordnet_file +
#ifdef WRITE_LOG
        " -i " + global_trans_file +
#endif
        " -H " + hmm_info_file +
        " -S " + global_lst_file +
        " " + spelling_file +
        " " + letters_file;

    auto hvite_result = exec_command(hvite_command);

    if (hvite_result.first) {
        std::cout << "HVite failed with result code: " << hvite_result.first << std::endl;
        std::cout << "Command: " << hvite_command << std::endl;
        std::cout << hvite_result.second << std::endl;
    } else {
        write_log(hvite_result.second, global_log_file);

        decltype(auto) result = hvite_result.second;

        std::istringstream f(result);
        std::string line;
        std::size_t i = start;
        while (std::getline(f, line)) {
            if(line.find("File: ") == 0){
                if(!line.find(test_image_names[i])){
                    std::cout << "I hate HTK" << std::endl;
                }

                // Go to the next line
                std::getline(f, line);

                if (line.find(" == ") != std::string::npos) {
                    auto begin = line.find("[Ac=");
                    auto end = line.find(" ", begin + 1);
                    std::string log_likelihood_str(line.begin() + begin + 4, line.begin() + end);
                    global_likelihoods[i] = -std::atof(log_likelihood_str.c_str());
                } else {
#ifdef HMM_VERBOSE
                    std::cout << "global accuracy was not found for image " << test_image_names[i] << std::endl;
#endif
                    global_likelihoods[i] = 1e8;
                }

                ++i;
            }
        }
    }
}

inline void keyword_likelihood_many(const hmm_p& base_folder, const hmm_p& folder, names test_image_names, std::vector<double>& keyword_likelihoods, std::size_t t, std::size_t start) {
    // Global files
    const std::string hmm_info_file   = base_folder + "/global/trained_" + std::to_string(n_hmm_gaussians) + ".mmf";
    const std::string htk_config_file = base_folder + "/global/htk_config";
    const std::string letters_file    = base_folder + "/global/letters";
    const std::string spelling_file   = base_folder + "/global/spelling";
    const std::string global_lst_file = base_folder + "/test/thread_" + std::to_string(t) + ".lst";

    // Keywords files
    const std::string keyword_wordnet_file = folder + "/grammar.wnet";

    // Generated files
    const std::string keyword_trans_file = folder + "/thread_" + std::to_string(t) + ".trans";
    const std::string keyword_log_file   = folder + "/thread_" + std::to_string(t) + ".log";

    std::string hvite_command =
        bin_hvite +
        bin_debug_args +
        " -C " + htk_config_file +
        " -w " + keyword_wordnet_file +
#ifdef WRITE_LOG
        " -i " + keyword_trans_file +
#endif
        " -H " + hmm_info_file +
        " -S " + global_lst_file +
        " " + spelling_file +
        " " + letters_file;

    auto hvite_result = exec_command(hvite_command);

    if (hvite_result.first) {
        std::cout << "HVite failed with result code: " << hvite_result.first << std::endl;
        std::cout << "Command: " << hvite_command << std::endl;
        std::cout << hvite_result.second << std::endl;
    } else {
        write_log(hvite_result.second, keyword_log_file);

        decltype(auto) result = hvite_result.second;

        std::istringstream f(result);
        std::string line;
        std::size_t i = start;
        while (std::getline(f, line)) {
            if(line.find("File: ") == 0){
                if(!line.find(test_image_names[i])){
                    std::cout << "I hate HTK" << std::endl;
                }

                // Go to the next line
                std::getline(f, line);

                if (line.find(" == ") != std::string::npos) {
                    auto begin = line.find("[Ac=");
                    auto end = line.find(" ", begin + 1);
                    std::string log_likelihood_str(line.begin() + begin + 4, line.begin() + end);
                    keyword_likelihoods[i] = -std::atof(log_likelihood_str.c_str());
                } else {
#ifdef HMM_VERBOSE
                    std::cout << "keyword accuracy was not found for image " << test_image_names[i] << " and keyword " << folder << std::endl;
#endif
                    keyword_likelihoods[i] = 1e8;
                }

                ++i;
            }
        }
    }
}

template <typename Dataset>
double hmm_distance(const Dataset& dataset, const std::string& test_image, names training_images, double global_acc, double keyword_acc) {
    auto pixel_width = dataset.word_images.at(test_image).size().width;

    double ref_width = 0;

    for(auto& image : training_images){
        ref_width += dataset.word_images.at(image + ".png").size().width;
    }

    ref_width /= training_images.size();

    const auto ratio = ref_width / pixel_width;

    if (ratio > 2.0 || ratio < 0.5) {
        return 1e8;
    }

    //return ((keyword_acc - global_acc) / double(pixel_width));
    return -(keyword_acc / global_acc);
}

} //end of namespace hmm_mlpack
