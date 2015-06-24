//=======================================================================
// Copyright Baptiste Wicht 2015.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <string>

#include <sys/stat.h>

#include "cpp_utils/assert.hpp"

#include "config.hpp"
#include "reports.hpp"

void generate_rel_files(
            const std::string& result_folder, const spot_dataset& dataset,
            const std::vector<std::string>& test_image_names, const std::vector<std::vector<std::string>>& keywords){
    std::cout << "Generate relevance files..." << std::endl;

    std::ofstream global_relevance_stream(result_folder + "/global_rel_file");
    std::ofstream local_relevance_stream(result_folder + "/local_rel_file");

    for(std::size_t k = 0; k < keywords.size(); ++k){
        auto& keyword = keywords[k];

        for(std::size_t t = 0; t < test_image_names.size(); ++t){
            decltype(auto) test_image = test_image_names[t];

            std::string keyword_str;
            keyword_str = std::accumulate(keyword.begin(), keyword.end(), keyword_str);

            global_relevance_stream << "cv1 0 " << keyword_str << "_" << test_image;
            local_relevance_stream << "cv1_" << keyword_str << " 0 " << test_image;

            if(dataset.word_labels.at({test_image.begin(), test_image.end() - 4}) == keyword){
                global_relevance_stream << " 1" << std::endl;
                local_relevance_stream << " 1" << std::endl;
            } else {
                global_relevance_stream << " 0" << std::endl;
                local_relevance_stream << " 0" << std::endl;
            }
        }
    }

    std::cout << "... done" << std::endl;
}

void update_stats(std::size_t k, const std::string& result_folder, const spot_dataset& dataset, const std::vector<std::string>& keyword, std::vector<std::pair<std::string, weight>> diffs_a, std::vector<double>& eer, std::vector<double>& ap, std::ofstream& global_top_stream, std::ofstream& local_top_stream, const std::vector<std::string>& test_image_names){
    std::sort(diffs_a.begin(), diffs_a.end(), [](auto& a, auto& b){ return a.second < b.second; });

    auto total_positive = std::count_if(test_image_names.begin(), test_image_names.end(),
        [&dataset, &keyword](auto& i){ return dataset.word_labels.at({i.begin(), i.end() - 4}) == keyword; });

    cpp_assert(total_positive > 0, "No example for one keyword");

    std::vector<std::size_t> tp(diffs_a.size());
    std::vector<std::size_t> fp(diffs_a.size());
    std::vector<std::size_t> fn(diffs_a.size());
    std::vector<double> tpr(diffs_a.size());
    std::vector<double> fpr(diffs_a.size());
    std::vector<double> recall(diffs_a.size());
    std::vector<double> precision(diffs_a.size());

    std::size_t ap_updates = 0;

    for(std::size_t n = 0; n < diffs_a.size(); ++n){
        std::string keyword_str;
        keyword_str = std::accumulate(keyword.begin(), keyword.end(), keyword_str);

        global_top_stream << "cv1 Q0 " << keyword_str << "_" << diffs_a[n].first << ".png 0 " << -diffs_a[n].second << " bw" << std::endl;
        local_top_stream << "cv1_" << keyword_str << " Q0 " << diffs_a[n].first << ".png 0 " << -diffs_a[n].second << " bw" << std::endl;

        std::size_t tp_n = n == 0 ? 0 : tp[n - 1];
        std::size_t fp_n = n == 0 ? 0 : fp[n - 1];
        std::size_t fn_n = n == 0 ? total_positive : fn[n - 1];

        if(dataset.word_labels.at(diffs_a[n].first) == keyword){
            ++tp_n;
            --fn_n;
        } else {
            ++fp_n;
        }

        tp[n] = tp_n;
        fp[n] = fp_n;
        fn[n] = fn_n;

        tpr[n] = static_cast<double>(tp_n) / (tp_n + fn_n);
        fpr[n] = static_cast<double>(fp_n) / (n + 1);

        recall[n] = tpr[n];
        precision[n] = static_cast<double>(tp_n) / (tp_n + fp_n);

        if(std::fabs(fpr[n] - (1.0 - tpr[n])) < 1e-7){
            eer[k] = fpr[n];
        }

        if(n == 0){
            ++ap_updates;
            ap[k] += precision[n];
        } else if(recall[n] != recall[n - 1]){
            ++ap_updates;
            ap[k] += precision[n];
        }
    }

    ap[k] /= ap_updates;

    if(generate_graphs){
        std::ofstream roc_gp_stream(result_folder + "/" + std::to_string(k) + "_roc.gp");

        roc_gp_stream << "set terminal png size 300,300 enhanced" << std::endl;
        roc_gp_stream << "set output '" << k << "_roc.png'" << std::endl;
        roc_gp_stream << "set title \"ROC(" << k << ")\"" << std::endl;
        roc_gp_stream << "set xlabel \"FPR\"" << std::endl;
        roc_gp_stream << "set ylabel \"TPR\"" << std::endl;
        roc_gp_stream << "plot [0:1] '" << k << "_roc.dat' with lines title ''" << std::endl;

        std::ofstream roc_data_stream(result_folder + "/" + std::to_string(k) + "_roc.dat");

        for(std::size_t nn = 0; nn < tpr.size(); ++nn){
            roc_data_stream << fpr[nn] << " " << tpr[nn] << std::endl;
        }

        std::ofstream pr_gp_stream(result_folder + "/" + std::to_string(k) + "_pr.gp");

        pr_gp_stream << "set terminal png size 300,300 enhanced" << std::endl;
        pr_gp_stream << "set output '" << k << "_pr.png'" << std::endl;
        pr_gp_stream << "set title \"PR(" << k << ")\"" << std::endl;
        pr_gp_stream << "set xlabel \"Recall\"" << std::endl;
        pr_gp_stream << "set ylabel \"Precision\"" << std::endl;
        pr_gp_stream << "plot [0:1] '" << k << "_pr.dat' with lines title ''" << std::endl;

        std::ofstream pr_data_stream(result_folder + "/" + std::to_string(k) + "_pr.dat");

        for(std::size_t nn = 0; nn < tpr.size(); ++nn){
            pr_data_stream << precision[nn] << " " << recall[nn] << std::endl;
        }
    }
}

void update_stats_light(std::size_t k, const spot_dataset& dataset, const std::vector<std::string>& keyword, std::vector<std::pair<std::string, weight>> diffs_a, std::vector<double>& ap, const std::vector<std::string>& test_image_names){
    std::sort(diffs_a.begin(), diffs_a.end(), [](auto& a, auto& b){ return a.second < b.second; });

    auto total_positive = std::count_if(test_image_names.begin(), test_image_names.end(),
        [&dataset, &keyword](auto& i){ return dataset.word_labels.at({i.begin(), i.end() - 4}) == keyword; });

    cpp_assert(total_positive > 0, "No example for one keyword");

    std::vector<double> recall(diffs_a.size());

    std::size_t tp_n = 0;
    std::size_t fp_n = 0;
    std::size_t fn_n = total_positive;

    std::size_t ap_updates = 0;

    for(std::size_t n = 0; n < diffs_a.size(); ++n){
        if(dataset.word_labels.at(diffs_a[n].first) == keyword){
            ++tp_n;
            --fn_n;
        } else {
            ++fp_n;
        }

        recall[n] = static_cast<double>(tp_n) / (tp_n + fn_n);

        if(n == 0){
            ++ap_updates;
            ap[k] += static_cast<double>(tp_n) / (tp_n + fp_n);
        } else if(recall[n] != recall[n - 1]){
            ++ap_updates;
            ap[k] += static_cast<double>(tp_n) / (tp_n + fp_n);
        }
    }

    ap[k] /= ap_updates;
}

std::string select_folder(const std::string& base_folder){
    std::cout << "Select a folder ..." << std::endl;

    mkdir(base_folder.c_str(), 0777);

    std::size_t result_name = 0;

    std::string result_folder;

    struct stat buffer;
    do {
        ++result_name;
        result_folder = base_folder + std::to_string(result_name);
    } while(stat(result_folder.c_str(), &buffer) == 0);

    mkdir(result_folder.c_str(), 0777);

    std::cout << "... " << result_folder << std::endl;

    return result_folder;
}
