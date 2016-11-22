//=======================================================================
// Copyright Baptiste Wicht 2015-2016.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <string>

#include "etl/etl.hpp"

#include "cpp_utils/parallel.hpp"

#include "dll/util/timers.hpp"

#include "config.hpp"
#include "standard.hpp"
#include "utils.hpp"
#include "reports.hpp"
#include "dtw.hpp"        //Dynamic time warping
#include "features.hpp"   //Features exporting
#include "evaluation.hpp" //Global evaluation functions

#define LOCAL_MEAN_SCALING
#include "scaling.hpp" //Scaling functions

//#define DEBUG_DISTANCES

namespace {

const bool interpolate = false;

std::vector<etl::dyn_vector<weight>> standard_features_rath_2003(const cv::Mat& clean_image) {
    const auto width  = static_cast<std::size_t>(clean_image.size().width);

    std::vector<etl::dyn_vector<weight>> features;

    for (std::size_t i = 0; i < width; ++i) {
        features.emplace_back(45);
    }

    // Convert image to float
    cv::Mat clean_image_float(clean_image.size(), CV_64F);
    clean_image.convertTo(clean_image_float, clean_image_float.type());

    // 1. Build the kernels

    cv::Mat gaussian_kernel = cv::Mat::ones(13, 13, CV_64F);
    cv::Mat gaussian_kernel_dx = cv::Mat::ones(13, 13, CV_64F);
    cv::Mat gaussian_kernel_dy = cv::Mat::ones(13, 13, CV_64F);

    auto step = gaussian_kernel.step;
    auto* kernel = gaussian_kernel.data;
    auto* kernel_dx = gaussian_kernel_dx.data;
    auto* kernel_dy = gaussian_kernel_dy.data;

    double sigma = 4.0;
    double sigma2 = sigma * sigma;
    double sigma4 = sigma2 * sigma2;

    for (std::size_t i = 0; i < 13; ++i) {
        for (std::size_t j = 0; j < 13; ++j) {
            kernel[i * step + j]    = (1.0 / (2.0 * M_PI * sigma2)) * std::exp(-1.0 * ((i * i + j * j) / (2.0 * sigma2)));
            kernel_dx[i * step + j] = -1.0 * ((i * std::exp(-((i * i + j * j) / (2.0 * sigma2)))) / (2.0 * M_PI * sigma4));
            kernel_dy[i * step + j] = -1.0 * ((j * std::exp(-((i * i + j * j) / (2.0 * sigma2)))) / (2.0 * M_PI * sigma4));
        }
    }

    // 1. Gaussian Smoothing

    cv::Mat gaussian_blurred;
    cv::filter2D(clean_image_float, gaussian_blurred, -1, gaussian_kernel);
    cv::Mat gaussian_blurred_scaled(cv::Size(width, 15), gaussian_blurred.type());
    cv::resize(gaussian_blurred, gaussian_blurred_scaled, gaussian_blurred_scaled.size(), cv::INTER_AREA);

    // 2. Gaussian horizontal partial derivative

    cv::Mat gaussian_blurred_dx;
    cv::filter2D(clean_image_float, gaussian_blurred_dx, -1, gaussian_kernel_dx);
    cv::Mat gaussian_blurred_dx_scaled(cv::Size(width, 15), gaussian_blurred_dx.type());
    cv::resize(gaussian_blurred_dx, gaussian_blurred_dx_scaled, gaussian_blurred_dx_scaled.size(), cv::INTER_AREA);

    // 3. Gaussian vertical partial derivative

    cv::Mat gaussian_blurred_dy;
    cv::filter2D(clean_image_float, gaussian_blurred_dy, -1, gaussian_kernel_dy);
    cv::Mat gaussian_blurred_dy_scaled(cv::Size(width, 15), gaussian_blurred_dy.type());
    cv::resize(gaussian_blurred_dy, gaussian_blurred_dy_scaled, gaussian_blurred_dy_scaled.size(), cv::INTER_AREA);

    // 4. Merge the feature set

    for (std::size_t i = 0; i < width; ++i) {
        for (std::size_t j = 0; j < 15; ++j) {
            features[i][j] = gaussian_blurred_scaled.at<double>(j, i);
            features[i][15+j] = gaussian_blurred_dx_scaled.at<double>(j, i);
            features[i][30+j] = gaussian_blurred_dy_scaled.at<double>(j, i);
        }
    }

#ifdef LOCAL_LINEAR_SCALING
    local_linear_feature_scaling(features);
#endif

#ifdef LOCAL_MEAN_SCALING
    local_mean_feature_scaling(features);
#endif

    return features;
}

std::vector<etl::dyn_vector<weight>> standard_features_rodriguez_2008(const cv::Mat& clean_image) {
    const auto height = static_cast<std::size_t>(clean_image.size().height);
    const auto width  = static_cast<std::size_t>(clean_image.size().width);

    const std::size_t w     = height;    //Square window
    const std::size_t left  = w / 2;     // Left context
    const std::size_t right = w / 2 - 1; // Right context

    std::vector<etl::dyn_vector<weight>> features;

    // 0. Convert image to float
    cv::Mat clean_image_float(clean_image.size(), CV_64F);
    clean_image.convertTo(clean_image_float, clean_image_float.type());

    // 1. Compute the smoothed image
    cv::Mat L;
    cv::GaussianBlur(clean_image_float, L, cv::Size(0, 0), 3.0, 3.0);

    //2. Compute vertical and horizontal gradients
    cv::Mat sGx(clean_image.size(), CV_64F);
    cv::Mat sGy(clean_image.size(), CV_64F);

    const auto outside = 1.0;

    for (std::size_t y = 0; y < height; ++y) {
        auto* sGx_ptr = sGx.ptr<double>(y);
        auto* L_ptr   = L.ptr<double>(y);

        sGx_ptr[0] = L_ptr[1] - outside;

        for (std::size_t x = 1; x < width - 1; ++x) {
            sGx_ptr[x] = L_ptr[x + 1] - L_ptr[x - 1];
        }

        sGx_ptr[width - 1] = outside - L_ptr[width - 1 - 1];
    }

    for (std::size_t y = 1; y < height - 1; ++y) {
        auto* sGy_ptr = sGy.ptr<double>(y);
        auto* Ll_ptr  = L.ptr<double>(y + 1);
        auto* Lr_ptr  = L.ptr<double>(y - 1);

        for (std::size_t x = 0; x < width; ++x) {
            sGy_ptr[x] = Ll_ptr[x] - Lr_ptr[x];
        }
    }

    for (std::size_t x = 0; x < width; ++x) {
        sGy.at<double>(0, x) = L.at<double>(1, x) - outside;
        sGy.at<double>(height - 1, x) = outside - L.at<double>(height - 1 - 1, x);
    }

    // 3. Enlarge the gradients to avoid boundary effects

    cv::Mat Gx(cv::Size(width + height, height), CV_64F);
    cv::Mat Gy(cv::Size(width + height, height), CV_64F);

    Gx = cv::Scalar(0.0);
    Gy = cv::Scalar(0.0);

#ifndef OPENCV_23
    sGx.copyTo(Gx(cv::Rect(left , 0, width, height)));
    sGy.copyTo(Gy(cv::Rect(left , 0, width, height)));
#else
    //This will be frakking slow, but grid machines are RETARDEDLY ANCIENT

    for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
            Gx.at<double>(y, x + left) = sGx.at<double>(y, x);
            Gy.at<double>(y, x + left) = sGy.at<double>(y, x);
        }
    }
#endif

    // 4. Compute magnitude and orientations of the gradients

    cv::Mat m(Gx.size(), CV_64F);
    cv::Mat o(Gx.size(), CV_64F);

    for (std::size_t y = 0; y < static_cast<std::size_t>(Gx.size().height); ++y) {
        auto* Gx_ptr = Gx.ptr<double>(y);
        auto* Gy_ptr = Gy.ptr<double>(y);

        auto* m_ptr = m.ptr<double>(y);
        auto* o_ptr = o.ptr<double>(y);

        for (std::size_t x = 0; x < static_cast<std::size_t>(Gx.size().width); ++x) {
            auto gx = Gx_ptr[x];
            auto gy = Gy_ptr[x];

            m_ptr[x] = std::sqrt(gx * gx + gy * gy);
            o_ptr[x] = std::atan2(gx, gy);
        }
    }

    // 5. Sliding window

    cpp_assert(height % 2 == 0, "Rodriguez2008 has only been implemented for even windows");

    constexpr const std::size_t M = 4; //Number of cells
    constexpr const std::size_t N = 4; //Number of cells
    constexpr const std::size_t T = 8; //Number of bins

    for (std::size_t real_x = 0; real_x < width; ++real_x) {
        const std::size_t real_first = std::max(static_cast<int>(real_x) - static_cast<int>(left), 0);
        const std::size_t real_last  = std::min(real_x + right, width);

        // Compute the upper and lower contours inside the window

        std::size_t lower = 0;
        std::size_t upper = 0;

        features.emplace_back(M * N * T);

        for(std::size_t i = real_first; i < real_last; ++i){
            std::size_t local_lower = 0;
            for (std::size_t y = height - 1; y > 0; --y) {
                if (clean_image.at<uint8_t>(y, i) == 0.0) {
                    local_lower = y;
                    break;
                }
            }

            std::size_t local_upper = 0;
            for (std::size_t y = 0; y < height; ++y) {
                if (clean_image.at<uint8_t>(y, i) == 0.0) {
                    local_upper = y;
                    break;
                }
            }

            if(i == real_first){
                lower = local_lower;
                upper = local_upper;
            } else if(!(local_lower == 0 && local_upper == 0)) {
                lower = std::max(lower, local_lower);
                upper = std::min(upper, local_upper);
            }
        }

        upper = upper > 0 ? upper - 1 : upper;
        lower = lower < height - 1 ? lower + 1 : lower;

        // Compute dimensions of the cells

        const std::size_t grid_height = lower - upper;
        const std::size_t cell_width = w / M;
        const std::size_t cell_height = grid_height / N;

        // Iterate through the cells (4x4)

        for (std::size_t cy = 0; cy < N; ++cy) {
            for (std::size_t cx = 0; cx < M; ++cx) {
                const auto x_start = real_x + cx * cell_width;
                const auto y_start = upper + cy * cell_height;

                std::array<double, T> bins;
                bins.fill(0.0);

                // Attribute each magnitude to a bin

                for(std::size_t yy = y_start; yy < y_start + cell_height; ++yy){
                    auto* m_ptr = m.ptr<double>(yy);
                    auto* o_ptr = o.ptr<double>(yy);

                    for(std::size_t xx = x_start; xx < x_start + cell_width; ++xx){
                        auto magnitude = m_ptr[xx];
                        auto angle = o_ptr[xx] + M_PI; //In the range [0, 2pi]

                        std::size_t bin = std::size_t(angle / (2.0 * M_PI / T)) % T;
                        std::size_t next_bin = (bin + 1) % T;

                        auto inside_angle = angle - bin * (2.0 * M_PI / T);
                        auto bin_contrib = ((2.0 * M_PI / T) - inside_angle) / (2.0 * M_PI / T);
                        auto next_contrib = inside_angle / (2.0 * M_PI / T);

                        bins[bin] += bin_contrib * magnitude;
                        bins[next_bin] += next_contrib * magnitude;
                    }
                }

                for(std::size_t t = 0; t < T; ++t){
                    features.back()[cy * N * T + cx * T + t] = bins[t];
                }
            }
        }
    }

    // Frame normalization

    for (std::size_t real_x = 0; real_x < width; ++real_x) {
        features[real_x] *= (1.0 / etl::sum(features[real_x]));
    }

#ifdef LOCAL_MEAN_SCALING
    local_linear_feature_scaling(features);
#endif

#ifdef LOCAL_MEAN_SCALING
    local_linear_feature_scaling(features);
#endif

    return features;
}

std::vector<etl::dyn_vector<weight>> standard_features_vinciarelli_2004(const cv::Mat& clean_image) {
    const auto height  = static_cast<std::size_t>(clean_image.size().height);
    const auto width  = static_cast<std::size_t>(clean_image.size().width);

    const std::size_t w     = height;    //Square window
    const std::size_t left  = w / 2;     // Left context
    const std::size_t right = w / 2 - 1; // Right context

    std::vector<etl::dyn_vector<weight>> features;

    // 1. Convert image to float

    cv::Mat clean_image_float(clean_image.size(), CV_64F);
    clean_image.convertTo(clean_image_float, clean_image_float.type());

    // 2. Enlarge the image to avoid boundary effects

    cv::Mat L(cv::Size(width + height, height), CV_64F);

    L = cv::Scalar(1.0);

#ifndef OPENCV_23
    clean_image_float.copyTo(L(cv::Rect(left , 0, width, height)));
#else
    //This is not efficient, but we need this because of the ANCIENT retarded grid machines
    for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
            L.at<double>(y, x + left) = clean_image_float.at<double>(y, x);
        }
    }
#endif

    // 3. Invert the image (for sums)

    for (std::size_t y = 0; y < static_cast<std::size_t>(L.size().height); ++y) {
        auto* L_ptr = L.ptr<double>(y);

        for (std::size_t x = 0; x < static_cast<std::size_t>(L.size().width); ++x) {
            L_ptr[x] = 1.0 - L_ptr[x];
        }
    }

    // 4. Sliding window

    cpp_assert(height % 2 == 0, "Vinciarelli2004 has only been implemented for even windows");

    constexpr const std::size_t M = 4; //Number of cells
    constexpr const std::size_t N = 4; //Number of cells

    for (std::size_t real_x = 0; real_x < width; ++real_x) {
        const std::size_t real_first = std::max(static_cast<int>(real_x) - static_cast<int>(left), 0);
        const std::size_t real_last  = std::min(real_x + right, width);

        // Compute the upper and lower contours inside the window

        std::size_t lower = 0;
        std::size_t upper = 0;

        features.emplace_back(M * N);

        for(std::size_t i = real_first; i < real_last; ++i){
            std::size_t local_lower = 0;
            for (std::size_t y = height - 1; y > 0; --y) {
                if (clean_image.at<uint8_t>(y, i) == 0.0) {
                    local_lower = y;
                    break;
                }
            }

            std::size_t local_upper = 0;
            for (std::size_t y = 0; y < height; ++y) {
                if (clean_image.at<uint8_t>(y, i) == 0.0) {
                    local_upper = y;
                    break;
                }
            }

            if(i == real_first){
                lower = local_lower;
                upper = local_upper;
            } else if(!(local_lower == 0 && local_upper == 0)) {
                lower = std::max(lower, local_lower);
                upper = std::min(upper, local_upper);
            }
        }

        upper = upper > 0 ? upper - 1 : upper;
        lower = lower < height - 1 ? lower + 1 : lower;

        // Compute dimensions of the cells

        const std::size_t grid_height = lower - upper;
        const std::size_t cell_width = w / M;
        const std::size_t cell_height = grid_height / N;

        // Compute the total sum of pixels in the window

        double total_sum = 0.0;

        //TODO These loops can be simplified
        for (std::size_t cy = 0; cy < N; ++cy) {
            for (std::size_t cx = 0; cx < M; ++cx) {
                const auto x_start = real_x + cx * cell_width;
                const auto y_start = upper + cy * cell_height;

                for(std::size_t yy = y_start; yy < y_start + cell_height; ++yy){
                    auto* L_ptr = L.ptr<double>(yy);

                    for(std::size_t xx = x_start; xx < x_start + cell_width; ++xx){
                        total_sum += L_ptr[xx];
                    }
                }
            }
        }

        // Iterate through the cells (4x4)

        for (std::size_t cy = 0; cy < N; ++cy) {
            for (std::size_t cx = 0; cx < M; ++cx) {
                const auto x_start = real_x + cx * cell_width;
                const auto y_start = upper + cy * cell_height;

                double sum = 0.0;

                for(std::size_t yy = y_start; yy < y_start + cell_height; ++yy){
                    auto* L_ptr = L.ptr<double>(yy);

                    for(std::size_t xx = x_start; xx < x_start + cell_width; ++xx){
                        sum += L_ptr[xx];
                    }
                }

                features.back()[cy * N + cx] = sum / total_sum;
            }
        }
    }

    return features;
}

std::vector<etl::dyn_vector<weight>> standard_features_terasawa_2009(const cv::Mat& clean_image) {
    const auto height  = static_cast<std::size_t>(clean_image.size().height);
    const auto width  = static_cast<std::size_t>(clean_image.size().width);

    const std::size_t w     = 16;        //
    const std::size_t left  = w / 2;     // Left context
    //const std::size_t right = w / 2 - 1; // Right context

    std::vector<etl::dyn_vector<weight>> features;

    // 0. Convert image to float
    cv::Mat clean_image_float(clean_image.size(), CV_64F);
    clean_image.convertTo(clean_image_float, clean_image_float.type());

    // 1. Compute the smoothed image
    cv::Mat L;
    cv::GaussianBlur(clean_image_float, L, cv::Size(0, 0), 3.0, 3.0);

    //2. Compute vertical and horizontal gradients
    cv::Mat sGx(clean_image.size(), CV_64F);
    cv::Mat sGy(clean_image.size(), CV_64F);

    const auto outside = 1.0;

    for (std::size_t y = 0; y < height; ++y) {
        auto* sGx_ptr = sGx.ptr<double>(y);
        auto* L_ptr   = L.ptr<double>(y);

        sGx_ptr[0] = L_ptr[1] - outside;

        for (std::size_t x = 1; x < width - 1; ++x) {
            sGx_ptr[x] = L_ptr[x + 1] - L_ptr[x - 1];
        }

        sGx_ptr[width - 1] = outside - L_ptr[width - 1 - 1];
    }

    for (std::size_t y = 1; y < height - 1; ++y) {
        auto* sGy_ptr = sGy.ptr<double>(y);
        auto* Ll_ptr  = L.ptr<double>(y + 1);
        auto* Lr_ptr  = L.ptr<double>(y - 1);

        for (std::size_t x = 0; x < width; ++x) {
            sGy_ptr[x] = Ll_ptr[x] - Lr_ptr[x];
        }
    }

    for (std::size_t x = 0; x < width; ++x) {
        sGy.at<double>(0, x) = L.at<double>(1, x) - outside;
        sGy.at<double>(height - 1, x) = outside - L.at<double>(height - 1 - 1, x);
    }

    // 3. Enlarge the gradients to avoid boundary effects

    cv::Mat Gx(cv::Size(width + height, height), CV_64F);
    cv::Mat Gy(cv::Size(width + height, height), CV_64F);

    Gx = cv::Scalar(0.0);
    Gy = cv::Scalar(0.0);

#ifndef OPENCV_23
    sGx.copyTo(Gx(cv::Rect(left , 0, width, height)));
    sGy.copyTo(Gy(cv::Rect(left , 0, width, height)));
#else
    //This will be frakking slow, but grid machines are RETARDEDLY ANCIENT

    for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
            Gx.at<double>(y, x + left) = sGx.at<double>(y, x);
            Gy.at<double>(y, x + left) = sGy.at<double>(y, x);
        }
    }
#endif

    // 4. Compute magnitude and orientations of the gradients

    cv::Mat m(Gx.size(), CV_64F);
    cv::Mat o(Gx.size(), CV_64F);

    for (std::size_t y = 0; y < static_cast<std::size_t>(Gx.size().height); ++y) {
        auto* Gx_ptr = Gx.ptr<double>(y);
        auto* Gy_ptr = Gy.ptr<double>(y);

        auto* m_ptr = m.ptr<double>(y);
        auto* o_ptr = o.ptr<double>(y);

        for (std::size_t x = 0; x < static_cast<std::size_t>(Gx.size().width); ++x) {
            auto gx = Gx_ptr[x];
            auto gy = Gy_ptr[x];

            m_ptr[x] = std::sqrt(gx * gx + gy * gy);
            o_ptr[x] = std::atan2(gx, gy);
        }
    }

    // 5. Sliding window

    cpp_assert(height % 2 == 0, "Terasawa2009 has only been implemented for even windows");

    constexpr const std::size_t M           = 4;       //Number of cells in horizontal
    constexpr const std::size_t N           = 4;       //Number of cells in vertical
    constexpr const std::size_t T           = 16;      //Number of bins
    constexpr const std::size_t B           = N - 1;   //Number of blocks
    constexpr const std::size_t BM          = M;       //Number of horizontal cells per block
    constexpr const std::size_t BN          = 2;       //Number of vertical cells per block
    constexpr const std::size_t CELLS_BLOCK = BM * BN; //Number of cells per block

    for (std::size_t real_x = 0; real_x < width; ++real_x) {
        features.emplace_back(B * CELLS_BLOCK * T);

        // Compute dimensions of the cells

        const std::size_t cell_width = w / M;
        const std::size_t cell_height = height / N;

        // Iterate through the blocks

        for(std::size_t by = 0; by < B; ++by){
            std::array<double, CELLS_BLOCK * T> block_bins;
            block_bins.fill(0.0);

            for (std::size_t cy = 0; cy < BN; ++cy) {
                for (std::size_t cx = 0; cx < M; ++cx) {
                    auto real_cy = by + cy;

                    const auto x_start = real_x + cx * cell_width;
                    const auto y_start = real_cy * cell_height;

                    std::array<double, T> bins;
                    bins.fill(0.0);

                    // Attribute each magnitude to a bin

                    for(std::size_t yy = y_start; yy < y_start + cell_height; ++yy){
                        auto* m_ptr = m.ptr<double>(yy);
                        auto* o_ptr = o.ptr<double>(yy);

                        for(std::size_t xx = x_start; xx < x_start + cell_width; ++xx){
                            auto magnitude = m_ptr[xx];
                            auto angle = o_ptr[xx] + M_PI; //In the range [0, 2pi]

                            std::size_t bin = std::size_t(angle / (2 * M_PI / T)) % T;
                            std::size_t next_bin = (bin + 1) % T;

                            auto inside_angle = angle - bin * (2.0 * M_PI / T);
                            auto bin_contrib = ((2.0 * M_PI / T) - inside_angle) / (2.0 * M_PI / T);
                            auto next_contrib = inside_angle / (2.0 * M_PI / T);

                            bins[bin] += bin_contrib * magnitude;
                            bins[next_bin] += next_contrib * magnitude;
                        }
                    }

                    //Accumulate for the whole block

                    for(std::size_t t = 0; t < T; ++t){
                        block_bins[cy * BN * T + cx * T + t] += bins[t];
                    }
                }
            }

            // Normalize the block

            constexpr const double epsilon = 16.0;

            double norm = 0.0;

            for(auto v : block_bins){
                norm += v * v;
            }

            auto normalizer = std::sqrt(norm + epsilon * epsilon);

            for(auto& v : block_bins){
                v /= normalizer;
            }

            // Add the normalized block to the features

            for(std::size_t t = 0; t < CELLS_BLOCK * T; ++t){
                features.back()[by * CELLS_BLOCK * T + t] = block_bins[t];
            }
        }
    }

    return features;
}

std::vector<etl::dyn_vector<weight>> standard_features(const config& conf, const cv::Mat& clean_image) {
    if(conf.manmatha){
        cv::bitwise_not(clean_image, clean_image);
    }

    if(conf.method == Method::Rath2003){
        return standard_features_rath_2003(clean_image);
    } else if(conf.method == Method::Vinciarelli2004){
        return standard_features_vinciarelli_2004(clean_image);
    } else if (conf.method == Method::Rodriguez2008 || conf.method == Method::Terasawa2009) {
        if (conf.manmatha) {
            const auto height = static_cast<std::size_t>(clean_image.size().height);
            const auto width  = static_cast<std::size_t>(clean_image.size().width);

            if (height % 2) {
                cv::Mat normalized(cv::Size(width, height + 1), CV_8U);
                normalized = cv::Scalar(255);

                clean_image.copyTo(normalized(cv::Rect(0, 0, width, height)));

                if (conf.method == Method::Rodriguez2008) {
                    return standard_features_rodriguez_2008(normalized);
                } else if (conf.method == Method::Terasawa2009) {
                    return standard_features_terasawa_2009(normalized);
                }
            }
        }

        if (conf.method == Method::Rodriguez2008) {
            return standard_features_rodriguez_2008(clean_image);
        } else if (conf.method == Method::Terasawa2009) {
            return standard_features_terasawa_2009(clean_image);
        }
    }

    std::vector<etl::dyn_vector<weight>> features;

    const auto width  = static_cast<std::size_t>(clean_image.size().width);
    const auto height = static_cast<std::size_t>(clean_image.size().height);

    for (std::size_t i = 0; i < width; ++i) {
        double lower = 0.0;
        for (std::size_t y = 0; y < height; ++y) {
            if (clean_image.at<uint8_t>(y, i) == 0) {
                lower = y;
                break;
            }
        }

        double upper = 0.0;
        for (std::size_t y = height - 1; y > 0; --y) {
            if (clean_image.at<uint8_t>(y, i) == 0) {
                upper = y;
                break;
            }
        }

        std::size_t black = 0;
        for (std::size_t y = 0; y < height; ++y) {
            if (clean_image.at<uint8_t>(y, i) == 0) {
                ++black;
            }
        }

        std::size_t inner_black = 0;
        for (std::size_t y = lower; y < upper + 1; ++y) {
            if (clean_image.at<uint8_t>(y, i) == 0) {
                ++inner_black;
            }
        }

        std::size_t transitions = 0;
        for (std::size_t y = 1; y < height; ++y) {
            if (clean_image.at<uint8_t>(y - 1, i) == 0 && clean_image.at<uint8_t>(y, i) != 0) {
                ++transitions;
            }
        }

        double gravity = 0;
        double moment = 0;
        for (std::size_t y = 0; y < height; ++y) {
            auto pixel = clean_image.at<uint8_t>(y, i) == 0 ? 0.0 : 1.0;
            gravity += y * pixel;
            moment += y * y * pixel;
        }
        gravity /= height;
        moment /= (height * height);

        if(conf.method == Method::Marti2001){
            features.emplace_back(9);

            auto& f = features.back();

            f[0] = black;
            f[1] = gravity;
            f[2] = moment;
            f[3] = lower;
            f[4] = upper;
            f[5] = 0.0;
            f[6] = 0.0;
            f[7] = transitions;
            f[8] = inner_black;
        } else if(conf.method == Method::Rath2007){
            features.emplace_back(4);

            auto& f = features.back();

            f[0] = black; //Number of black pixels
            f[1] = upper;
            f[2] = lower;
            f[3] = transitions;
        }
    }

    if(conf.method == Method::Marti2001){
        for (std::size_t i = 0; i < width - 1; ++i) {
            //TODO Should be 3 and 4...
            features[i][5] = features[i + 1][1] - features[i][1];
            features[i][6] = features[i + 1][2] - features[i][2];
        }
    } else if (conf.method == Method::Rath2007 && interpolate){
        //Interpolate contour gaps

        //1. Fill the gap starting from column 0 (if any)

        if(features[0][1] == 0.0 && features[0][2] == 0.0){
            for (std::size_t i = 1; i < width; ++i) {
                if(!(features[i][1] == 0.0 && features[i][2] == 0.0)){
                    auto upper = features[i][1];
                    auto lower = features[i][2];

                    while(i-- > 0){
                        features[i][1] = upper;
                        features[i][2] = lower;
                    }

                    break;
                }
            }
        }

        //2. Fill the gap starting from the end (if any)

        if(features[width - 1][1] == 0.0 && features[width - 1][2] == 0.0){
            for (std::size_t i = width - 1; i > 0; --i) {
                if(!(features[i][1] == 0.0 && features[i][2] == 0.0)){
                    auto upper = features[i][1];
                    auto lower = features[i][2];

                    while(i++ < width - 1){
                        features[i][1] = upper;
                        features[i][2] = lower;
                    }

                    break;
                }
            }
        }

        //3. Fill the middle gaps

        for (std::size_t i = 1; i < width - 1; ++i) {
            if(features[i][1] == 0.0 && features[i][2] == 0.0){
                std::size_t end = i;
                for (std::size_t j = i; j < width; ++j) {
                    if(!(features[j][1] == 0.0 && features[j][2] == 0.0)){
                        end = j;
                        break;
                    }
                }

                auto upper_diff = features[end][1] - features[i - 1][1];
                auto lower_diff = features[end][2] - features[i - 1][2];

                auto step = 1.0 / (end - i + 1);

                for(std::size_t j = i; j < end; ++j){
                    features[j][1] = features[i - 1][1] + upper_diff * step * (j - i + 1);
                    features[j][2] = features[i - 1][2] + lower_diff * step * (j - i + 1);
                }
            }
        }
    }

#ifdef LOCAL_LINEAR_SCALING
    local_linear_feature_scaling(features);
#endif

#ifdef LOCAL_MEAN_SCALING
    local_mean_feature_scaling(features);
#endif

    return features;
}

void scale(std::vector<std::vector<etl::dyn_vector<weight>>>& test_features, const config& conf, bool training) {
#ifdef GLOBAL_MEAN_SCALING
    auto scale = global_mean_scaling(test_features, conf, training);
#endif

#ifdef GLOBAL_LINEAR_SCALING
    auto scale = global_linear_scaling(test_features, conf, training);
#endif

#ifdef GLOBAL_SCALING
    for (std::size_t t = 0; t < test_features.size(); ++t) {
        for (std::size_t i = 0; i < test_features[t].size(); ++i) {
            for (std::size_t f = 0; f < test_features.back().back().size(); ++f) {
                test_features[t][i][f] = scale(test_features[t][i][f], conf.scale_a[f], conf.scale_b[f]);
            }
        }
    }
#else
    cpp_unused(test_features);
    cpp_unused(training);
    cpp_unused(conf);
#endif
}

std::vector<std::vector<etl::dyn_vector<weight>>> prepare_outputs(thread_pool& pool, const spot_dataset& dataset, const config& conf, names test_image_names, bool training){
    std::vector<std::vector<etl::dyn_vector<weight>>> test_features(test_image_names.size());

    cpp::parallel_foreach_i(pool, test_image_names.begin(), test_image_names.end(), [&](auto& test_image, std::size_t e){
        test_features[e] = standard_features(conf, dataset.word_images.at(test_image));
    });

    scale(test_features, conf, training);

    return test_features;
}

std::vector<std::vector<etl::dyn_vector<weight>>> compute_reference(thread_pool& pool, const spot_dataset& dataset, const config& conf, names training_images) {
    std::vector<std::vector<etl::dyn_vector<weight>>> ref_a(training_images.size());

    cpp::parallel_foreach_i(pool, training_images.begin(), training_images.end(), [&](auto& training_image, std::size_t e) {
        ref_a[e] = standard_features(conf, dataset.word_images.at(training_image + ".png"));
    });

    scale(ref_a, conf, false);

    return ref_a;
}

parameters get_parameters(const config& /*conf*/){
    parameters parameters;

    parameters.sc_band = 0.12;

    return parameters;
}

template <typename Set>
void evaluate_dtw(const spot_dataset& dataset, const Set& set, const config& conf, names train_word_names, names test_image_names, bool training) {
    thread_pool pool;

    auto parameters = get_parameters(conf);

    // 0. Select the keywords

    auto keywords = select_keywords(dataset, set, train_word_names, test_image_names);

    // 1. Select a folder

    auto result_folder = select_folder("./results/");

    // 2. Generate the rel files

    generate_rel_files(result_folder, dataset, test_image_names, keywords);

    // 3. Prepare all the outputs

    auto test_features = prepare_outputs(pool, dataset, conf, test_image_names, training);

    // 4. Evaluate the performances

    std::cout << "Evaluate performance..." << std::endl;

    std::vector<double> eer(keywords.size());
    std::vector<double> ap(keywords.size());

    std::ofstream global_top_stream(result_folder + "/global_top_file");
    std::ofstream local_top_stream(result_folder + "/local_top_file");

    for (std::size_t k = 0; k < keywords.size(); ++k) {
        auto& keyword = keywords[k];

        // a) Select the training images

        auto training_images = select_training_images(dataset, keyword, train_word_names);

        // b) Compute the reference features

        auto ref = compute_reference(pool, dataset, conf, training_images);

        // c) Compute the distances

        auto diffs = compute_distances(conf, pool, dataset, test_features, ref, training_images,
            test_image_names, train_word_names,
            parameters, [&](names train_names){ return compute_reference(pool, dataset, conf, train_names); });

#ifdef DEBUG_DISTANCES
        std::cout << "Keyword: " << keyword << std::endl;

        std::sort(diffs.begin(), diffs.end(), [](auto& a, auto& b) { return a.second < b.second; });

        for(auto& diff : diffs){
            std::cout << dataset.word_labels.at(diff.first) << " <-> " << diff.second << std::endl;
        }
#endif

        // d) Update the local stats

        update_stats(k, result_folder, dataset, keyword, diffs, eer, ap, global_top_stream, local_top_stream, test_image_names);

        if((k + 1) % (keywords.size() / 10) == 0){
            std::cout << ((k + 1) / (keywords.size() / 10)) * 10 << "%" << std::endl;
        }
    }

    std::cout << "... done" << std::endl;

    // 5. Finalize the results

    std::cout << keywords.size() << " keywords evaluated" << std::endl;

    double mean_eer = std::accumulate(eer.begin(), eer.end(), 0.0) / eer.size();
    double mean_ap  = std::accumulate(ap.begin(), ap.end(), 0.0) / ap.size();

    std::cout << "Mean EER: " << mean_eer << std::endl;
    std::cout << "Mean AP: " << mean_ap << std::endl;
}

void load_features(const spot_dataset& dataset, const config& conf, const std::vector<std::string>& test_image_names, bool training) {
    std::cout << "Load features ..." << std::endl;

    std::vector<std::vector<etl::dyn_vector<weight>>> test_features;

    for (auto& test_image : test_image_names) {
        test_features.push_back(standard_features(conf, dataset.word_images.at(test_image)));
    }

    scale(test_features, conf, training);

    std::cout << "... done" << std::endl;
}

void extract_features(const spot_dataset& dataset, const config& conf, const std::vector<std::string>& test_image_names, bool training, bool silent = false) {
    if(!silent){
        std::cout << "Extract features ..." << std::endl;
    }

    std::vector<std::vector<etl::dyn_vector<weight>>> test_features;

    for (auto& test_image : test_image_names) {
        test_features.push_back(standard_features(conf, dataset.word_images.at(test_image)));
    }

    scale(test_features, conf, training);

    std::string suffix = ".0";

    if(conf.method == Method::Marti2001){
        suffix = ".1";
    } else if(conf.method == Method::Rodriguez2008){
        suffix = ".8";
    } else if(conf.method == Method::Terasawa2009){
        suffix = ".9";
    }

    export_features(conf, test_image_names, test_features, suffix);

    if(!silent){
        std::cout << "... done" << std::endl;
    }
}

} //end of anonymous namespace

void standard_train(
    const spot_dataset& dataset, const spot_dataset_set& set, config& conf,
    names train_word_names, names train_image_names, names valid_image_names, names test_image_names) {

    if (!conf.notrain) {
        std::cout << "Evaluate on training set" << std::endl;
        evaluate_dtw(dataset, set, conf, train_word_names, train_image_names, true);
    }

    if (!conf.novalid) {
        std::cout << "Evaluate on validation set" << std::endl;
        evaluate_dtw(dataset, set, conf, train_word_names, valid_image_names, false);
    }

    std::cout << "Evaluate on test set" << std::endl;
    evaluate_dtw(dataset, set, conf, train_word_names, test_image_names, false);
}

void standard_features(
    const spot_dataset& dataset, const spot_dataset_set& /*set*/, config& conf,
    names /*train_word_names*/, names train_image_names, names valid_image_names, names test_image_names) {

    std::cout << "Extract features on training set" << std::endl;
    extract_features(dataset, conf, train_image_names, true);

    std::cout << "Extract features on validation set" << std::endl;
    extract_features(dataset, conf, valid_image_names, false);

    std::cout << "Extract features on test set" << std::endl;
    extract_features(dataset, conf, test_image_names, false);
}

void standard_runtime(const spot_dataset& dataset, config& conf, names image_names) {
    extract_features(dataset, conf, image_names, true, true);
}
