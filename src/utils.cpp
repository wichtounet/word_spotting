//=======================================================================
// Copyright Baptiste Wicht 2015.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <fstream>

#include "utils.hpp"

etl::dyn_matrix<weight> mat_to_dyn(const config& conf, const cv::Mat& image){
    cv::Mat normalized(cv::Size(WIDTH, HEIGHT), CV_8U);
    normalized = cv::Scalar(255);

    image.copyTo(normalized(cv::Rect((WIDTH - image.size().width) / 2, 0, image.size().width, HEIGHT)));

    cv::Mat scaled_normalized(cv::Size(WIDTH / conf.downscale, HEIGHT / conf.downscale), CV_8U);
    cv::resize(normalized, scaled_normalized, scaled_normalized.size(), cv::INTER_AREA);
    cv::adaptiveThreshold(scaled_normalized, normalized, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 7, 2);

    etl::dyn_matrix<weight> training_image(normalized.size().height, normalized.size().width);

    for(std::size_t y = 0; y < static_cast<std::size_t>(normalized.size().height); ++y){
        for(std::size_t x = 0; x < static_cast<std::size_t>(normalized.size().width); ++x){
            auto pixel = normalized.at<uint8_t>(cv::Point(x, y));

            training_image(y, x) = pixel == 0 ? 0.0 : 1.0;

            if(pixel != 0 && pixel != 255){
                std::cout << "The normalized input image is not binary! pixel:" << static_cast<int>(pixel) << std::endl;
            }
        }
    }

    return training_image;
}

std::vector<etl::dyn_matrix<weight>> mat_to_patches(const config& conf, const cv::Mat& image){
    cv::Mat clean_image;

    cv::Mat scaled_normalized(cv::Size(image.size().width / conf.downscale, image.size().height / conf.downscale), CV_8U);
    cv::resize(image, scaled_normalized, scaled_normalized.size(), cv::INTER_AREA);
    cv::adaptiveThreshold(scaled_normalized, clean_image, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 7, 2);

    std::vector<etl::dyn_matrix<weight>> patches;

    const auto context = conf.patch_width / 2;

    for(std::size_t i = 0; i < static_cast<std::size_t>(clean_image.size().width); i += conf.patch_stride){
        patches.emplace_back(static_cast<std::size_t>(clean_image.size().height), static_cast<std::size_t>(conf.patch_width));

        auto& patch = patches.back();

        for(std::size_t y = 0; y < static_cast<std::size_t>(clean_image.size().height); ++y){
            for(int x = i - context; x < static_cast<int>(i + context); ++x){
                uint8_t pixel = 1;

                if(x >= 0 && x < clean_image.size().width){
                    pixel = image.at<uint8_t>(y, x + i * conf.patch_stride);
                }

                patch(y, x - i + context) = pixel == 0 ? 0.0 : 1.0;
            }
        }
    }

    return patches;
}
