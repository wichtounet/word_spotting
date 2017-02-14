//=======================================================================
// Copyright Baptiste Wicht 2015-2017.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <fstream>

#include "utils.hpp"

//#define DEBUG_COMMAND

std::pair<int, std::string> exec_command(const std::string& command) {
#ifdef DEBUG_COMMAND
    std::cout << "Command " << command << std::endl;
#endif

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

std::pair<int, std::string> exec_command_safe(const std::string& command) {
    auto result = exec_command(command);

    if(result.first){
        std::cout << "Command failed: " << command << std::endl;
        std::cout << result.second << std::endl;
    }

    return result;
}

etl::dyn_matrix<weight, 3> mat_for_patches(const config& conf, const cv::Mat& image) {
    cv::Mat buffer_image;

    if (conf.downscale > 1) {
        cv::Mat scaled_normalized(
            cv::Size(std::max(1UL, static_cast<size_t>(image.size().width)), std::max(1UL, image.size().height / conf.downscale)),
            CV_8U);
        cv::resize(image, scaled_normalized, scaled_normalized.size(), cv::INTER_AREA);
        cv::adaptiveThreshold(scaled_normalized, buffer_image, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 7, 2);
    }

    const cv::Mat& clean_image = conf.downscale > 1 ? buffer_image : image;

    etl::dyn_matrix<weight, 3> training_image(std::size_t(1), std::size_t(clean_image.size().height), std::size_t(clean_image.size().width));

    for (std::size_t y = 0; y < static_cast<std::size_t>(clean_image.size().height); ++y) {
        for (std::size_t x = 0; x < static_cast<std::size_t>(clean_image.size().width); ++x) {
            auto pixel = clean_image.at<uint8_t>(cv::Point(x, y));

            training_image(0, y, x) = pixel == 0 ? 0.0 : 1.0;

            if (pixel != 0 && pixel != 255) {
                std::cout << "The input image is not binary! pixel:" << static_cast<int>(pixel) << std::endl;
            }
        }
    }

    return training_image;
}

cv::Mat elastic_distort(const cv::Mat& clean_image){
    const std::size_t width = clean_image.size().width;
    const std::size_t height = clean_image.size().height;

    // 0. Generate random displacement fields

    cv::Mat d_x(cv::Size(width, height), CV_64F);
    cv::Mat d_y(cv::Size(width, height), CV_64F);

    static std::random_device rd;
    static std::default_random_engine generator(rd());
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);

    for(std::size_t y = 0; y < height; ++y){
        for(std::size_t x = 0; x < width; ++x){
            d_x.at<double>(y, x) = distribution(generator);
            d_y.at<double>(y, x) = distribution(generator);
        }
    }

    // 1. Gaussian blur the displacement fields

    cv::Mat d_x_blur(cv::Size(width, height), CV_64F);
    cv::Mat d_y_blur(cv::Size(width, height), CV_64F);

    cv::GaussianBlur(d_x, d_x_blur, cv::Size(0,0), 10.0);
    cv::GaussianBlur(d_y, d_y_blur, cv::Size(0,0), 10.0);

    // 2. Normalize the displacement field

    double d_x_sum = 0;
    double d_y_sum = 0;

    for(std::size_t y = 0; y < height; ++y){
        for(std::size_t x = 0; x < width; ++x){
            d_x_sum += d_x_blur.at<double>(y, x);
            d_y_sum += d_y_blur.at<double>(y, x);
        }
    }

    for(std::size_t y = 0; y < height; ++y){
        for(std::size_t x = 0; x < width; ++x){
            d_x_blur.at<double>(y, x) /= d_x_sum;
            d_y_blur.at<double>(y, x) /= d_y_sum;
        }
    }

    // 3. Scale the displacement field

    double alpha = 8;

    for(std::size_t y = 0; y < height; ++y){
        for(std::size_t x = 0; x < width; ++x){
            d_x_blur.at<double>(y, x) *= alpha;
            d_y_blur.at<double>(y, x) *= alpha;
        }
    }

    // 4. Apply the displacement field (using bilinear interpolation)

    cv::Mat d_image(cv::Size(width, height), CV_64F);

    auto safe = [&](auto x, auto y){
        if(x < 0 || y < 0 || x > width - 1  || y > height - 1){
            return double(clean_image.at<uint8_t>(0, 0));
        } else {
            return double(clean_image.at<uint8_t>(y, x));
        }
    };

    for(int y = 0; y < int(height); ++y){
        for(int x = 0; x < int(width); ++x){
            auto dx = d_x_blur.at<double>(y, x);
            auto dy = d_y_blur.at<double>(y, x);

            double px = x + dx;
            double py = y + dy;

            auto a = safe(std::floor(px), std::floor(py));
            auto b = safe(std::ceil(px), std::floor(py));
            auto c = safe(std::ceil(px), std::ceil(py));
            auto d = safe(std::floor(px), std::ceil(py));

            auto e = a * (1.0 - (px - std::floor(px))) + d * (px - std::floor(px));
            auto f = b * (1.0 - (px - std::floor(px))) + c * (px - std::floor(px));

            auto value = e * (1.0 - (py - std::floor(py))) + f * (py - std::floor(py));

            d_image.at<double>(y, x) = value;
        }
    }

    cv::Mat final_image(cv::Size(width, height), CV_8U);
    d_image.convertTo(final_image, CV_8U);

    return final_image;
}

