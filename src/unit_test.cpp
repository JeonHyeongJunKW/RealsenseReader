#include "realsense_reader/unit_test.hpp"


void realsense_reader::UnitTest::examine_depth(const cv::Mat & depth_image)
{
  const int region_num = unit_regions_.size();
  result_.reserve(region_num);
  for (auto region : unit_regions_) {
    const float total_pixel_num = static_cast<float>(region.width * region.height);
    float mean_depth = 0.0f;
    int valid_pixel_num = 0;
    for (int y = region.top_left.y; y < region.top_left.y + region.height; y++) {
      for (int x = region.top_left.x; x < region.top_left.x + region.width; x++) {
        float current_depth = depth_image.at<float>(y, x);
        mean_depth += current_depth;
        if (current_depth > min_depth_range_ && current_depth < max_depth_range_) {
          valid_pixel_num++;
        }
      }
    }

    UnitRegionResult result;
    result.mean_depth = mean_depth / total_pixel_num;
    result.valid_pixel_ratio = static_cast<float>(valid_pixel_num) / total_pixel_num;

    result.valid_depth = true;
    if (std::fabs(depth_gt_ - result.mean_depth) > mean_depth_thres_) {
      result.valid_depth = false;
    }

    result.sufficient_valid_pixel = true;
    if (result.valid_pixel_ratio < valid_pixel_ratio_thres_) {
      result.sufficient_valid_pixel = false;
    }

    result_.emplace_back(result);
  }
}

void realsense_reader::UnitTest::print_total_result()
{
  const int region_num = result_.size();
  std::cout << "Total region : " << region_num << std::endl;
  for (auto result : result_) {
    std::cout << "Mean depth : " << result.mean_depth << " Valid pixel ratio : " <<
      result.valid_pixel_ratio << " Valid depth : " << result.valid_depth <<
      " Sufficient valid pixel : " << result.sufficient_valid_pixel << std::endl;
  }
}

void realsense_reader::UnitTest::reset_data()
{
  unit_regions_.clear();
  result_.clear();
}

void realsense_reader::UnitTest::run(cv::Mat & depth_image, cv::Mat & color_image)
{
  std::cout << "Start Camera unit test" << std::endl;
  if(!validate_depth_for_each_zone(depth_image, color_image)) {
    std::cerr << "Failed to validate depth for each zone" << std::endl;
  }
}

cv::Scalar realsense_reader::UnitTest::set_color(
  const bool valid_depth,
  const bool sufficient_valid_pixel)
{
  if (valid_depth && sufficient_valid_pixel) {
    return color_.green;
  } else if (!valid_depth && !sufficient_valid_pixel) {
    return color_.red;
  } else {
    return color_.blue;
  }
}

std::vector<cv::Point2f> realsense_reader::UnitTest::set_roi(const cv::Mat & depth_image)
{
  std::vector<cv::Point2f> roi;

  if (!depth_image.empty()) {
    const int height = depth_image.rows;
    const int width = depth_image.cols;

    const int x1 = static_cast<int>((1.0f - roi_ratio_) * width);
    const int y1 = static_cast<int>((1.0f - roi_ratio_) * height);
    const int x2 = static_cast<int>(roi_ratio_ * width);
    const int y2 = static_cast<int>(roi_ratio_ * height);

    roi.push_back(cv::Point2f(x1, y1));
    roi.push_back(cv::Point2f(x2, y2));
  }

  return roi;
}

void realsense_reader::UnitTest::set_unit_regions(const std::vector<cv::Point2f> & roi)
{
  int cols_delta = static_cast<int>(roi[1].x - roi[0].x);
  int rows_delta = static_cast<int>(roi[1].y - roi[0].y);
  cols_delta /= region_cols_num_;
  rows_delta /= region_rows_num_;

  const int region_num = region_cols_num_ * region_rows_num_;
  unit_regions_.resize(region_num);
  for (int j = 0; j < region_rows_num_; j++) {
    for (int i = 0; i < region_cols_num_; i++) {
      int index = region_cols_num_ * j + i;
      unit_regions_.at(index).top_left =
        cv::Point2i(roi[0].x + cols_delta * i, roi[0].y + rows_delta * j);
      unit_regions_.at(index).width = cols_delta;
      unit_regions_.at(index).height = rows_delta;
    }
  }
}

bool realsense_reader::UnitTest::validate_depth_for_each_zone(
  cv::Mat & depth_image, cv::Mat & color_image)
{
  std::vector<cv::Point2f> roi = set_roi(depth_image);
  if (roi.empty()) {
    return false;
  }

  set_unit_regions(roi);
  examine_depth(depth_image);
  visualize_result(depth_image, color_image);

  reset_data();

  return true;
}

void realsense_reader::UnitTest::visualize_result(
  cv::Mat & depth_image, cv::Mat & color_image)
{
  const int region_num = unit_regions_.size();
  for (int i = 0; i < region_num; i++) {
    cv::Scalar color = set_color(result_.at(i).valid_depth, result_.at(i).sufficient_valid_pixel);
    const cv::Point2i top_left = unit_regions_.at(i).top_left;
    cv::rectangle(
      color_image,
      top_left,
      cv::Point(top_left.x + unit_regions_.at(i).width, top_left.y + unit_regions_.at(i).height),
      color, 1);
    cv::rectangle(
      depth_image,
      top_left,
      cv::Point(top_left.x + unit_regions_.at(i).width, top_left.y + unit_regions_.at(i).height),
      cv::Scalar(255, 0 ,0), 1);
  }

  print_total_result();
}
