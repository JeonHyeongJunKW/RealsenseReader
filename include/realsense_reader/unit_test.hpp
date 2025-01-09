#ifndef REALSENSE_READER__UNIT_TEST_HPP_
#define REALSENSE_READER__UNIT_TEST_HPP_


#include <memory>
#include <vector>

#include "opencv2/opencv.hpp"


namespace realsense_reader
{
struct UnitRegion
{
  cv::Point2i top_left;
  int width;
  int height;
};

struct UnitRegionResult
{
  bool valid_depth;
  bool sufficient_valid_pixel;
  float mean_depth;
  float valid_pixel_ratio;
};

struct Color
{
  cv::Scalar red = cv::Scalar(0, 0, 255);
  cv::Scalar green = cv::Scalar(0, 255, 0);
  cv::Scalar blue = cv::Scalar(255, 0, 0);
};

class UnitTest
{
public:
  explicit UnitTest() {}
  virtual ~UnitTest() {}

  void run(cv::Mat & depth_image, cv::Mat & color_image);
private:
  void examine_depth(const cv::Mat & depth_image);
  void print_total_result();
  void reset_data();
  cv::Scalar set_color(
    const bool valid_depth,
    const bool sufficient_valid_pixel);
  std::vector<cv::Point2f> set_roi(const cv::Mat & depth_image);
  void set_unit_regions(const std::vector<cv::Point2f> & roi);
  bool validate_depth_for_each_zone(cv::Mat & depth_image, cv::Mat & color_image);
  void visualize_result(cv::Mat & depth_image, cv::Mat & color_image);

  std::vector<UnitRegion> unit_regions_;
  std::vector<UnitRegionResult> result_;

  const Color color_;
  const float roi_ratio_ = 0.95f;
  const int region_cols_num_ = 7;
  const int region_rows_num_ = 5;
  const float min_depth_range_ = 0.1f;
  const float max_depth_range_ = 15.0f;

  const float depth_gt_ = 5.0f;
  const float mean_depth_thres_ = 5.0f;
  const float valid_pixel_ratio_thres_ = 0.9f;
};
}  // namespace realsense_reader
#endif  // REALSENSE_READER__UNIT_TEST_HPP_
