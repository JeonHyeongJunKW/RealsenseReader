// Copyright 2025 Hyeongjun Jeon

#include <iostream>
#include <string>

#include "opencv2/opencv.hpp"

#include "realsense_reader/image_handler.hpp"

int main(int argc, char ** argv)
{
  (void) argc;
  (void) argv;
  int target_camera_index = 0;
  if (argc >= 2) {
    target_camera_index = std::stoi(argv[1]);
  }

  std::cout << "target_camera_index: " << target_camera_index << std::endl;

  realsense_reader::ImageHandler handler(target_camera_index);

  handler.run();

  return 0;
}
