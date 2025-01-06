#include <iostream>
#include <string>

#include "opencv2/opencv.hpp"

#include "realsense_reader/image_handler.hpp"

int main(int argc, char ** argv)
{
  (void) argc;
  (void) argv;

  realsense_reader::ImageHandler handler;

  handler.run();

  return 0;
}
