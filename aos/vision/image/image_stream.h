#ifndef AOS_VISION_IMAGE_IMAGE_STREAM_H_
#define AOS_VISION_IMAGE_IMAGE_STREAM_H_

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "aos/time/time.h"
#include "aos/vision/events/epoll_events.h"
#include "aos/vision/image/camera_params.pb.h"
#include "aos/vision/image/image_types.h"
#include "aos/vision/image/reader.h"

namespace aos::vision {

// Converts a camera reader into a virtual base class that calls ProcessImage
// on each new image.
class ImageStreamEvent : public ::aos::events::EpollEvent {
 public:
  static std::unique_ptr<::camera::Reader> GetCamera(
      const std::string &fname, ImageStreamEvent *obj,
      aos::vision::CameraParams params) {
    using namespace std::placeholders;
    std::unique_ptr<::camera::Reader> camread(new ::camera::Reader(
        fname, std::bind(&ImageStreamEvent::ProcessHelper, obj, _1, _2),
        params));
    camread->StartAsync();
    return camread;
  }

  explicit ImageStreamEvent(std::unique_ptr<::camera::Reader> reader)
      : ::aos::events::EpollEvent(reader->fd()), reader_(std::move(reader)) {}

  explicit ImageStreamEvent(const std::string &fname,
                            aos::vision::CameraParams params)
      : ImageStreamEvent(GetCamera(fname, this, params)) {}

  virtual void ProcessImage(DataRef data,
                            aos::monotonic_clock::time_point timestamp) = 0;

  void ReadEvent() override { reader_->HandleFrame(); }

  bool SetExposure(int abs_exp) { return reader_->SetExposure(abs_exp); }

 private:
  void ProcessHelper(DataRef data, aos::monotonic_clock::time_point timestamp);

  std::unique_ptr<::camera::Reader> reader_;
};

}  // namespace aos::vision

#endif  // AOS_VISION_IMAGE_IMAGE_STREAM_H_
