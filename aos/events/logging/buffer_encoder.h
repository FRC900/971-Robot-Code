#ifndef AOS_EVENTS_LOGGING_BUFFER_ENCODER_H_
#define AOS_EVENTS_LOGGING_BUFFER_ENCODER_H_

#include "absl/types/span.h"
#include "flatbuffers/flatbuffers.h"
#include "glog/logging.h"

#include "aos/containers/resizeable_buffer.h"
#include "aos/events/logging/logger_generated.h"
#include "aos/time/time.h"

namespace aos::logger {

// Interface to encode data as it is written to a file.
class DataEncoder {
 public:
  virtual ~DataEncoder() = default;

  // Size of an aligned sector used to detect when the data is aligned enough to
  // use O_DIRECT instead.
  static constexpr size_t kSector = 512u;

  // Interface to copy data into a buffer.
  class Copier {
   public:
    Copier(size_t size) : size_(size) {}

    // Returns the data this will write.
    size_t size() const { return size_; }

    // Writes size() bytes to data, and returns the data written.
    [[nodiscard]] virtual size_t Copy(uint8_t *data, size_t start_byte,
                                      size_t end_byte) = 0;

   private:
    size_t size_;
  };

  // Copies a span.  The span must have a longer lifetime than the coppier is
  // being used.
  class SpanCopier : public Copier {
   public:
    SpanCopier(absl::Span<const uint8_t> data)
        : Copier(data.size()), data_(data) {
      CHECK(data_.data() != nullptr);
    }

    size_t Copy(uint8_t *data, size_t start_byte, size_t end_byte) final {
      DCHECK_LE(start_byte, end_byte);
      DCHECK_LE(end_byte, data_.size());

      std::memcpy(data, data_.data() + start_byte, end_byte - start_byte);
      return end_byte - start_byte;
    }

   private:
    const absl::Span<const uint8_t> data_;
  };

  // Returns true if there is space in the buffer for the next request, or if
  // the output needs to be flushed.
  virtual bool HasSpace(size_t request) const = 0;

  // Returns the space available.
  virtual size_t space() const = 0;

  // Encodes and enqueues the given data encoder.  Starts at the start byte
  // (which must be a multiple of 8 bytes), and goes as far as it can.  Returns
  // the amount encoded. The `encode_duration` is optional, when provided it
  // will be set to the amount of time spent by the encoder during this call.
  virtual size_t Encode(
      Copier *copy, size_t start_byte,
      std::chrono::nanoseconds *encode_duration = nullptr) = 0;

  // Finalizes the encoding process. After this, queue_size() represents the
  // full extent of data which will be written to this file.
  // This function may invoke the encoder to encode any remaining data remaining
  // in the queue. The `encode_duration` is optional, when provided it will be
  // set to the amount of time spent by the encoder during this call. Do not
  // call Encode after calling this method.
  virtual void Finish(std::chrono::nanoseconds *encode_duration = nullptr) = 0;

  // Clears the first n encoded buffers from the queue.
  virtual void Clear(int n) = 0;

  // Returns a view of the queue of encoded buffers.  Valid until any other
  // method on this class is called.
  virtual absl::Span<const absl::Span<const uint8_t>> queue() = 0;

  // Returns the total number of of bytes currently queued up.
  virtual size_t queued_bytes() const = 0;

  // Returns the cumulative number of bytes which have been queued. This
  // includes data which has been removed via Clear.
  virtual size_t total_bytes() const = 0;

  // Returns the number of elements in the queue.
  virtual size_t queue_size() const = 0;
};

// This class does not encode the data. It just claims ownership of the raw data
// and queues it up as is.
class DummyEncoder final : public DataEncoder {
 public:
  DummyEncoder(size_t max_message_size, size_t buffer_size = 128 * 1024);
  DummyEncoder(const DummyEncoder &) = delete;
  DummyEncoder(DummyEncoder &&other) = delete;
  DummyEncoder &operator=(const DummyEncoder &) = delete;
  DummyEncoder &operator=(DummyEncoder &&other) = delete;
  ~DummyEncoder() override = default;

  bool HasSpace(size_t request) const final;
  size_t space() const final;

  // See base class for commments.
  size_t Encode(Copier *copy, size_t start_byte,
                std::chrono::nanoseconds *encode_duration = nullptr) final;
  void Finish(std::chrono::nanoseconds * /*encode_duration*/ = nullptr) final {}
  void Clear(int n) final;
  absl::Span<const absl::Span<const uint8_t>> queue() final;
  size_t queued_bytes() const final;
  size_t total_bytes() const final { return total_bytes_; }
  size_t queue_size() const final {
    return input_buffer_.size() != 0 ? 1u : 0u;
  }

 private:
  size_t total_bytes_ = 0;

  AllocatorResizeableBuffer<aos::AlignedReallocator<kSector>> input_buffer_;
  std::vector<absl::Span<const uint8_t>> return_queue_;
};

// Interface to decode chunks of data. Implementations of this interface will
// manage opening, reading, and closing the file stream.
class DataDecoder {
 public:
  virtual ~DataDecoder() = default;

  // Reads data into the given range. Returns the number of bytes read.
  //
  // Returns less than end-begin if all bytes have been read. Otherwise, this
  // will always fill the whole range.
  virtual size_t Read(uint8_t *begin, uint8_t *end) = 0;

  // Returns the underlying filename, for debugging purposes.
  virtual std::string_view filename() const = 0;
};

// Simply reads the contents of the file into the target buffer.
class DummyDecoder final : public DataDecoder {
 public:
  explicit DummyDecoder(std::string_view filename);
  DummyDecoder(const DummyDecoder &) = delete;
  DummyDecoder(DummyDecoder &&other) = delete;
  DummyDecoder &operator=(const DummyDecoder &) = delete;
  DummyDecoder &operator=(DummyDecoder &&other) = delete;
  ~DummyDecoder() override;

  size_t Read(uint8_t *begin, uint8_t *end) final;
  std::string_view filename() const final { return filename_; }

 private:
  const std::string filename_;

  // File descriptor for the log file.
  int fd_;

  // Cached bit for if we have reached the end of the file.  Otherwise we will
  // hammer on the kernel asking for more data each time we send.
  bool end_of_file_ = false;
};

}  // namespace aos::logger

#endif  // AOS_EVENTS_LOGGING_BUFFER_ENCODER_H_
