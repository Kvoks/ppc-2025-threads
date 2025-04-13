#include "tbb/khovansky_d_double_radix_batcher/include/ops_tbb.hpp"

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>
#include <tbb/tbb.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "oneapi/tbb/task_arena.h"
#include "oneapi/tbb/task_group.h"
#include "oneapi/tbb/enumerable_thread_specific.h"

namespace khovansky_d_double_radix_batcher_tbb {
namespace {
uint64_t EncodeDoubleToUint64(double value) {
  uint64_t bit_representation = 0;
  std::memcpy(&bit_representation, &value, sizeof(value));

  if ((bit_representation >> 63) != 0) {
    return ~bit_representation;
  }
  return bit_representation ^ (1ULL << 63);
}

double DecodeUint64ToDouble(uint64_t encoded) {
  if ((encoded >> 63) != 0) {
    encoded ^= (1ULL << 63);
  } else {
    encoded = ~encoded;
  }

  double result = 0.0;
  std::memcpy(&result, &encoded, sizeof(result));
  return result;
}

void RadixSort(std::vector<uint64_t>& array) {
  const int bits_in_byte = 8;
  const int total_bits = 64;
  const int bucket_count = 256;

  std::vector<uint64_t> buffer(array.size(), 0);

  for (int shift = 0; shift < total_bits; shift += bits_in_byte) {
    tbb::enumerable_thread_specific<std::vector<int>> local_histograms(
        [=] { return std::vector<int>(bucket_count, 0); });

    tbb::parallel_for(tbb::blocked_range<size_t>(0, array.size()), [&](const tbb::blocked_range<size_t>& r) {
      auto& local = local_histograms.local();
      for (size_t i = r.begin(); i < r.end(); ++i) {
        auto bucket = static_cast<uint8_t>((array[i] >> shift) & 0xFF);
        local[bucket]++;
      }
    });

    std::vector<int> frequency(bucket_count, 0);
    for (const auto& local : local_histograms) {
      for (int i = 0; i < bucket_count; ++i) {
        frequency[i] += local[i];
      }
    }

    std::vector<int> offsets(bucket_count, 0);
    for (int i = 1; i < bucket_count; ++i) {
      offsets[i] = offsets[i - 1] + frequency[i - 1];
    }

    tbb::enumerable_thread_specific<std::vector<int>> local_offsets([&]() { return offsets; });

    tbb::parallel_for(tbb::blocked_range<size_t>(0, array.size()), [&](const tbb::blocked_range<size_t>& r) {
      auto& local_off = local_offsets.local();
      for (size_t i = r.begin(); i < r.end(); ++i) {
        uint64_t val = array[i];
        auto bucket = static_cast<uint8_t>((val >> shift) & 0xFF);
        size_t pos = local_off[bucket]++;
        buffer[pos] = val;
      }
    });

    array.swap(buffer);
  }
}

void OddEvenMergeSort(std::vector<uint64_t>& array, int left, int right) {
  if (right - left <= 1) {
    return;
  }

  int middle = left + ((right - left) / 2);

  OddEvenMergeSort(array, left, middle);
  OddEvenMergeSort(array, middle, right);

  for (int i = left; i + 1 < right; i += 2) {
    if (array[i] > array[i + 1]) {
      std::swap(array[i], array[i + 1]);
    }
  }
}

void RadixBatcherSort(std::vector<double>& data) {
  std::vector<uint64_t> transformed_data(data.size(), 0);

  for (std::size_t i = 0; i < data.size(); i++) {
    transformed_data[i] = EncodeDoubleToUint64(data[i]);
  }

  RadixSort(transformed_data);
  OddEvenMergeSort(transformed_data, 0, static_cast<int>(transformed_data.size()));

  for (std::size_t i = 0; i < data.size(); i++) {
    data[i] = DecodeUint64ToDouble(transformed_data[i]);
  }
}
}  // namespace
}  // namespace khovansky_d_double_radix_batcher_tbb

bool khovansky_d_double_radix_batcher_tbb::RadixTBB::PreProcessingImpl() {
  auto* in_ptr = reinterpret_cast<double*>(task_data->inputs[0]);

  unsigned int input_size = task_data->inputs_count[0];
  unsigned int output_size = task_data->outputs_count[0];

  input_ = std::vector<double>(in_ptr, in_ptr + input_size);
  output_ = std::vector<double>(output_size, 0);

  return true;
}

bool khovansky_d_double_radix_batcher_tbb::RadixTBB::ValidationImpl() {
  if (!task_data) {
    return false;
  }

  if (task_data->inputs[0] == nullptr && task_data->inputs_count[0] == 0) {
    return false;
  }

  if (task_data->outputs[0] == nullptr) {
    return false;
  }

  if (task_data->inputs_count[0] < 2) {
    return false;
  }

  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool khovansky_d_double_radix_batcher_tbb::RadixTBB::RunImpl() {
  oneapi::tbb::task_arena arena(1);
  arena.execute([&] {
    tbb::task_group tg;
    tg.run([&] {
      output_ = input_;
      khovansky_d_double_radix_batcher_tbb::RadixBatcherSort(output_);
    });
    tg.wait();
  });
  return true;
}

bool khovansky_d_double_radix_batcher_tbb::RadixTBB::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<double*>(task_data->outputs[0])[i] = output_[i];
  }

  return true;
}
