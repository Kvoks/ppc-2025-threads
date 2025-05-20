#include "mpi/khovansky_d_double_radix_batcher/include/ops_all.hpp"

#include <mpi.h>

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/all_gather.hpp>
#include <boost/mpi/collectives/gather.hpp>
#include <boost/mpi/collectives/scatter.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <future>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

namespace khovansky_d_double_radix_batcher_all {
namespace {
uint64_t EncodeDoubleToUint64(double value) {
  uint64_t bit_representation = 0;
  std::memcpy(&bit_representation, &value, sizeof(value));

  if ((bit_representation >> 63) != 0) {
    return ~bit_representation;
  }
  return bit_representation ^ (1ULL << 63);
}

double DecodeUint64ToDouble(uint64_t transformed_data) {
  if ((transformed_data >> 63) != 0) {
    transformed_data ^= (1ULL << 63);
  } else {
    transformed_data = ~transformed_data;
  }

  double result = 0.0;
  std::memcpy(&result, &transformed_data, sizeof(result));
  return result;
}

/*void RadixSort(std::vector<uint64_t>& array, int thread_count) {
  const int bits_in_byte = 8;
  const int total_bits = 64;
  const int bucket_count = 256;

  std::vector<uint64_t> buffer(array.size(), 0);
  std::vector<std::vector<int>> local_frequencies(thread_count, std::vector<int>(bucket_count, 0));

  for (int shift = 0; shift < total_bits; shift += bits_in_byte) {
    std::vector<std::thread> threads;
    size_t n = array.size();
    size_t block_size = (n + thread_count - 1) / thread_count;

    for (int t = 0; t < thread_count; ++t) {
      size_t begin = t * block_size;
      size_t end = std::min(begin + block_size, n);

      threads.emplace_back([&, begin, end, t]() {
        for (size_t i = begin; i < end; ++i) {
          auto bucket = static_cast<uint8_t>((array[i] >> shift) & 0xFF);
          local_frequencies[t][bucket]++;
        }
      });
    }
    for (auto& th : threads) {
      th.join();
    }

    std::vector<int> frequency(bucket_count, 0);
    for (int b = 0; b < bucket_count; ++b) {
      for (int t = 0; t < thread_count; ++t) {
        frequency[b] += local_frequencies[t][b];
        local_frequencies[t][b] = 0;
      }
    }

    for (int i = 1; i < bucket_count; i++) {
      frequency[i] += frequency[i - 1];
    }

    for (int i = static_cast<int>(array.size()) - 1; i >= 0; i--) {
      auto bucket = static_cast<uint8_t>((array[i] >> shift) & 0xFF);
      buffer[--frequency[bucket]] = array[i];
    }

    array.swap(buffer);
  }
}*/

void RadixSort(std::vector<uint64_t>& array) {
  const int bits_in_byte = 8;
  const int total_bits = 64;
  const int bucket_count = 256;

  std::vector<uint64_t> buffer(array.size(), 0);
  std::vector<int> frequency(bucket_count, 0);

  for (int shift = 0; shift < total_bits; shift += bits_in_byte) {
    std::ranges::fill(frequency, 0);

    for (uint64_t number : array) {
      auto bucket = static_cast<uint8_t>((number >> shift) & 0xFF);
      frequency[bucket]++;
    }

    for (int i = 1; i < bucket_count; i++) {
      frequency[i] += frequency[i - 1];
    }

    for (int i = static_cast<int>(array.size()) - 1; i >= 0; i--) {
      auto bucket = static_cast<uint8_t>((array[i] >> shift) & 0xFF);
      buffer[--frequency[bucket]] = array[i];
    }

    array.swap(buffer);
  }
}

void OddEvenMerge(std::vector<uint64_t>& local_data, int rank, int size, boost::mpi::communicator& world) {
  int local_size = static_cast<int>(local_data.size());
  std::vector<uint64_t> recv_buffer(local_size);

  for (int phase = 0; phase < size; ++phase) {
    int partner = ((rank + phase) % 2 == 0) ? rank + 1 : rank - 1;

    if (partner < 0 || partner >= size) continue;

    world.send(partner, 0, local_data);
    world.recv(partner, 0, recv_buffer);

    std::vector<uint64_t> merged;
    merged.reserve(2 * local_size);
    std::merge(local_data.begin(), local_data.end(), recv_buffer.begin(), recv_buffer.end(),
               std::back_inserter(merged));

    if (rank < partner)
      std::copy(merged.begin(), merged.begin() + local_size, local_data.begin());
    else
      std::copy(merged.end() - local_size, merged.end(), local_data.begin());
  }
}

/*void RadixBatcherSort(std::vector<double>& local_data, int rank, int size, boost::mpi::communicator& world) {
  std::vector<uint64_t> transformed_data(local_data.size());
  size_t n = local_data.size();
  const int thread_count = std::max(1, std::min(static_cast<int>(n), ppc::util::GetPPCNumThreads()));
  size_t block_size = (n + thread_count - 1) / thread_count;
  std::vector<std::thread> threads(thread_count);

  for (int t = 0; t < thread_count; ++t) {
    threads[t] = std::thread([&local_data, &transformed_data, t, block_size, n]() {
      size_t begin = t * block_size;
      size_t end = std::min(begin + block_size, n);
      for (size_t i = begin; i < end; ++i) {
        transformed_data[i] = EncodeDoubleToUint64(local_data[i]);
      }
    });
  }
  for (auto& th : threads) {
    th.join();
  }

  RadixSort(transformed_data, thread_count);
  OddEvenMerge(transformed_data, rank, size, world);

  threads.clear();
  threads.resize(thread_count);

  for (int t = 0; t < thread_count; ++t) {
    threads[t] = std::thread([&local_data, &transformed_data, t, block_size, n]() {
      size_t begin = t * block_size;
      size_t end = std::min(begin + block_size, n);
      for (size_t i = begin; i < end; ++i) {
        local_data[i] = DecodeUint64ToDouble(transformed_data[i]);
      }
    });
  }
  for (auto& th : threads) {
    th.join();
  }
}*/

void RadixBatcherSort(std::vector<double>& local_data, int rank, int size, boost::mpi::communicator& world) {
  std::vector<uint64_t> encoded(local_data.size());

  for (size_t i = 0; i < local_data.size(); ++i) {
    encoded[i] = EncodeDoubleToUint64(local_data[i]);
  }
  RadixSort(encoded);
  OddEvenMerge(encoded, rank, size, world);

  for (size_t i = 0; i < local_data.size(); ++i) {
    local_data[i] = DecodeUint64ToDouble(encoded[i]);
  }
}
}  // namespace
}  // namespace khovansky_d_double_radix_batcher_all

// --- Реализация методов класса --- //

bool khovansky_d_double_radix_batcher_all::RadixAll::PreProcessingImpl() {
  int rank = world_.rank();
  int size = world_.size();

  int total_size = static_cast<int>(task_data->inputs_count[0]);
  auto* in_ptr = reinterpret_cast<double*>(task_data->inputs[0]);

  std::vector<int> sendcounts(size, 0);
  std::vector<int> displs(size, 0);

  int base = total_size / size;
  int rem = total_size % size;

  for (int i = 0; i < size; ++i) {
    sendcounts[i] = base + (i < rem ? 1 : 0);
    displs[i] = (i > 0) ? displs[i - 1] + sendcounts[i - 1] : 0;
  }

  input_.resize(sendcounts[rank]);

  MPI_Scatterv(in_ptr, sendcounts.data(), displs.data(), MPI_DOUBLE, input_.data(), sendcounts[rank], MPI_DOUBLE, 0,
               MPI_COMM_WORLD);

  output_.resize(input_.size());
  return true;
}

bool khovansky_d_double_radix_batcher_all::RadixAll::ValidationImpl() {
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

bool khovansky_d_double_radix_batcher_all::RadixAll::RunImpl() {
  output_ = input_;
  RadixBatcherSort(output_, world_.rank(), world_.size(), world_);
  return true;
}

bool khovansky_d_double_radix_batcher_all::RadixAll::PostProcessingImpl() {
  int rank = world_.rank();
  int size = world_.size();

  int total_size = static_cast<int>(task_data->outputs_count[0]);

  std::vector<int> recvcounts(size, 0);
  std::vector<int> displs(size, 0);

  int base = total_size / size;
  int rem = total_size % size;

  for (int i = 0; i < size; ++i) {
    recvcounts[i] = base + (i < rem ? 1 : 0);
    displs[i] = (i > 0) ? displs[i - 1] + recvcounts[i - 1] : 0;
  }

  MPI_Gatherv(output_.data(), recvcounts[rank], MPI_DOUBLE, reinterpret_cast<double*>(task_data->outputs[0]),
              recvcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  return true;
}