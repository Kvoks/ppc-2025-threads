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

#include <boost/serialization/vector.hpp>
#include <boost/serialization/utility.hpp>
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

/*void RadixSort(std::vector<uint64_t>& array) {
  const int bits_in_byte = 8;
  const int total_bits = 64;
  const int bucket_count = 256;

  std::vector<uint64_t> buffer(array.size());
  std::vector<int> frequency(bucket_count, 0);

  for (int shift = 0; shift < total_bits; shift += bits_in_byte) {
    std::fill(frequency.begin(), frequency.end(), 0);

    for (uint64_t number : array) {
      uint8_t bucket = static_cast<uint8_t>((number >> shift) & 0xFF);
      frequency[bucket]++;
    }

    for (int i = 1; i < bucket_count; i++) {
      frequency[i] += frequency[i - 1];
    }

    for (int i = static_cast<int>(array.size()) - 1; i >= 0; i--) {
      uint8_t bucket = static_cast<uint8_t>((array[i] >> shift) & 0xFF);
      buffer[--frequency[bucket]] = array[i];
    }

    array.swap(buffer);
  }
}*/

void RadixSort(std::vector<uint64_t>& array, int thread_count) {
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
}
}  // namespace
}  // namespace khovansky_d_double_radix_batcher_all
bool khovansky_d_double_radix_batcher_all::RadixAll::PreProcessingImpl() {
  int rank = world_.rank();
  int size = world_.size();

  if (rank == 0) {
    size_t total = task_data->inputs_count[0];
    size_t per_proc = (total + size - 1) / size;
    input_.resize(per_proc * size, std::numeric_limits<double>::max());
    double* src = reinterpret_cast<double*>(task_data->inputs[0]);
    std::copy(src, src + total, input_.begin());
  }

  size_t per_proc;
  if (rank == 0) per_proc = input_.size() / size;
  boost::mpi::broadcast(world_, per_proc, 0);

  std::vector<double> local(per_proc);
  boost::mpi::scatter(world_, input_, local.data(), per_proc, 0);
  input_.swap(local);
  
  return true;
}

bool khovansky_d_double_radix_batcher_all::RadixAll::ValidationImpl() {
  if (world_.rank() != 0) {
    return true;
  }

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
  int rank = world_.rank();
  int size = world_.size();
  
  std::vector<uint64_t> local;
  for (auto d : input_) local.push_back(EncodeDoubleToUint64(d));
  size_t n = local.size();
  const int thread_count = std::max(1, std::min(static_cast<int>(n), ppc::util::GetPPCNumThreads()));
  RadixSort(local, thread_count);

  int stages = static_cast<int>(std::ceil(std::log2(size)));
  for (int stage = 0; stage < stages; ++stage) {
    int offset = 1 << (stages - stage - 1);
    
    for (int step = offset; step > 0; step >>= 1) {
      int partner = rank ^ step;
      if (partner >= size) continue;

      const int data_size = static_cast<int>(local.size());

      boost::mpi::request reqs[2];
      std::vector<uint64_t> recv_data(local.size());

      if (rank < partner) {
        reqs[0] = world_.isend(partner, 0, local.data(), data_size);
        reqs[1] = world_.irecv(partner, 0, recv_data.data(), data_size);
      } else {
        reqs[0] = world_.irecv(partner, 0, recv_data.data(), data_size);
        reqs[1] = world_.isend(partner, 0, local.data(), data_size);
      }
      boost::mpi::wait_all(reqs, reqs + 2);

      std::vector<uint64_t> merged;
      std::merge(local.begin(), local.end(),
                recv_data.begin(), recv_data.end(),
                std::back_inserter(merged));

      if (rank < partner) {
        local.assign(merged.begin(), merged.begin() + local.size());
      } else {
        local.assign(merged.end() - local.size(), merged.end());
      }
    }
    world_.barrier();
  }

  output_.resize(local.size());
  for (size_t i = 0; i < local.size(); ++i) {
    output_[i] = DecodeUint64ToDouble(local[i]);
  }
  
  return true;
}

bool khovansky_d_double_radix_batcher_all::RadixAll::PostProcessingImpl() {
  std::vector<std::vector<double>> all_data;

  if (world_.rank() == 0) {
    boost::mpi::gather(world_, output_, all_data, 0);
  } else {
    boost::mpi::gather(world_, output_, 0);
  }

  if (world_.rank() == 0) {
    std::vector<double> gathered;
    for (const auto& part : all_data) {
      gathered.insert(gathered.end(), part.begin(), part.end());
    }

    auto it = std::remove(gathered.begin(), gathered.end(), 
                        std::numeric_limits<double>::max());
    gathered.erase(it, gathered.end());

    std::sort(gathered.begin(), gathered.end());

    std::copy(gathered.begin(), gathered.end(), 
             reinterpret_cast<double*>(task_data->outputs[0]));
  }

  return true;
}
