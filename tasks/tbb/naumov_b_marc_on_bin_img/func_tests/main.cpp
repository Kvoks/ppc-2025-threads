#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <set>
#include <vector>

#include "core/task/include/task.hpp"
#include "tbb/naumov_b_marc_on_bin_img/include/ops_tbb.hpp"

namespace {
void VerifyBinaryOutput(const std::vector<int> &in, const std::vector<int> &out) {
  for (size_t i = 0; i < in.size(); ++i) {
    if (in[i] == 1) {
      EXPECT_GT(out[i], 0);
    } else {
      EXPECT_EQ(out[i], 0);
    }
  }
}

void CheckTopNeighbor(const std::vector<int> &in, const std::vector<int> &out, int i, int j, int n) {
  if (i > 0 && in[((i - 1) * n) + j] == 1) {
    EXPECT_EQ(out[(i * n) + j], out[((i - 1) * n) + j]);
  }
}

void CheckLeftNeighbor(const std::vector<int> &in, const std::vector<int> &out, int i, int j, int n) {
  if (j > 0 && in[(i * n) + (j - 1)] == 1) {
    EXPECT_EQ(out[(i * n) + j], out[(i * n) + (j - 1)]);
  }
}

void VerifyNeighborConsistency(const std::vector<int> &in, const std::vector<int> &out, int m, int n) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      const int idx = (i * n) + j;
      if (in[idx] != 1) {
        continue;
      }

      CheckTopNeighbor(in, out, i, j, n);
      CheckLeftNeighbor(in, out, i, j, n);
    }
  }
}
}  // namespace

TEST(naumov_b_marc_on_bin_img_tbb, Validation_1) {
  int m = 3;
  int n = 3;

  std::vector<int> in = {1, 2, 1, 0, 1, 0, 1, 0, 1};
  std::vector<int> out(m * n, 0);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(m);
  task_data_tbb->inputs_count.emplace_back(n);
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(m);
  task_data_tbb->outputs_count.emplace_back(n);

  naumov_b_marc_on_bin_img_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  EXPECT_FALSE(test_task_tbb.Validation());
}

TEST(naumov_b_marc_on_bin_img_tbb, Validation_2) {
  int m = 3;
  int n = 3;

  std::vector<int> in;
  std::vector<int> out(m * n, 0);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(m);
  task_data_tbb->inputs_count.emplace_back(n);
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(m);
  task_data_tbb->outputs_count.emplace_back(n);

  naumov_b_marc_on_bin_img_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  EXPECT_FALSE(test_task_tbb.Validation());
}

TEST(naumov_b_marc_on_bin_img_tbb, Validation_3) {
  int m = 0;
  int n = 0;

  std::vector<int> in;
  std::vector<int> out;

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(m);
  task_data_tbb->inputs_count.emplace_back(n);
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(m);
  task_data_tbb->outputs_count.emplace_back(n);

  naumov_b_marc_on_bin_img_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  EXPECT_FALSE(test_task_tbb.Validation());
}

TEST(naumov_b_marc_on_bin_img_tbb, SingleCtbbonentInCorner) {
  int m = 3;
  int n = 3;

  std::vector<int> in = {1, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int> exp_out = {1, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int> out(m * n, 0);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(m);
  task_data_tbb->inputs_count.emplace_back(n);
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(m);
  task_data_tbb->outputs_count.emplace_back(n);

  naumov_b_marc_on_bin_img_tbb::TestTaskTBB test_task_tbb(task_data_tbb);

  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  EXPECT_EQ(out, exp_out);
}

TEST(naumov_b_marc_on_bin_img_tbb, RingShape) {
  int m = 5;
  int n = 5;

  std::vector<int> in = {0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0};
  std::vector<int> exp_out = {0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0};
  std::vector<int> out(m * n, 0);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(m);
  task_data_tbb->inputs_count.emplace_back(n);
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(m);
  task_data_tbb->outputs_count.emplace_back(n);

  naumov_b_marc_on_bin_img_tbb::TestTaskTBB test_task_tbb(task_data_tbb);

  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  EXPECT_EQ(out, exp_out);
}

TEST(naumov_b_marc_on_bin_img_tbb, MazeStructure) {
  int m = 7;
  int n = 7;

  std::vector<int> in = {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0,
                         1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1};

  std::vector<int> exp_out = {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 2, 2, 2, 0, 1, 1, 0, 2, 0,
                              2, 0, 1, 1, 0, 2, 2, 2, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1};

  std::vector<int> out(m * n, 0);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(m);
  task_data_tbb->inputs_count.emplace_back(n);
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(m);
  task_data_tbb->outputs_count.emplace_back(n);

  naumov_b_marc_on_bin_img_tbb::TestTaskTBB test_task_tbb(task_data_tbb);

  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  EXPECT_EQ(out, exp_out);
}

TEST(naumov_b_marc_on_bin_img_tbb, SimpleTest_1) {
  int m = 3;
  int n = 4;

  std::vector<int> in = {1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1};
  std::vector<int> exp_out = {1, 1, 0, 0, 1, 1, 0, 2, 0, 0, 0, 2};
  std::vector<int> out(m * n, 0);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(m);
  task_data_tbb->inputs_count.emplace_back(n);
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(m);
  task_data_tbb->outputs_count.emplace_back(n);

  naumov_b_marc_on_bin_img_tbb::TestTaskTBB test_task_tbb(task_data_tbb);

  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  EXPECT_EQ(out, exp_out);
}

TEST(naumov_b_marc_on_bin_img_tbb, SimpleTest_2) {
  int m = 4;
  int n = 4;

  std::vector<int> in = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  std::vector<int> exp_out = {1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4};
  std::vector<int> out(m * n, 0);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(m);
  task_data_tbb->inputs_count.emplace_back(n);
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(m);
  task_data_tbb->outputs_count.emplace_back(n);

  naumov_b_marc_on_bin_img_tbb::TestTaskTBB test_task_tbb(task_data_tbb);

  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  EXPECT_EQ(out, exp_out);
}

TEST(naumov_b_marc_on_bin_img_tbb, SingleCtbbonent) {
  int m = 3;
  int n = 3;

  std::vector<int> in = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<int> exp_out = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<int> out(m * n, 0);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(m);
  task_data_tbb->inputs_count.emplace_back(n);
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(m);
  task_data_tbb->outputs_count.emplace_back(n);

  naumov_b_marc_on_bin_img_tbb::TestTaskTBB test_task_tbb(task_data_tbb);

  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  EXPECT_EQ(out, exp_out);
}

TEST(naumov_b_marc_on_bin_img_tbb, OnlyBackground) {
  int m = 3;
  int n = 3;

  std::vector<int> in = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int> exp_out = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int> out(m * n, 0);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(m);
  task_data_tbb->inputs_count.emplace_back(n);
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(m);
  task_data_tbb->outputs_count.emplace_back(n);

  naumov_b_marc_on_bin_img_tbb::TestTaskTBB test_task_tbb(task_data_tbb);

  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  EXPECT_EQ(out, exp_out);
}

TEST(naumov_b_marc_on_bin_img_tbb, SingleRow) {
  int m = 1;
  int n = 5;

  std::vector<int> in = {1, 0, 1, 0, 1};
  std::vector<int> exp_out = {1, 0, 2, 0, 3};
  std::vector<int> out(m * n, 0);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(m);
  task_data_tbb->inputs_count.emplace_back(n);
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(m);
  task_data_tbb->outputs_count.emplace_back(n);

  naumov_b_marc_on_bin_img_tbb::TestTaskTBB test_task_tbb(task_data_tbb);

  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  EXPECT_EQ(out, exp_out);
}

TEST(naumov_b_marc_on_bin_img_tbb, SingleColumn) {
  int m = 5;
  int n = 1;

  std::vector<int> in = {1, 0, 1, 0, 1};
  std::vector<int> exp_out = {1, 0, 2, 0, 3};
  std::vector<int> out(m * n, 0);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(m);
  task_data_tbb->inputs_count.emplace_back(n);
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(m);
  task_data_tbb->outputs_count.emplace_back(n);

  naumov_b_marc_on_bin_img_tbb::TestTaskTBB test_task_tbb(task_data_tbb);

  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  EXPECT_EQ(out, exp_out);
}

TEST(naumov_b_marc_on_bin_img_tbb, large3) {
  int m = 200;
  int n = 200;

  std::vector<int> in(m * n, 1);
  std::vector<int> out(m * n, 0);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(m);
  task_data_tbb->inputs_count.emplace_back(n);
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(m);
  task_data_tbb->outputs_count.emplace_back(n);

  naumov_b_marc_on_bin_img_tbb::TestTaskTBB test_task_tbb(task_data_tbb);

  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  EXPECT_EQ(in, out);
}

TEST(naumov_b_marc_on_bin_img_tbb, RandomSmallMatrix) {
  constexpr int kM = 10;
  constexpr int kN = 10;

  auto in = naumov_b_marc_on_bin_img_tbb::GenerateRandomBinaryMatrix(kM, kN);
  std::vector<int> out(kM * kN, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(kM);
  task_data->inputs_count.emplace_back(kN);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(kM);
  task_data->outputs_count.emplace_back(kN);

  naumov_b_marc_on_bin_img_tbb::TestTaskTBB test_task_tbb(task_data);
  ASSERT_TRUE(test_task_tbb.Validation());
  ASSERT_TRUE(test_task_tbb.PreProcessing());
  ASSERT_TRUE(test_task_tbb.Run());
  ASSERT_TRUE(test_task_tbb.PostProcessing());

  VerifyBinaryOutput(in, out);
}

TEST(naumov_b_marc_on_bin_img_tbb, RandomLargeMatrix) {
  constexpr int kM = 100;
  constexpr int kN = 100;

  auto in = naumov_b_marc_on_bin_img_tbb::GenerateRandomBinaryMatrix(kM, kN, 0.3);
  std::vector<int> out(kM * kN, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(kM);
  task_data->inputs_count.emplace_back(kN);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(kM);
  task_data->outputs_count.emplace_back(kN);

  naumov_b_marc_on_bin_img_tbb::TestTaskTBB test_task_tbb(task_data);
  ASSERT_TRUE(test_task_tbb.Validation());
  ASSERT_TRUE(test_task_tbb.PreProcessing());
  ASSERT_TRUE(test_task_tbb.Run());
  ASSERT_TRUE(test_task_tbb.PostProcessing());

  VerifyNeighborConsistency(in, out, kM, kN);
}

TEST(naumov_b_marc_on_bin_img_tbb, RandomSparseMatrix) {
  const int m = 50;
  const int n = 50;

  auto in = naumov_b_marc_on_bin_img_tbb::GenerateSparseBinaryMatrix(m, n);
  std::vector<int> out(m * n, 0);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(m);
  task_data_tbb->inputs_count.emplace_back(n);
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(m);
  task_data_tbb->outputs_count.emplace_back(n);

  naumov_b_marc_on_bin_img_tbb::TestTaskTBB test_task_tbb(task_data_tbb);

  ASSERT_TRUE(test_task_tbb.Validation());
  ASSERT_TRUE(test_task_tbb.PreProcessing());
  ASSERT_TRUE(test_task_tbb.Run());
  ASSERT_TRUE(test_task_tbb.PostProcessing());

  std::set<int> unique_labels;
  for (int val : out) {
    if (val > 0) {
      unique_labels.insert(val);
    }
  }

  const size_t ones_count = static_cast<size_t>(std::count(in.begin(), in.end(), 1));
  EXPECT_GE(unique_labels.size(), static_cast<size_t>(ones_count * 0.6));
}

TEST(naumov_b_marc_on_bin_img_tbb, RandomDenseMatrix) {
  const int m = 20;
  const int n = 20;

  auto in = naumov_b_marc_on_bin_img_tbb::GenerateDenseBinaryMatrix(m, n);
  std::vector<int> out(m * n, 0);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(m);
  task_data_tbb->inputs_count.emplace_back(n);
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(m);
  task_data_tbb->outputs_count.emplace_back(n);

  naumov_b_marc_on_bin_img_tbb::TestTaskTBB test_task_tbb(task_data_tbb);

  ASSERT_TRUE(test_task_tbb.Validation());
  ASSERT_TRUE(test_task_tbb.PreProcessing());
  ASSERT_TRUE(test_task_tbb.Run());
  ASSERT_TRUE(test_task_tbb.PostProcessing());

  std::set<int> unique_labels;
  for (int val : out) {
    if (val > 0) {
      unique_labels.insert(val);
    }
  }

  EXPECT_LE(unique_labels.size(), static_cast<size_t>(6));
}

TEST(naumov_b_marc_on_bin_img_tbb, RandomDenseMatrix2) {
  const int m = 17;
  const int n = 23;

  auto in = naumov_b_marc_on_bin_img_tbb::GenerateDenseBinaryMatrix(m, n);
  std::vector<int> out(m * n, 0);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(m);
  task_data_tbb->inputs_count.emplace_back(n);
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(m);
  task_data_tbb->outputs_count.emplace_back(n);

  naumov_b_marc_on_bin_img_tbb::TestTaskTBB test_task_tbb(task_data_tbb);

  ASSERT_TRUE(test_task_tbb.Validation());
  ASSERT_TRUE(test_task_tbb.PreProcessing());
  ASSERT_TRUE(test_task_tbb.Run());
  ASSERT_TRUE(test_task_tbb.PostProcessing());

  std::set<int> unique_labels;
  for (int val : out) {
    if (val > 0) {
      unique_labels.insert(val);
    }
  }

  EXPECT_LE(unique_labels.size(), static_cast<size_t>(5));
}

TEST(naumov_b_marc_on_bin_img_tbb, ZeroByZeroMatrix) {
  int m = 0;
  int n = 0;

  std::vector<int> in;
  std::vector<int> out;

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(m);
  task_data_tbb->inputs_count.emplace_back(n);
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(m);
  task_data_tbb->outputs_count.emplace_back(n);

  naumov_b_marc_on_bin_img_tbb::TestTaskTBB test_task_tbb(task_data_tbb);

  EXPECT_FALSE(test_task_tbb.Validation());
}

TEST(naumov_b_marc_on_bin_img_tbb, SinglePixelMatrix_Background) {
  int m = 1;
  int n = 1;

  std::vector<int> in = {0};
  std::vector<int> exp_out = {0};
  std::vector<int> out(m * n, -1);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(m);
  task_data_tbb->inputs_count.emplace_back(n);
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(m);
  task_data_tbb->outputs_count.emplace_back(n);

  naumov_b_marc_on_bin_img_tbb::TestTaskTBB test_task_tbb(task_data_tbb);

  ASSERT_TRUE(test_task_tbb.Validation());
  ASSERT_TRUE(test_task_tbb.PreProcessing());
  ASSERT_TRUE(test_task_tbb.Run());
  ASSERT_TRUE(test_task_tbb.PostProcessing());
  EXPECT_EQ(out, exp_out);
}

TEST(naumov_b_marc_on_bin_img_tbb, SinglePixelMatrix_Foreground) {
  int m = 1;
  int n = 1;

  std::vector<int> in = {1};
  std::vector<int> exp_out = {1};
  std::vector<int> out(m * n, -1);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(m);
  task_data_tbb->inputs_count.emplace_back(n);
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(m);
  task_data_tbb->outputs_count.emplace_back(n);

  naumov_b_marc_on_bin_img_tbb::TestTaskTBB test_task_tbb(task_data_tbb);

  ASSERT_TRUE(test_task_tbb.Validation());
  ASSERT_TRUE(test_task_tbb.PreProcessing());
  ASSERT_TRUE(test_task_tbb.Run());
  ASSERT_TRUE(test_task_tbb.PostProcessing());
  EXPECT_EQ(out, exp_out);
}