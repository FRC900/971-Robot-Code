#include "aos/vision/blob/transpose.h"

#include <algorithm>
#include <string>

#include "gtest/gtest.h"

#include "aos/vision/blob/test_utils.h"

namespace aos::vision {

TEST(TransposeTest, Tranpspose) {
  RangeImage img = LoadFromTestData(20, R"(
    -----------
    -----  ----
   ------------
   -------------
   ------------
    ----------
   ------------
     ---------
)");

  auto b = Transpose(img);
  auto c = Transpose(b);
  EXPECT_EQ(ShortDebugPrint({img}), ShortDebugPrint({c}));
}

}  // namespace aos::vision
