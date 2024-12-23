#include "frc971/orin/stage2_keypoint_group_trust.h"
#include <iostream>

struct GroupTrustCounters
{
    float    scoreSum{0.f};
    uint32_t fgWhiteCount{0};
    uint32_t fgBlackCount{0};

    GroupTrustCounters() = default;
    GroupTrustCounters(const float score, const int label, const float confidence) :
        scoreSum(score),
        fgWhiteCount(((score >= confidence) && (label == 1)) ? 1 : 0),
        fgBlackCount(((score >= confidence) && (label == 0)) ? 1 : 0)
    {
    }

    GroupTrustCounters(const GroupTrustCounters &other) = default;
    GroupTrustCounters(GroupTrustCounters &&other) noexcept = default;

    GroupTrustCounters &operator=(const GroupTrustCounters &other) = default;
    GroupTrustCounters &operator=(GroupTrustCounters &&other) noexcept = default;

    GroupTrustCounters &operator+=(const GroupTrustCounters &other)
    {
        scoreSum += other.scoreSum;
        fgWhiteCount += other.fgWhiteCount;
        fgBlackCount += other.fgBlackCount;
        return *this;
    }

    GroupTrustCounters operator+(const GroupTrustCounters &other) const
    {
        return GroupTrustCounters(*this) += other;
    }
};

bool Stage2KeypointGroupTrust::check(const tcb::span<const Stage2KeypointGroup> &stage2keypointGroups,
                                     const float trustConfidence,
                                     const float keypointGroupConfidence)
{
  if (stage2keypointGroups.size() == 0) {
    return false;
  }
  auto counter = GroupTrustCounters(stage2keypointGroups[0].m_score, stage2keypointGroups[0].m_label, keypointGroupConfidence);
  for (size_t i = 1; i < stage2keypointGroups.size(); i++) {
    counter += GroupTrustCounters(stage2keypointGroups[i].m_score, stage2keypointGroups[i].m_label, keypointGroupConfidence);
  }
  std::cout << "scoreSum = " << counter.scoreSum << " scoreSum/size = " << counter.scoreSum / stage2keypointGroups.size() <<" fgWhiteCount = " << counter.fgWhiteCount << " fgBlackCount = " << counter.fgBlackCount << " size = " << stage2keypointGroups.size() << std::endl;
  if ((counter.scoreSum / static_cast<float>(stage2keypointGroups.size())) < trustConfidence) {
    return false;
  }
  if (counter.fgWhiteCount <= 4) {
    return false;
  }
  return true;
}
