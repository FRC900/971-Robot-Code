  #include "frc971/orin/stage2_keypoint_group.h"
  #include "frc971/orin/stage2_keypoint.h"

  #include <cstdio>

__host__ Stage2KeypointGroup::Stage2KeypointGroup(void)
{
}

__host__ Stage2KeypointGroup::~Stage2KeypointGroup(void) = default;

__host__ void Stage2KeypointGroup::append(const Stage2Keypoint &stage2Keypoint)
{
    // TODO - what if the label doesn't match previously seen labels for this group?
    m_label = stage2Keypoint.m_label;
    // Score is max confidence of all group member's scores
    m_score = max(m_score, stage2Keypoint.m_scoreCand);
    m_score_sum += stage2Keypoint.m_scoreCand;

    // Keypoints is weighted sum of input keypoints.
    // Weight is normalized confidence of each input
    m_keypoint.x += stage2Keypoint.m_keypointCand.x * stage2Keypoint.m_scoreCand;
    m_keypoint.y += stage2Keypoint.m_keypointCand.y * stage2Keypoint.m_scoreCand;
}

__host__ void Stage2KeypointGroup::end()
{
    m_keypoint.x /= m_score_sum;
    m_keypoint.y /= m_score_sum;
}