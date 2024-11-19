#ifndef STAGE2_KEYPOINT_GROUP_TRUST_INC__
#define STAGE2_KEYPOINT_GROUP_TRUST_INC__
#include "frc971/orin/stage2_keypoint_group.h"
#include "gpu_apriltag/span.hpp"

// Given a list of keypoint groups, check that enough of them
// have a high enough confidence that we trust that the group
// of them is actually from a detection of a real tag.
// Also check that there are enough white blocks in the group
// to be a reasonable approximation of a valid tag.
class Stage2KeypointGroupTrust
{
public:
    Stage2KeypointGroupTrust() = default;

    Stage2KeypointGroupTrust(const Stage2KeypointGroupTrust &other) = delete;
    Stage2KeypointGroupTrust(Stage2KeypointGroupTrust &&other) noexcept = delete;

    Stage2KeypointGroupTrust &operator=(const Stage2KeypointGroupTrust &other) = delete;
    Stage2KeypointGroupTrust &operator=(Stage2KeypointGroupTrust &&other) noexcept = delete;

    virtual ~Stage2KeypointGroupTrust() = default;
    bool check(const tcb::span<const Stage2KeypointGroup> &stage2KeypointGroups,
               const float trustConfidence,
               const float keypointConfidence);
};
#endif