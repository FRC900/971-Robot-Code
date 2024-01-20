#include "frc971/wpilib/buffered_solenoid.h"

#include "frc971/wpilib/buffered_pcm.h"

namespace frc971::wpilib {

void BufferedSolenoid::Set(bool value) { pcm_->DoSet(number_, value); }

}  // namespace frc971::wpilib
