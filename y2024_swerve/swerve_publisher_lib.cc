#include "y2024_swerve/swerve_publisher_lib.h"

y2024_swerve::SwervePublisher::SwervePublisher(aos::EventLoop *event_loop,
                                               aos::ExitHandle *exit_handle,
                                               const std::string &filename,
                                               double duration)
    : drivetrain_output_sender_(
          event_loop->MakeSender<frc971::control_loops::swerve::Output>(
              "/drivetrain")) {
  event_loop
      ->AddTimer([this, filename]() {
        auto output_builder = drivetrain_output_sender_.MakeBuilder();

        auto drivetrain_output =
            aos::JsonFileToFlatbuffer<frc971::control_loops::swerve::Output>(
                filename);

        auto copied_flatbuffer =
            aos::CopyFlatBuffer<frc971::control_loops::swerve::Output>(
                drivetrain_output, output_builder.fbb());
        CHECK(drivetrain_output.Verify());

        output_builder.CheckOk(output_builder.Send(copied_flatbuffer));
      })
      ->Schedule(event_loop->monotonic_now(),
                 std::chrono::duration_cast<aos::monotonic_clock::duration>(
                     std::chrono::milliseconds(5)));
  event_loop
      ->AddTimer([this, exit_handle]() {
        auto builder = drivetrain_output_sender_.MakeBuilder();
        frc971::control_loops::swerve::SwerveModuleOutput::Builder
            swerve_module_builder = builder.MakeBuilder<
                frc971::control_loops::swerve::SwerveModuleOutput>();

        swerve_module_builder.add_rotation_current(0.0);
        swerve_module_builder.add_translation_current(0.0);

        auto swerve_module_offset = swerve_module_builder.Finish();

        frc971::control_loops::swerve::Output::Builder
            drivetrain_output_builder =
                builder.MakeBuilder<frc971::control_loops::swerve::Output>();

        drivetrain_output_builder.add_front_left_output(swerve_module_offset);
        drivetrain_output_builder.add_front_right_output(swerve_module_offset);
        drivetrain_output_builder.add_back_left_output(swerve_module_offset);
        drivetrain_output_builder.add_back_right_output(swerve_module_offset);

        builder.CheckOk(builder.Send(drivetrain_output_builder.Finish()));

        exit_handle->Exit();
      })
      ->Schedule(event_loop->monotonic_now() +
                 std::chrono::milliseconds((int)duration));
}
