#include "aos/starter/subprocess.h"

#include <errno.h>
#include <grp.h>
#include <pwd.h>
#include <signal.h>
#include <stdlib.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <compare>
#include <iterator>
#include <ostream>
#include <ratio>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "glog/logging.h"

#include "aos/util/file.h"
#include "aos/util/process_info_generated.h"

namespace aos::starter {

// Blocks all signals while an instance of this class is in scope.
class ScopedCompleteSignalBlocker {
 public:
  ScopedCompleteSignalBlocker() {
    sigset_t mask;
    sigfillset(&mask);
    // Remember the current mask.
    PCHECK(sigprocmask(SIG_SETMASK, &mask, &old_mask_) == 0);
  }

  ~ScopedCompleteSignalBlocker() {
    // Restore the remembered mask.
    PCHECK(sigprocmask(SIG_SETMASK, &old_mask_, nullptr) == 0);
  }

 private:
  sigset_t old_mask_;
};

namespace {
std::optional<ino_t> GetInodeForPath(const std::filesystem::path &path) {
  struct stat stat_buf;
  if (0 != stat(path.c_str(), &stat_buf)) {
    return std::nullopt;
  }
  return stat_buf.st_ino;
}
bool InodeChanged(const std::filesystem::path &path, ino_t previous_inode) {
  const std::optional<ino_t> current_inode = GetInodeForPath(path);
  if (!current_inode.has_value()) {
    return true;
  }
  return current_inode.value() != previous_inode;
}
}  // namespace

std::filesystem::path ResolvePath(std::string_view command) {
  std::filesystem::path command_path = command;
  if (command.find("/") != std::string_view::npos) {
    CHECK(std::filesystem::exists(command_path))
        << ": " << command << " does not exist.";
    return std::filesystem::canonical(command_path);
  }
  const char *system_path = getenv("PATH");
  std::string system_path_buffer;
  if (system_path == nullptr) {
    const size_t default_path_length = confstr(_CS_PATH, nullptr, 0);
    PCHECK(default_path_length != 0) << ": Unable to resolve " << command;
    system_path_buffer.resize(default_path_length);
    confstr(_CS_PATH, system_path_buffer.data(), system_path_buffer.size());
    system_path = system_path_buffer.c_str();
    VLOG(2) << "Using default path of " << system_path
            << " in the absence of PATH being set.";
  }
  const std::vector<std::string_view> search_paths =
      absl::StrSplit(system_path, ':');
  for (const std::string_view search_path : search_paths) {
    const std::filesystem::path candidate =
        std::filesystem::path(search_path) / command_path;
    if (std::filesystem::exists(candidate)) {
      return std::filesystem::canonical(candidate);
    }
  }
  LOG(FATAL) << "Unable to resolve " << command;
}

// RAII class to become root and restore back to the original user and group
// afterwards.
class Sudo {
 public:
  Sudo() {
    // Save what we were.
    PCHECK(getresuid(&ruid_, &euid_, &suid_) == 0);
    PCHECK(getresgid(&rgid_, &egid_, &sgid_) == 0);

    // Become root.
    PCHECK(setresuid(/* ruid */ 0 /* root */, /* euid */ 0, /* suid */ 0) == 0)
        << ": Failed to become root";
    PCHECK(setresgid(/* ruid */ 0 /* root */, /* euid */ 0, /* suid */ 0) == 0)
        << ": Failed to become root";
  }

  ~Sudo() {
    // And recover.
    PCHECK(setresgid(rgid_, egid_, sgid_) == 0);
    PCHECK(setresuid(ruid_, euid_, suid_) == 0);
  }

  uid_t ruid_, euid_, suid_;
  gid_t rgid_, egid_, sgid_;
};

MemoryCGroup::MemoryCGroup(std::string_view name, Create should_create)
    : cgroup_(absl::StrCat("/sys/fs/cgroup/memory/aos_", name)),
      should_create_(should_create) {
  if (should_create_ == Create::kDoCreate) {
    Sudo sudo;
    int ret = mkdir(cgroup_.c_str(), 0755);

    if (ret != 0) {
      if (errno == EEXIST) {
        PCHECK(rmdir(cgroup_.c_str()) == 0)
            << ": Failed to remove previous cgroup " << cgroup_;
        ret = mkdir(cgroup_.c_str(), 0755);
      }
    }

    if (ret != 0) {
      PLOG(FATAL) << ": Failed to create cgroup aos_" << cgroup_
                  << ", do you have permission?";
    }
  }
}

void MemoryCGroup::AddTid(pid_t pid) {
  if (pid == 0) {
    pid = getpid();
  }
  if (should_create_ == Create::kDoCreate) {
    Sudo sudo;
    util::WriteStringToFileOrDie(absl::StrCat(cgroup_, "/tasks").c_str(),
                                 std::to_string(pid));
  } else {
    util::WriteStringToFileOrDie(absl::StrCat(cgroup_, "/tasks").c_str(),
                                 std::to_string(pid));
  }
}

void MemoryCGroup::SetLimit(std::string_view limit_name, uint64_t limit_value) {
  if (should_create_ == Create::kDoCreate) {
    Sudo sudo;
    util::WriteStringToFileOrDie(absl::StrCat(cgroup_, "/", limit_name).c_str(),
                                 std::to_string(limit_value));
  } else {
    util::WriteStringToFileOrDie(absl::StrCat(cgroup_, "/", limit_name).c_str(),
                                 std::to_string(limit_value));
  }
}

MemoryCGroup::~MemoryCGroup() {
  if (should_create_ == Create::kDoCreate) {
    Sudo sudo;
    PCHECK(rmdir(absl::StrCat(cgroup_).c_str()) == 0);
  }
}

SignalListener::SignalListener(aos::ShmEventLoop *loop,
                               std::function<void(signalfd_siginfo)> callback)
    : SignalListener(loop->epoll(), std::move(callback)) {}

SignalListener::SignalListener(aos::internal::EPoll *epoll,
                               std::function<void(signalfd_siginfo)> callback)
    : SignalListener(epoll, callback,
                     {SIGHUP, SIGINT, SIGQUIT, SIGABRT, SIGFPE, SIGSEGV,
                      SIGPIPE, SIGTERM, SIGBUS, SIGXCPU, SIGCHLD}) {}

SignalListener::SignalListener(aos::ShmEventLoop *loop,
                               std::function<void(signalfd_siginfo)> callback,
                               std::initializer_list<unsigned int> signals)
    : SignalListener(loop->epoll(), std::move(callback), std::move(signals)) {}

SignalListener::SignalListener(aos::internal::EPoll *epoll,
                               std::function<void(signalfd_siginfo)> callback,
                               std::initializer_list<unsigned int> signals)
    : epoll_(epoll), callback_(std::move(callback)), signalfd_(signals) {
  epoll_->OnReadable(signalfd_.fd(), [this] {
    signalfd_siginfo info = signalfd_.Read();

    if (info.ssi_signo == 0) {
      LOG(WARNING) << "Could not read " << sizeof(signalfd_siginfo) << " bytes";
      return;
    }

    callback_(info);
  });
}

SignalListener::~SignalListener() { epoll_->DeleteFd(signalfd_.fd()); }

Application::Application(std::string_view name,
                         std::string_view executable_name,
                         aos::EventLoop *event_loop,
                         std::function<void()> on_change,
                         QuietLogging quiet_flag)
    : name_(name),
      path_(ResolvePath(executable_name)),
      event_loop_(event_loop),
      start_timer_(event_loop_->AddTimer([this] {
        status_ = aos::starter::State::RUNNING;
        LOG_IF(INFO, quiet_flag_ == QuietLogging::kNo)
            << "Started '" << name_ << "' pid: " << pid_;
        // Check if the file on disk changed while we were starting up. We allow
        // this state for the same reason that we don't just use /proc/$pid/exe
        // to determine if the file is deleted--we may be running a script or
        // sudo or the such and determining the state of the file that we
        // actually care about sounds like more work than we want to deal with.
        if (InodeChanged(path_, pre_fork_inode_)) {
          file_state_ = FileState::CHANGED_DURING_STARTUP;
        } else {
          file_state_ = FileState::NO_CHANGE;
        }

        OnChange();
      })),
      restart_timer_(event_loop_->AddTimer([this] { DoStart(); })),
      stop_timer_(event_loop_->AddTimer([this] {
        if (kill(pid_, SIGKILL) == 0) {
          LOG_IF(WARNING, quiet_flag_ == QuietLogging::kNo ||
                              quiet_flag_ == QuietLogging::kNotForDebugging)
              << "Failed to stop, sending SIGKILL to '" << name_
              << "' pid: " << pid_;
        } else {
          PLOG_IF(WARNING, quiet_flag_ == QuietLogging::kNo ||
                               quiet_flag_ == QuietLogging::kNotForDebugging)
              << "Failed to send SIGKILL to '" << name_ << "' pid: " << pid_;
          stop_timer_->Schedule(event_loop_->monotonic_now() +
                                std::chrono::seconds(1));
        }
      })),
      pipe_timer_(event_loop_->AddTimer([this]() { FetchOutputs(); })),
      child_status_handler_(
          event_loop_->AddTimer([this]() { MaybeHandleSignal(); })),
      on_change_({on_change}),
      quiet_flag_(quiet_flag) {
  // Keep the length of the timer name bounded to some reasonable length.
  start_timer_->set_name(absl::StrCat("app_start_", name.substr(0, 10)));
  restart_timer_->set_name(absl::StrCat("app_restart_", name.substr(0, 10)));
  stop_timer_->set_name(absl::StrCat("app_stop_", name.substr(0, 10)));
  pipe_timer_->set_name(absl::StrCat("app_pipe_", name.substr(0, 10)));
  child_status_handler_->set_name(
      absl::StrCat("app_status_handler_", name.substr(0, 10)));
  // Every second poll to check if the child is dead. This is used as a
  // default for the case where the user is not directly catching SIGCHLD
  // and calling MaybeHandleSignal for us.
  child_status_handler_->Schedule(event_loop_->monotonic_now(),
                                  std::chrono::seconds(1));
}

Application::Application(const aos::Application *application,
                         aos::EventLoop *event_loop,
                         std::function<void()> on_change,
                         QuietLogging quiet_flag)
    : Application(application->name()->string_view(),
                  application->has_executable_name()
                      ? application->executable_name()->string_view()
                      : application->name()->string_view(),
                  event_loop, on_change, quiet_flag) {
  user_name_ = application->has_user() ? application->user()->str() : "";
  user_ = application->has_user() ? FindUid(user_name_.c_str()) : std::nullopt;
  group_ = application->has_user() ? FindPrimaryGidForUser(user_name_.c_str())
                                   : std::nullopt;
  autostart_ = application->autostart();
  autorestart_ = application->autorestart();
  if (application->has_args()) {
    set_args(*application->args());
  }

  if (application->has_memory_limit() && application->memory_limit() > 0) {
    SetMemoryLimit(application->memory_limit());
  }

  set_stop_grace_period(std::chrono::nanoseconds(application->stop_time()));
}

void Application::DoStart() {
  if (status_ != aos::starter::State::WAITING) {
    return;
  }

  start_timer_->Disable();
  restart_timer_->Disable();

  status_pipes_ = util::ScopedPipe::MakePipe();

  if (capture_stdout_) {
    stdout_pipes_ = util::ScopedPipe::MakePipe();
    stdout_.clear();
  }
  if (capture_stderr_) {
    stderr_pipes_ = util::ScopedPipe::MakePipe();
    stderr_.clear();
  }

  pipe_timer_->Schedule(event_loop_->monotonic_now(),
                        std::chrono::milliseconds(100));

  {
    // Block all signals during the fork() call. Together with the default
    // signal handler restoration below, This prevents signal handlers from
    // getting called in the child and accidentally affecting the parent. In
    // particular, the exit handler for shm_event_loop could be called here if
    // we don't exec() quickly enough.
    ScopedCompleteSignalBlocker signal_blocker;
    {
      const std::optional<ino_t> inode = GetInodeForPath(path_);
      CHECK(inode.has_value())
          << ": " << path_ << " does not seem to be stat'able.";
      pre_fork_inode_ = inode.value();
    }
    const pid_t pid = fork();

    if (pid != 0) {
      if (pid == -1) {
        PLOG_IF(WARNING, quiet_flag_ == QuietLogging::kNo ||
                             quiet_flag_ == QuietLogging::kNotForDebugging)
            << "Failed to fork '" << name_ << "'";
        stop_reason_ = aos::starter::LastStopReason::FORK_ERR;
        status_ = aos::starter::State::STOPPED;
      } else {
        pid_ = pid;
        id_ = next_id_++;
        start_time_ = event_loop_->monotonic_now();
        status_ = aos::starter::State::STARTING;
        latest_timing_report_version_.reset();
        LOG_IF(INFO, quiet_flag_ == QuietLogging::kNo)
            << "Starting '" << name_ << "' pid " << pid_;

        // Set up timer which moves application to RUNNING state if it is still
        // alive in 1 second.
        start_timer_->Schedule(event_loop_->monotonic_now() +
                               std::chrono::seconds(1));
        // Since we are the parent process, clear our write-side of all the
        // pipes.
        status_pipes_.write.reset();
        stdout_pipes_.write.reset();
        stderr_pipes_.write.reset();
      }
      OnChange();
      return;
    }

    // Clear any signal handlers so that they don't accidentally interfere with
    // the parent process. Is there a better way to iterate over all the
    // signals? Right now we're just dealing with the most common ones.
    for (int signal : {SIGINT, SIGHUP, SIGTERM}) {
      struct sigaction action;
      sigemptyset(&action.sa_mask);
      action.sa_flags = 0;
      action.sa_handler = SIG_DFL;
      PCHECK(sigaction(signal, &action, nullptr) == 0);
    }
  }

  if (memory_cgroup_) {
    memory_cgroup_->AddTid();
  }

  // Since we are the child process, clear our read-side of all the pipes.
  status_pipes_.read.reset();
  stdout_pipes_.read.reset();
  stderr_pipes_.read.reset();

  // The status pipe will not be needed if the execve succeeds.
  status_pipes_.write->SetCloexec();

  // Clear out signal mask of parent so forked process receives all signals
  // normally.
  sigset_t empty_mask;
  sigemptyset(&empty_mask);
  sigprocmask(SIG_SETMASK, &empty_mask, nullptr);

  // Cleanup children if starter dies in a way that is not handled gracefully.
  if (prctl(PR_SET_PDEATHSIG, SIGKILL) == -1) {
    status_pipes_.write->Write(
        static_cast<uint32_t>(aos::starter::LastStopReason::SET_PRCTL_ERR));
    PLOG(FATAL) << "Could not set PR_SET_PDEATHSIG to SIGKILL";
  }

  if (group_) {
    CHECK(!user_name_.empty());
    // The manpage for setgroups says we just need CAP_SETGID, but empirically
    // we also need the effective UID to be 0 to make it work. user_ must also
    // be set so we change this effective UID back later.
    CHECK(user_);
    if (seteuid(0) == -1) {
      status_pipes_.write->Write(
          static_cast<uint32_t>(aos::starter::LastStopReason::SET_GRP_ERR));
      PLOG(FATAL) << "Could not seteuid(0) for " << name_
                  << " in preparation for setting groups";
    }
    if (initgroups(user_name_.c_str(), *group_) == -1) {
      status_pipes_.write->Write(
          static_cast<uint32_t>(aos::starter::LastStopReason::SET_GRP_ERR));
      PLOG(FATAL) << "Could not initialize normal groups for " << name_
                  << " as " << user_name_ << " with " << *group_;
    }
    if (setgid(*group_) == -1) {
      status_pipes_.write->Write(
          static_cast<uint32_t>(aos::starter::LastStopReason::SET_GRP_ERR));
      PLOG(FATAL) << "Could not set group for " << name_ << " to " << *group_;
    }
  }

  if (user_) {
    if (setuid(*user_) == -1) {
      status_pipes_.write->Write(
          static_cast<uint32_t>(aos::starter::LastStopReason::SET_USR_ERR));
      PLOG(FATAL) << "Could not set user for " << name_ << " to " << *user_;
    }
  }

  if (capture_stdout_) {
    PCHECK(STDOUT_FILENO == dup2(stdout_pipes_.write->fd(), STDOUT_FILENO));
    stdout_pipes_.write.reset();
  }

  if (capture_stderr_) {
    PCHECK(STDERR_FILENO == dup2(stderr_pipes_.write->fd(), STDERR_FILENO));
    stderr_pipes_.write.reset();
  }

  if (run_as_sudo_) {
    // For sudo we must supply the actual path
    args_.insert(args_.begin(), path_.c_str());
    args_.insert(args_.begin(), kSudo);
  } else {
    // argv[0] should be the program name
    args_.insert(args_.begin(), name_);
  }

  std::vector<char *> cargs = CArgs();
  const char *path = run_as_sudo_ ? kSudo : path_.c_str();
  execvp(path, cargs.data());

  // If we got here, something went wrong
  status_pipes_.write->Write(
      static_cast<uint32_t>(aos::starter::LastStopReason::EXECV_ERR));
  PLOG_IF(WARNING, quiet_flag_ == QuietLogging::kNo ||
                       quiet_flag_ == QuietLogging::kNotForDebugging)
      << "Could not execute " << name_ << " (" << path_ << ')';

  _exit(EXIT_FAILURE);
}

void Application::ObserveTimingReport(
    const aos::monotonic_clock::time_point send_time,
    const aos::timing::Report *msg) {
  if (msg->name()->string_view() == name_ && msg->pid() == pid_ &&
      msg->has_version()) {
    latest_timing_report_version_ = msg->version()->str();
    last_timing_report_ = send_time;
  }
}

void Application::FetchOutputs() {
  if (capture_stdout_) {
    stdout_pipes_.read->Read(&stdout_);
  }
  if (capture_stderr_) {
    stderr_pipes_.read->Read(&stderr_);
  }
}

const std::string &Application::GetStdout() {
  CHECK(capture_stdout_);
  FetchOutputs();
  return stdout_;
}

const std::string &Application::GetStderr() {
  CHECK(capture_stderr_);
  FetchOutputs();
  return stderr_;
}

void Application::DoStop(bool restart) {
  // If stop or restart received, the old state of these is no longer applicable
  // so cancel both.
  restart_timer_->Disable();
  start_timer_->Disable();

  FetchOutputs();

  switch (status_) {
    case aos::starter::State::STARTING:
    case aos::starter::State::RUNNING: {
      file_state_ = FileState::NOT_RUNNING;
      LOG_IF(INFO, quiet_flag_ == QuietLogging::kNo ||
                       quiet_flag_ == QuietLogging::kNotForDebugging)
          << "Stopping '" << name_ << "' pid: " << pid_ << " with signal "
          << SIGINT;
      status_ = aos::starter::State::STOPPING;

      if (kill(pid_, SIGINT) != 0) {
        PLOG_IF(INFO, quiet_flag_ == QuietLogging::kNo ||
                          quiet_flag_ == QuietLogging::kNotForDebugging)
            << "Failed to send signal " << SIGINT << " to '" << name_
            << "' pid: " << pid_;
      }

      // Watchdog timer to SIGKILL application if it is still running 1 second
      // after SIGINT
      stop_timer_->Schedule(event_loop_->monotonic_now() + stop_grace_period_);
      queue_restart_ = restart;
      OnChange();
      break;
    }
    case aos::starter::State::WAITING: {
      // If waiting to restart, and receives restart, skip the waiting period
      // and restart immediately. If stop received, all we have to do is move
      // to the STOPPED state.
      if (restart) {
        DoStart();
      } else {
        status_ = aos::starter::State::STOPPED;
        OnChange();
      }
      break;
    }
    case aos::starter::State::STOPPING: {
      // If the application is already stopping, then we just need to update the
      // restart flag to the most recent status.
      queue_restart_ = restart;
      break;
    }
    case aos::starter::State::STOPPED: {
      // Restart immediately if the application is already stopped
      if (restart) {
        status_ = aos::starter::State::WAITING;
        DoStart();
      }
      break;
    }
  }
}

void Application::QueueStart() {
  status_ = aos::starter::State::WAITING;

  LOG_IF(INFO, quiet_flag_ == QuietLogging::kNo)
      << "Restarting " << name_ << " in 3 seconds";
  restart_timer_->Schedule(event_loop_->monotonic_now() +
                           std::chrono::seconds(3));
  start_timer_->Disable();
  stop_timer_->Disable();
  OnChange();
}

std::vector<char *> Application::CArgs() {
  std::vector<char *> cargs;
  std::transform(args_.begin(), args_.end(), std::back_inserter(cargs),
                 [](std::string &str) { return str.data(); });
  cargs.push_back(nullptr);
  return cargs;
}

void Application::set_args(
    const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>> &v) {
  args_.clear();
  std::transform(v.begin(), v.end(), std::back_inserter(args_),
                 [](const flatbuffers::String *str) { return str->str(); });
}

void Application::set_args(std::vector<std::string> args) {
  args_ = std::move(args);
}

void Application::set_capture_stdout(bool capture) {
  capture_stdout_ = capture;
}

void Application::set_capture_stderr(bool capture) {
  capture_stderr_ = capture;
}

std::optional<uid_t> Application::FindUid(const char *name) {
  // TODO(austin): Use the reentrant version.  This should be safe.
  struct passwd *user_data = getpwnam(name);
  if (user_data != nullptr) {
    return user_data->pw_uid;
  } else {
    LOG(FATAL) << "Could not find user " << name;
    return std::nullopt;
  }
}

std::optional<gid_t> Application::FindPrimaryGidForUser(const char *name) {
  // TODO(austin): Use the reentrant version.  This should be safe.
  struct passwd *user_data = getpwnam(name);
  if (user_data != nullptr) {
    return user_data->pw_gid;
  } else {
    LOG(FATAL) << "Could not find user " << name;
    return std::nullopt;
  }
}

FileState Application::UpdateFileState() {
  // On every call, check if a different file is present on disk. Note that
  // while the applications is running, the file cannot be changed without the
  // inode changing.
  // We could presumably use inotify or the such to watch the file instead,
  // but this works and we do not expect substantial cost from reading the inode
  // of a file every time we send out a status message.
  if (InodeChanged(path_, pre_fork_inode_)) {
    switch (file_state_) {
      case FileState::NO_CHANGE:
        file_state_ = FileState::CHANGED;
        break;
      case FileState::NOT_RUNNING:
      case FileState::CHANGED_DURING_STARTUP:
      case FileState::CHANGED:
        break;
    }
  }
  return file_state_;
}

flatbuffers::Offset<aos::starter::ApplicationStatus>
Application::PopulateStatus(flatbuffers::FlatBufferBuilder *builder,
                            util::Top *top) {
  UpdateFileState();

  CHECK(builder != nullptr);
  auto name_fbs = builder->CreateString(name_);

  const bool valid_pid = pid_ > 0 && status_ != aos::starter::State::STOPPED;
  const flatbuffers::Offset<util::ProcessInfo> process_info =
      valid_pid ? top->InfoForProcess(builder, pid_)
                : flatbuffers::Offset<util::ProcessInfo>();

  aos::starter::ApplicationStatus::Builder status_builder(*builder);
  status_builder.add_name(name_fbs);
  status_builder.add_state(status_);
  if (exit_code_.has_value()) {
    status_builder.add_last_exit_code(exit_code_.value());
  }
  status_builder.add_has_active_timing_report(
      last_timing_report_ +
          // Leave a bit of margin on the timing report receipt time, to allow
          // for timing errors.
          3 * std::chrono::milliseconds(FLAGS_timing_report_ms) >
      event_loop_->monotonic_now());
  status_builder.add_last_stop_reason(stop_reason_);
  if (pid_ != -1) {
    status_builder.add_pid(pid_);
    status_builder.add_id(id_);
  }
  // Note that even if process_info is null, calling add_process_info is fine.
  status_builder.add_process_info(process_info);
  status_builder.add_last_start_time(start_time_.time_since_epoch().count());
  status_builder.add_file_state(file_state_);
  return status_builder.Finish();
}

void Application::Terminate() {
  stop_reason_ = aos::starter::LastStopReason::TERMINATE;
  DoStop(false);
  terminating_ = true;
}

void Application::HandleCommand(aos::starter::Command cmd) {
  switch (cmd) {
    case aos::starter::Command::START: {
      switch (status_) {
        case aos::starter::State::WAITING: {
          restart_timer_->Disable();
          DoStart();
          break;
        }
        case aos::starter::State::STARTING: {
          break;
        }
        case aos::starter::State::RUNNING: {
          break;
        }
        case aos::starter::State::STOPPING: {
          queue_restart_ = true;
          break;
        }
        case aos::starter::State::STOPPED: {
          status_ = aos::starter::State::WAITING;
          DoStart();
          break;
        }
      }
      break;
    }
    case aos::starter::Command::STOP: {
      stop_reason_ = aos::starter::LastStopReason::STOP_REQUESTED;
      DoStop(false);
      break;
    }
    case aos::starter::Command::RESTART: {
      stop_reason_ = aos::starter::LastStopReason::RESTART_REQUESTED;
      DoStop(true);
      break;
    }
  }
}

bool Application::MaybeHandleSignal() {
  int status;

  if (status_ == aos::starter::State::WAITING ||
      status_ == aos::starter::State::STOPPED) {
    // We can't possibly have received a signal meant for this process.
    return false;
  }

  // Check if the status of this process has changed
  // The PID won't be -1 if this application has ever been run successfully
  if (pid_ == -1 || waitpid(pid_, &status, WNOHANG) != pid_) {
    return false;
  }

  // Check that the event was the process exiting
  if (!WIFEXITED(status) && !WIFSIGNALED(status)) {
    return false;
  }

  start_timer_->Disable();
  exit_time_ = event_loop_->monotonic_now();
  exit_code_ = WIFEXITED(status) ? WEXITSTATUS(status) : WTERMSIG(status);
  file_state_ = FileState::NOT_RUNNING;

  if (auto read_result = status_pipes_.read->Read()) {
    stop_reason_ = static_cast<aos::starter::LastStopReason>(*read_result);
  }

  const std::string starter_version_string =
      absl::StrCat("starter version '",
                   event_loop_->VersionString().value_or("unknown"), "'");
  switch (status_) {
    case aos::starter::State::STARTING: {
      if (exit_code_.value() == 0) {
        LOG_IF(INFO, quiet_flag_ == QuietLogging::kNo)
            << "Application '" << name_ << "' pid " << pid_
            << " exited with status " << exit_code_.value() << " and "
            << starter_version_string;
      } else {
        LOG_IF(WARNING, quiet_flag_ == QuietLogging::kNo ||
                            quiet_flag_ == QuietLogging::kNotForDebugging)
            << "Failed to start '" << name_ << "' on pid " << pid_
            << " : Exited with status " << exit_code_.value() << " and "
            << starter_version_string;
      }
      if (autorestart()) {
        QueueStart();
      } else {
        status_ = aos::starter::State::STOPPED;
        OnChange();
      }
      break;
    }
    case aos::starter::State::RUNNING: {
      if (exit_code_.value() == 0) {
        LOG_IF(INFO, quiet_flag_ == QuietLogging::kNo)
            << "Application '" << name_ << "' pid " << pid_
            << " exited with status " << exit_code_.value();
      } else {
        if (quiet_flag_ == QuietLogging::kNo ||
            quiet_flag_ == QuietLogging::kNotForDebugging) {
          const std::string version_string =
              latest_timing_report_version_.has_value()
                  ? absl::StrCat("version '",
                                 latest_timing_report_version_.value(), "'")
                  : starter_version_string;
          LOG_IF(WARNING, quiet_flag_ == QuietLogging::kNo)
              << "Application '" << name_ << "' pid " << pid_ << " "
              << version_string << " exited unexpectedly with status "
              << exit_code_.value();
        }
      }
      if (autorestart()) {
        QueueStart();
      } else {
        status_ = aos::starter::State::STOPPED;
        OnChange();
      }
      break;
    }
    case aos::starter::State::STOPPING: {
      LOG_IF(INFO, quiet_flag_ == QuietLogging::kNo)
          << "Successfully stopped '" << name_ << "' pid: " << pid_
          << " with status " << exit_code_.value();
      status_ = aos::starter::State::STOPPED;

      // Disable force stop timer since the process already died
      stop_timer_->Disable();

      OnChange();
      if (terminating_) {
        return true;
      }

      if (queue_restart_) {
        queue_restart_ = false;
        status_ = aos::starter::State::WAITING;
        DoStart();
      }
      break;
    }
    case aos::starter::State::WAITING:
    case aos::starter::State::STOPPED: {
      __builtin_unreachable();
      break;
    }
  }

  return false;
}

void Application::OnChange() {
  for (auto &fn : on_change_) {
    fn();
  }
}

Application::~Application() {
  start_timer_->Disable();
  restart_timer_->Disable();
  stop_timer_->Disable();
  pipe_timer_->Disable();
  child_status_handler_->Disable();
}

}  // namespace aos::starter
