#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <mqueue.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <semaphore.h>
#include <stdio.h>
#include <string.h>
#include <sys/eventfd.h>
#include <sys/ipc.h>
#include <sys/msg.h>
#include <sys/sem.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <compare>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <utility>

#include "absl/flags/flag.h"
#include "absl/flags/usage.h"

#include "aos/condition.h"
#include "aos/init.h"
#include "aos/ipc_lib/event.h"
#include "aos/logging/implementations.h"
#include "aos/mutex/mutex.h"
#include "aos/realtime.h"
#include "aos/time/time.h"

ABSL_FLAG(std::string, method, "", "Which IPC method to use");
ABSL_FLAG(int32_t, messages, 1000000,
          "How many messages to send back and forth");
ABSL_FLAG(int32_t, client_cpu, 0, "CPU to pin client to");
ABSL_FLAG(int32_t, server_cpu, 0, "CPU to pin server to");
ABSL_FLAG(int32_t, client_priority, 1,
          "Realtime priority for client. Negative for don't change");
ABSL_FLAG(int32_t, server_priority, 1,
          "Realtime priority for server. Negative for don't change");

namespace aos {

namespace chrono = ::std::chrono;

// A generic interface for an object which can send some data to another thread
// and back.
//
// One side is called the "server". It should constantly Wait, do something with
// the result, and then call Pong.
// The other side is called the "client". It should repeatedly call Ping.
class PingPongerInterface {
 public:
  // A chunk of memory definitely on its own cache line anywhere sane.
  typedef uint8_t Data[1024] __attribute__((aligned(128)));

  virtual ~PingPongerInterface() {}

  // Returns where the "client" side should write data in preparation to send to
  // the server.
  // The result is valid until the next Ping call.
  virtual Data *PingData() = 0;

  // Sends the data returned from the most recent PingData call to the "server"
  // side and returns its response.
  // PingData must be called exactly once before each call of this method.
  // The result is valid until the next PingData call.
  virtual const Data *Ping() = 0;

  // Waits for a Ping call and then returns the associated data.
  // The result is valid until the beginning of the next Pong call.
  virtual const Data *Wait() = 0;

  // Returns where the "server" side should write data in preparation to send
  // back to the "client".
  // The result is valid until the next Pong call.
  virtual Data *PongData() = 0;

  // Sends data back to an in-progress Ping.
  // Sends the data returned from the most recent PongData call back to an
  // in-progress Ping.
  // PongData must be called exactly once before each call of this method.
  virtual void Pong() = 0;
};

// Base class for implementations which simple use a pair of Data objects for
// all Pings and Pongs.
class StaticPingPonger : public PingPongerInterface {
 public:
  Data *PingData() override { return &ping_data_; }
  Data *PongData() override { return &pong_data_; }

 private:
  Data ping_data_, pong_data_;
};

// Implements ping-pong by sending the data over file descriptors.
class FDPingPonger : public StaticPingPonger {
 protected:
  // Subclasses must override and call Init.
  FDPingPonger() {}

  // Subclasses must call this in their constructor.
  // Does not take ownership of any of the file descriptors, any/all of which
  // may be the same.
  // {server,client}_read must be open for reading and {server,client}_write
  // must be open for writing.
  void Init(int server_read, int server_write, int client_read,
            int client_write) {
    server_read_ = server_read;
    server_write_ = server_write;
    client_read_ = client_read;
    client_write_ = client_write;
  }

 private:
  const Data *Ping() override {
    WriteFully(client_write_, *PingData());
    ReadFully(client_read_, &read_by_client_);
    return &read_by_client_;
  }

  const Data *Wait() override {
    ReadFully(server_read_, &read_by_server_);
    return &read_by_server_;
  }

  void Pong() override { WriteFully(server_write_, *PongData()); }

  void ReadFully(int fd, Data *data) {
    size_t remaining = sizeof(*data);
    uint8_t *current = &(*data)[0];
    while (remaining > 0) {
      const ssize_t result = AOS_PCHECK(read(fd, current, remaining));
      AOS_CHECK_LE(static_cast<size_t>(result), remaining);
      remaining -= result;
      current += result;
    }
  }

  void WriteFully(int fd, const Data &data) {
    size_t remaining = sizeof(data);
    const uint8_t *current = &data[0];
    while (remaining > 0) {
      const ssize_t result = AOS_PCHECK(write(fd, current, remaining));
      AOS_CHECK_LE(static_cast<size_t>(result), remaining);
      remaining -= result;
      current += result;
    }
  }

  Data read_by_client_, read_by_server_;
  int server_read_ = -1, server_write_ = -1, client_read_ = -1,
      client_write_ = -1;
};

class PipePingPonger : public FDPingPonger {
 public:
  PipePingPonger() {
    AOS_PCHECK(pipe(to_server));
    AOS_PCHECK(pipe(from_server));
    Init(to_server[0], from_server[1], from_server[0], to_server[1]);
  }
  ~PipePingPonger() {
    AOS_PCHECK(close(to_server[0]));
    AOS_PCHECK(close(to_server[1]));
    AOS_PCHECK(close(from_server[0]));
    AOS_PCHECK(close(from_server[1]));
  }

 private:
  int to_server[2], from_server[2];
};

class NamedPipePingPonger : public FDPingPonger {
 public:
  NamedPipePingPonger() {
    OpenFifo("/tmp/to_server", &client_write_, &server_read_);
    OpenFifo("/tmp/from_server", &server_write_, &client_read_);

    Init(server_read_, server_write_, client_read_, client_write_);
  }
  ~NamedPipePingPonger() {
    AOS_PCHECK(close(server_read_));
    AOS_PCHECK(close(client_write_));
    AOS_PCHECK(close(client_read_));
    AOS_PCHECK(close(server_write_));
  }

 private:
  void OpenFifo(const char *name, int *write, int *read) {
    {
      const int ret = unlink(name);
      if (ret == -1 && errno != ENOENT) {
        AOS_PLOG(FATAL, "unlink(%s)", name);
      }
      AOS_PCHECK(mkfifo(name, S_IWUSR | S_IRUSR));
      // Have to open it nonblocking because the other end isn't open yet...
      *read = AOS_PCHECK(open(name, O_RDONLY | O_NONBLOCK));
      *write = AOS_PCHECK(open(name, O_WRONLY));
      {
        const int flags = AOS_PCHECK(fcntl(*read, F_GETFL));
        AOS_PCHECK(fcntl(*read, F_SETFL, flags & ~O_NONBLOCK));
      }
    }
  }

  int server_read_, server_write_, client_read_, client_write_;
};

class UnixPingPonger : public FDPingPonger {
 public:
  UnixPingPonger(int type) {
    AOS_PCHECK(socketpair(AF_UNIX, type, 0, to_server));
    AOS_PCHECK(socketpair(AF_UNIX, type, 0, from_server));
    Init(to_server[0], from_server[1], from_server[0], to_server[1]);
  }
  ~UnixPingPonger() {
    AOS_PCHECK(close(to_server[0]));
    AOS_PCHECK(close(to_server[1]));
    AOS_PCHECK(close(from_server[0]));
    AOS_PCHECK(close(from_server[1]));
  }

 private:
  int to_server[2], from_server[2];
};

class TCPPingPonger : public FDPingPonger {
 public:
  TCPPingPonger(bool nodelay) {
    server_ = AOS_PCHECK(socket(AF_INET, SOCK_STREAM, 0));
    if (nodelay) {
      const int yes = 1;
      AOS_PCHECK(
          setsockopt(server_, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes)));
    }
    {
      sockaddr_in server_address;
      memset(&server_address, 0, sizeof(server_address));
      server_address.sin_family = AF_INET;
      server_address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
      AOS_PCHECK(bind(server_, reinterpret_cast<sockaddr *>(&server_address),
                      sizeof(server_address)));
    }
    AOS_PCHECK(listen(server_, 1));

    client_ = AOS_PCHECK(socket(AF_INET, SOCK_STREAM, 0));
    if (nodelay) {
      const int yes = 1;
      AOS_PCHECK(
          setsockopt(client_, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes)));
    }
    {
      sockaddr_in client_address;
      unsigned int length = sizeof(client_address);
      AOS_PCHECK(getsockname(
          server_, reinterpret_cast<sockaddr *>(&client_address), &length));
      AOS_PCHECK(connect(client_, reinterpret_cast<sockaddr *>(&client_address),
                         length));
    }
    server_connection_ = AOS_PCHECK(accept(server_, nullptr, 0));

    Init(server_connection_, server_connection_, client_, client_);
  }
  ~TCPPingPonger() {
    AOS_PCHECK(close(client_));
    AOS_PCHECK(close(server_connection_));
    AOS_PCHECK(close(server_));
  }

 private:
  int server_, client_, server_connection_;
};

class UDPPingPonger : public FDPingPonger {
 public:
  UDPPingPonger() {
    CreatePair(&server_read_, &client_write_);
    CreatePair(&client_read_, &server_write_);

    Init(server_read_, server_write_, client_read_, client_write_);
  }
  ~UDPPingPonger() {
    AOS_PCHECK(close(server_read_));
    AOS_PCHECK(close(client_write_));
    AOS_PCHECK(close(client_read_));
    AOS_PCHECK(close(server_write_));
  }

 private:
  void CreatePair(int *server, int *client) {
    *server = AOS_PCHECK(socket(AF_INET, SOCK_DGRAM, 0));
    {
      sockaddr_in server_address;
      memset(&server_address, 0, sizeof(server_address));
      server_address.sin_family = AF_INET;
      server_address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
      // server_address.sin_port = htons(server_ + 3000);
      AOS_PCHECK(bind(*server, reinterpret_cast<sockaddr *>(&server_address),
                      sizeof(server_address)));
    }
    *client = AOS_PCHECK(socket(AF_INET, SOCK_DGRAM, 0));
    {
      sockaddr_in client_address;
      unsigned int length = sizeof(client_address);
      AOS_PCHECK(getsockname(
          *server, reinterpret_cast<sockaddr *>(&client_address), &length));
      AOS_PCHECK(connect(*client, reinterpret_cast<sockaddr *>(&client_address),
                         length));
    }
  }

  int server_read_, server_write_, client_read_, client_write_;
};

// Implements ping-pong without copying the data using a condition variable-like
// interface.
class ConditionVariablePingPonger : public StaticPingPonger {
 protected:
  // Represents a condition variable bundled with a mutex.
  //
  // Wait may return spuriously.
  class ConditionVariableInterface {
   public:
    virtual ~ConditionVariableInterface() {}

    // Locks the mutex.
    virtual void Lock() = 0;

    // Unlocks the mutex.
    virtual void Unlock() = 0;

    // Waits on the condition variable.
    //
    // The mutex must be locked when this is called.
    virtual void Wait() = 0;

    // Signals the condition variable.
    //
    // The mutex does not have to be locked during this.
    virtual void Signal() = 0;
  };

  ConditionVariablePingPonger(
      ::std::unique_ptr<ConditionVariableInterface> ping,
      ::std::unique_ptr<ConditionVariableInterface> pong)
      : ping_(::std::move(ping)), pong_(::std::move(pong)) {}

 private:
  const Data *Ping() override {
    ping_->Lock();
    to_server_ = PingData();
    ping_->Unlock();
    ping_->Signal();
    pong_->Lock();
    while (from_server_ == nullptr) {
      pong_->Wait();
    }
    const Data *r = from_server_;
    from_server_ = nullptr;
    pong_->Unlock();
    return r;
  }

  const Data *Wait() override {
    ping_->Lock();
    while (to_server_ == nullptr) {
      ping_->Wait();
    }
    const Data *r = to_server_;
    to_server_ = nullptr;
    ping_->Unlock();
    return r;
  }

  void Pong() override {
    pong_->Lock();
    from_server_ = PongData();
    pong_->Unlock();
    pong_->Signal();
  }

  const Data *to_server_ = nullptr, *from_server_ = nullptr;
  const ::std::unique_ptr<ConditionVariableInterface> ping_, pong_;
};

// Implements ping-pong without copying the data using a semaphore-like
// interface.
class SemaphorePingPonger : public StaticPingPonger {
 protected:
  // Represents a semaphore, which need only count to 1.
  //
  // The behavior when calling Get/Put in anything other than alternating order
  // is undefined.
  //
  // Wait may NOT return spuriously.
  class SemaphoreInterface {
   public:
    virtual ~SemaphoreInterface() {}

    virtual void Get() = 0;
    virtual void Put() = 0;
  };

  SemaphorePingPonger(::std::unique_ptr<SemaphoreInterface> ping,
                      ::std::unique_ptr<SemaphoreInterface> pong)
      : ping_(::std::move(ping)), pong_(::std::move(pong)) {}

 private:
  const Data *Ping() override {
    to_server_ = PingData();
    ping_->Put();
    pong_->Get();
    return from_server_;
  }

  const Data *Wait() override {
    ping_->Get();
    return to_server_;
  }

  void Pong() override {
    from_server_ = PongData();
    pong_->Put();
  }

  const Data *to_server_ = nullptr, *from_server_ = nullptr;
  const ::std::unique_ptr<SemaphoreInterface> ping_, pong_;
};

class AOSMutexPingPonger : public ConditionVariablePingPonger {
 public:
  AOSMutexPingPonger()
      : ConditionVariablePingPonger(
            ::std::unique_ptr<ConditionVariableInterface>(
                new AOSConditionVariable()),
            ::std::unique_ptr<ConditionVariableInterface>(
                new AOSConditionVariable())) {}

 private:
  class AOSConditionVariable : public ConditionVariableInterface {
   public:
    AOSConditionVariable() : condition_(&mutex_) {}

   private:
    void Lock() override { AOS_CHECK(!mutex_.Lock()); }
    void Unlock() override { mutex_.Unlock(); }
    void Wait() override { AOS_CHECK(!condition_.Wait()); }
    void Signal() override { condition_.Broadcast(); }

    Mutex mutex_;
    Condition condition_;
  };
};

class AOSEventPingPonger : public SemaphorePingPonger {
 public:
  AOSEventPingPonger()
      : SemaphorePingPonger(
            ::std::unique_ptr<SemaphoreInterface>(new AOSEventSemaphore()),
            ::std::unique_ptr<SemaphoreInterface>(new AOSEventSemaphore())) {}

 private:
  class AOSEventSemaphore : public SemaphoreInterface {
   private:
    void Get() override {
      event_.Wait();
      event_.Clear();
    }
    void Put() override { event_.Set(); }

    Event event_;
  };
};

class PthreadMutexPingPonger : public ConditionVariablePingPonger {
 public:
  PthreadMutexPingPonger(int pshared, bool pi)
      : ConditionVariablePingPonger(
            ::std::unique_ptr<ConditionVariableInterface>(
                new PthreadConditionVariable(pshared, pi)),
            ::std::unique_ptr<ConditionVariableInterface>(
                new PthreadConditionVariable(pshared, pi))) {}

 private:
  class PthreadConditionVariable : public ConditionVariableInterface {
   public:
    PthreadConditionVariable(bool pshared, bool pi) {
      {
        pthread_condattr_t cond_attr;
        AOS_PRCHECK(pthread_condattr_init(&cond_attr));
        if (pshared) {
          AOS_PRCHECK(
              pthread_condattr_setpshared(&cond_attr, PTHREAD_PROCESS_SHARED));
        }
        AOS_PRCHECK(pthread_cond_init(&condition_, &cond_attr));
        AOS_PRCHECK(pthread_condattr_destroy(&cond_attr));
      }

      {
        pthread_mutexattr_t mutex_attr;
        AOS_PRCHECK(pthread_mutexattr_init(&mutex_attr));
        if (pshared) {
          AOS_PRCHECK(pthread_mutexattr_setpshared(&mutex_attr,
                                                   PTHREAD_PROCESS_SHARED));
        }
        if (pi) {
          AOS_PRCHECK(
              pthread_mutexattr_setprotocol(&mutex_attr, PTHREAD_PRIO_INHERIT));
        }
        AOS_PRCHECK(pthread_mutex_init(&mutex_, nullptr));
        AOS_PRCHECK(pthread_mutexattr_destroy(&mutex_attr));
      }
    }
    ~PthreadConditionVariable() {
      AOS_PRCHECK(pthread_mutex_destroy(&mutex_));
      AOS_PRCHECK(pthread_cond_destroy(&condition_));
    }

   private:
    void Lock() override { AOS_PRCHECK(pthread_mutex_lock(&mutex_)); }
    void Unlock() override { AOS_PRCHECK(pthread_mutex_unlock(&mutex_)); }
    void Wait() override {
      AOS_PRCHECK(pthread_cond_wait(&condition_, &mutex_));
    }
    void Signal() override { AOS_PRCHECK(pthread_cond_broadcast(&condition_)); }

    pthread_cond_t condition_;
    pthread_mutex_t mutex_;
  };
};

class EventFDPingPonger : public SemaphorePingPonger {
 public:
  EventFDPingPonger()
      : SemaphorePingPonger(
            ::std::unique_ptr<SemaphoreInterface>(new EventFDSemaphore()),
            ::std::unique_ptr<SemaphoreInterface>(new EventFDSemaphore())) {}

 private:
  class EventFDSemaphore : public SemaphoreInterface {
   public:
    EventFDSemaphore() : fd_(AOS_PCHECK(eventfd(0, 0))) {}
    ~EventFDSemaphore() { AOS_PCHECK(close(fd_)); }

   private:
    void Get() override {
      uint64_t value;
      if (read(fd_, &value, sizeof(value)) != sizeof(value)) {
        AOS_PLOG(FATAL, "reading from eventfd %d failed\n", fd_);
      }
      AOS_CHECK_EQ(1u, value);
    }
    void Put() override {
      uint64_t value = 1;
      if (write(fd_, &value, sizeof(value)) != sizeof(value)) {
        AOS_PLOG(FATAL, "writing to eventfd %d failed\n", fd_);
      }
    }

    const int fd_;
  };
};

class SysvSemaphorePingPonger : public SemaphorePingPonger {
 public:
  SysvSemaphorePingPonger()
      : SemaphorePingPonger(
            ::std::unique_ptr<SemaphoreInterface>(new SysvSemaphore()),
            ::std::unique_ptr<SemaphoreInterface>(new SysvSemaphore())) {}

 private:
  class SysvSemaphore : public SemaphoreInterface {
   public:
    SysvSemaphore() : sem_id_(AOS_PCHECK(semget(IPC_PRIVATE, 1, 0600))) {}

   private:
    void Get() override {
      struct sembuf op;
      op.sem_num = 0;
      op.sem_op = -1;
      op.sem_flg = 0;
      AOS_PCHECK(semop(sem_id_, &op, 1));
    }
    void Put() override {
      struct sembuf op;
      op.sem_num = 0;
      op.sem_op = 1;
      op.sem_flg = 0;
      AOS_PCHECK(semop(sem_id_, &op, 1));
    }

    const int sem_id_;
  };
};

class PosixSemaphorePingPonger : public SemaphorePingPonger {
 protected:
  PosixSemaphorePingPonger(sem_t *ping_sem, sem_t *pong_sem)
      : SemaphorePingPonger(
            ::std::unique_ptr<SemaphoreInterface>(new PosixSemaphore(ping_sem)),
            ::std::unique_ptr<SemaphoreInterface>(
                new PosixSemaphore(pong_sem))) {}

 private:
  class PosixSemaphore : public SemaphoreInterface {
   public:
    PosixSemaphore(sem_t *sem) : sem_(sem) {}

   private:
    void Get() override { AOS_PCHECK(sem_wait(sem_)); }
    void Put() override { AOS_PCHECK(sem_post(sem_)); }

    sem_t *const sem_;
  };
};

class SysvQueuePingPonger : public StaticPingPonger {
 public:
  SysvQueuePingPonger()
      : ping_(AOS_PCHECK(msgget(IPC_PRIVATE, 0600))),
        pong_(AOS_PCHECK(msgget(IPC_PRIVATE, 0600))) {}

  const Data *Ping() override {
    {
      Message to_send;
      memcpy(&to_send.data, PingData(), sizeof(Data));
      AOS_PCHECK(msgsnd(ping_, &to_send, sizeof(Data), 0));
    }
    {
      Message received;
      AOS_PCHECK(msgrcv(pong_, &received, sizeof(Data), 1, 0));
      memcpy(&pong_received_, &received.data, sizeof(Data));
    }
    return &pong_received_;
  }

  const Data *Wait() override {
    {
      Message received;
      AOS_PCHECK(msgrcv(ping_, &received, sizeof(Data), 1, 0));
      memcpy(&ping_received_, &received.data, sizeof(Data));
    }
    return &ping_received_;
  }

  virtual void Pong() override {
    Message to_send;
    memcpy(&to_send.data, PongData(), sizeof(Data));
    AOS_PCHECK(msgsnd(pong_, &to_send, sizeof(Data), 0));
  }

 private:
  struct Message {
    long mtype = 1;
    char data[sizeof(Data)];
  };

  Data ping_received_, pong_received_;

  const int ping_, pong_;
};

class PosixQueuePingPonger : public StaticPingPonger {
 public:
  PosixQueuePingPonger() : ping_(Open("/ping")), pong_(Open("/pong")) {}
  ~PosixQueuePingPonger() {
    AOS_PCHECK(mq_close(ping_));
    AOS_PCHECK(mq_close(pong_));
  }

  const Data *Ping() override {
    AOS_PCHECK(mq_send(ping_,
                       static_cast<char *>(static_cast<void *>(PingData())),
                       sizeof(Data), 1));
    AOS_PCHECK(mq_receive(
        pong_, static_cast<char *>(static_cast<void *>(&pong_received_)),
        sizeof(Data), nullptr));
    return &pong_received_;
  }

  const Data *Wait() override {
    AOS_PCHECK(mq_receive(
        ping_, static_cast<char *>(static_cast<void *>(&ping_received_)),
        sizeof(Data), nullptr));
    return &ping_received_;
  }

  virtual void Pong() override {
    AOS_PCHECK(mq_send(pong_,
                       static_cast<char *>(static_cast<void *>(PongData())),
                       sizeof(Data), 1));
  }

 private:
  mqd_t Open(const char *name) {
    if (mq_unlink(name) == -1 && errno != ENOENT) {
      AOS_PLOG(FATAL, "mq_unlink(%s) failed", name);
    }
    struct mq_attr attr;
    attr.mq_flags = 0;
    attr.mq_maxmsg = 1;
    attr.mq_msgsize = sizeof(Data);
    attr.mq_curmsgs = 0;
    const mqd_t r = mq_open(name, O_CREAT | O_RDWR | O_EXCL, 0600, &attr);
    if (r == reinterpret_cast<mqd_t>(-1)) {
      AOS_PLOG(FATAL, "mq_open(%s, O_CREAT | O_RDWR | O_EXCL) failed", name);
    }
    return r;
  }

  const mqd_t ping_, pong_;
  Data ping_received_, pong_received_;
};

class PosixUnnamedSemaphorePingPonger : public PosixSemaphorePingPonger {
 public:
  PosixUnnamedSemaphorePingPonger(int pshared)
      : PosixSemaphorePingPonger(&ping_sem_, &pong_sem_) {
    AOS_PCHECK(sem_init(&ping_sem_, pshared, 0));
    AOS_PCHECK(sem_init(&pong_sem_, pshared, 0));
  }
  ~PosixUnnamedSemaphorePingPonger() {
    AOS_PCHECK(sem_destroy(&ping_sem_));
    AOS_PCHECK(sem_destroy(&pong_sem_));
  }

 private:
  sem_t ping_sem_, pong_sem_;
};

class PosixNamedSemaphorePingPonger : public PosixSemaphorePingPonger {
 public:
  PosixNamedSemaphorePingPonger()
      : PosixSemaphorePingPonger(ping_sem_ = Open("/ping"),
                                 pong_sem_ = Open("/pong")) {}
  ~PosixNamedSemaphorePingPonger() {
    AOS_PCHECK(sem_close(ping_sem_));
    AOS_PCHECK(sem_close(pong_sem_));
  }

 private:
  sem_t *Open(const char *name) {
    if (sem_unlink(name) == -1 && errno != ENOENT) {
      AOS_PLOG(FATAL, "shm_unlink(%s) failed", name);
    }
    sem_t *const r = sem_open(name, O_CREAT | O_RDWR | O_EXCL, 0600, 0);
    if (r == SEM_FAILED) {
      AOS_PLOG(FATAL, "sem_open(%s, O_CREAT | O_RDWR | O_EXCL) failed", name);
    }
    return r;
  }

  sem_t *ping_sem_, *pong_sem_;
};

int Main() {
  ::std::unique_ptr<PingPongerInterface> ping_ponger;
  if (absl::GetFlag(FLAGS_method) == "pipe") {
    ping_ponger.reset(new PipePingPonger());
  } else if (absl::GetFlag(FLAGS_method) == "named_pipe") {
    ping_ponger.reset(new NamedPipePingPonger());
  } else if (absl::GetFlag(FLAGS_method) == "aos_mutex") {
    ping_ponger.reset(new AOSMutexPingPonger());
  } else if (absl::GetFlag(FLAGS_method) == "aos_event") {
    ping_ponger.reset(new AOSEventPingPonger());
  } else if (absl::GetFlag(FLAGS_method) == "pthread_mutex") {
    ping_ponger.reset(new PthreadMutexPingPonger(false, false));
  } else if (absl::GetFlag(FLAGS_method) == "pthread_mutex_pshared") {
    ping_ponger.reset(new PthreadMutexPingPonger(true, false));
  } else if (absl::GetFlag(FLAGS_method) == "pthread_mutex_pshared_pi") {
    ping_ponger.reset(new PthreadMutexPingPonger(true, true));
  } else if (absl::GetFlag(FLAGS_method) == "pthread_mutex_pi") {
    ping_ponger.reset(new PthreadMutexPingPonger(false, true));
  } else if (absl::GetFlag(FLAGS_method) == "eventfd") {
    ping_ponger.reset(new EventFDPingPonger());
  } else if (absl::GetFlag(FLAGS_method) == "sysv_semaphore") {
    ping_ponger.reset(new SysvSemaphorePingPonger());
  } else if (absl::GetFlag(FLAGS_method) == "sysv_queue") {
    ping_ponger.reset(new SysvQueuePingPonger());
  } else if (absl::GetFlag(FLAGS_method) == "posix_semaphore_unnamed_shared") {
    ping_ponger.reset(new PosixUnnamedSemaphorePingPonger(1));
  } else if (absl::GetFlag(FLAGS_method) ==
             "posix_semaphore_unnamed_unshared") {
    ping_ponger.reset(new PosixUnnamedSemaphorePingPonger(0));
  } else if (absl::GetFlag(FLAGS_method) == "posix_semaphore_named") {
    ping_ponger.reset(new PosixNamedSemaphorePingPonger());
  } else if (absl::GetFlag(FLAGS_method) == "posix_queue") {
    ping_ponger.reset(new PosixQueuePingPonger());
  } else if (absl::GetFlag(FLAGS_method) == "unix_stream") {
    ping_ponger.reset(new UnixPingPonger(SOCK_STREAM));
  } else if (absl::GetFlag(FLAGS_method) == "unix_datagram") {
    ping_ponger.reset(new UnixPingPonger(SOCK_DGRAM));
  } else if (absl::GetFlag(FLAGS_method) == "unix_seqpacket") {
    ping_ponger.reset(new UnixPingPonger(SOCK_SEQPACKET));
  } else if (absl::GetFlag(FLAGS_method) == "tcp") {
    ping_ponger.reset(new TCPPingPonger(false));
  } else if (absl::GetFlag(FLAGS_method) == "tcp_nodelay") {
    ping_ponger.reset(new TCPPingPonger(true));
  } else if (absl::GetFlag(FLAGS_method) == "udp") {
    ping_ponger.reset(new UDPPingPonger());
  } else {
    fprintf(stderr, "Unknown IPC method to test '%s'\n",
            absl::GetFlag(FLAGS_method).c_str());
    return 1;
  }

  ::std::atomic<bool> done{false};

  ::std::thread server([&ping_ponger, &done]() {
    if (absl::GetFlag(FLAGS_server_priority) > 0) {
      SetCurrentThreadRealtimePriority(absl::GetFlag(FLAGS_server_priority));
    }
    SetCurrentThreadAffinity(
        MakeCpusetFromCpus({absl::GetFlag(FLAGS_server_cpu)}));

    while (!done) {
      const PingPongerInterface::Data *data = ping_ponger->Wait();
      PingPongerInterface::Data *response = ping_ponger->PongData();
      for (size_t i = 0; i < sizeof(*data); ++i) {
        (*response)[i] = (*data)[i] + 1;
      }
      ping_ponger->Pong();
    }
  });

  if (absl::GetFlag(FLAGS_client_priority) > 0) {
    SetCurrentThreadRealtimePriority(absl::GetFlag(FLAGS_client_priority));
  }
  SetCurrentThreadAffinity(
      MakeCpusetFromCpus({absl::GetFlag(FLAGS_client_cpu)}));

  // Warm everything up.
  for (int i = 0; i < 1000; ++i) {
    PingPongerInterface::Data *warmup_data = ping_ponger->PingData();
    memset(*warmup_data, i % 255, sizeof(*warmup_data));
    ping_ponger->Ping();
  }

  const monotonic_clock::time_point start = monotonic_clock::now();

  for (int32_t i = 0; i < absl::GetFlag(FLAGS_messages); ++i) {
    PingPongerInterface::Data *to_send = ping_ponger->PingData();
    memset(*to_send, i % 123, sizeof(*to_send));
    const PingPongerInterface::Data *received = ping_ponger->Ping();
    for (size_t ii = 0; ii < sizeof(*received); ++ii) {
      AOS_CHECK_EQ(((i % 123) + 1) % 255, (*received)[ii]);
    }
  }

  const monotonic_clock::time_point end = monotonic_clock::now();

  // Try to make sure the server thread gets past its check of done so our
  // Ping() down below doesn't hang. Kind of lame, but doing better would
  // require complicating the interface to each implementation which isn't worth
  // it here.
  ::std::this_thread::sleep_for(::std::chrono::milliseconds(200));
  done = true;
  ping_ponger->PingData();
  ping_ponger->Ping();
  server.join();

  AOS_LOG(INFO, "Took %f seconds to send %" PRId32 " messages\n",
          ::aos::time::DurationInSeconds(end - start),
          absl::GetFlag(FLAGS_messages));
  const chrono::nanoseconds per_message =
      (end - start) / absl::GetFlag(FLAGS_messages);
  if (per_message >= chrono::seconds(1)) {
    AOS_LOG(INFO, "More than 1 second per message ?!?\n");
  } else {
    AOS_LOG(INFO, "That is %" PRId32 " nanoseconds per message\n",
            static_cast<int>(per_message.count()));
  }

  return 0;
}

}  // namespace aos

int main(int argc, char **argv) {
  absl::SetProgramUsageMessage(
      ::std::string("Compares various forms of IPC. Usage:\n") + argv[0] +
      " --method=METHOD\n"
      "METHOD can be one of the following:\n"
      "\tpipe\n"
      "\tnamed_pipe\n"
      "\taos_mutex\n"
      "\taos_event\n"
      "\tpthread_mutex\n"
      "\tpthread_mutex_pshared\n"
      "\tpthread_mutex_pshared_pi\n"
      "\tpthread_mutex_pi\n"
      "\teventfd\n"
      "\tsysv_semaphore\n"
      "\tsysv_queue\n"
      "\tposix_semaphore_unnamed_shared\n"
      "\tposix_semaphore_unnamed_unshared\n"
      "\tposix_semaphore_named\n"
      "\tposix_queue\n"
      "\tunix_stream\n"
      "\tunix_datagram\n"
      "\tunix_seqpacket\n"
      "\ttcp\n"
      "\ttcp_nodelay\n"
      "\tudp\n");
  aos::InitGoogle(&argc, &argv);

  return ::aos::Main();
}
