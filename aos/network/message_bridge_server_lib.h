#ifndef AOS_NETWORK_MESSAGE_BRIDGE_SERVER_LIB_H_
#define AOS_NETWORK_MESSAGE_BRIDGE_SERVER_LIB_H_

#include <deque>

#include "absl/types/span.h"
#include "aos/events/logging/log_reader.h"
#include "aos/events/logging/logger_generated.h"
#include "aos/events/shm_event_loop.h"
#include "aos/network/connect_generated.h"
#include "aos/network/message_bridge_client_generated.h"
#include "aos/network/message_bridge_server_generated.h"
#include "aos/network/message_bridge_server_status.h"
#include "aos/network/remote_data_generated.h"
#include "aos/network/remote_message_generated.h"
#include "aos/network/sctp_server.h"
#include "aos/network/timestamp_channel.h"
#include "aos/network/timestamp_generated.h"
#include "glog/logging.h"

namespace aos {
namespace message_bridge {

// See message_bridge_protocol.h for more details about the protocol.

// Class to encapsulate all the state per channel.  This is the dispatcher for a
// new message from the event loop.
class ChannelState {
 public:
  ChannelState(const Channel *channel, int channel_index,
               std::unique_ptr<aos::RawFetcher> last_message_fetcher)
      : channel_index_(channel_index),
        channel_(channel),
        last_message_fetcher_(std::move(last_message_fetcher)) {}

  // Class to encapsulate all the state per client on a channel.  A client may
  // be subscribed to multiple channels.
  struct Peer {
    Peer(const Connection *new_connection, int new_node_index,
         ServerConnection *new_server_connection_statistics,
         MessageBridgeServerStatus *new_server_status, bool new_logged_remotely,
         aos::Sender<RemoteMessage> *new_timestamp_logger)
        : connection(new_connection),
          node_index(new_node_index),
          server_connection_statistics(new_server_connection_statistics),
          server_status(new_server_status),
          timestamp_logger(new_timestamp_logger),
          logged_remotely(new_logged_remotely) {}

    // Valid if != 0.
    sctp_assoc_t sac_assoc_id = 0;

    size_t stream = 0;
    const aos::Connection *connection;
    const int node_index;
    ServerConnection *server_connection_statistics;
    MessageBridgeServerStatus *server_status;
    aos::Sender<RemoteMessage> *timestamp_logger = nullptr;

    // If true, this message will be logged on a receiving node.  We need to
    // keep it around to log it locally if that fails.
    bool logged_remotely = false;
  };

  // Needs to be called when a node (might have) disconnected.
  // Returns the node index which [dis]connected, or -1 if it didn't match.
  // reconnected is a vector of associations which have already connected.
  // This will potentially grow to the number of associations as we find reconnects.
  int NodeDisconnected(sctp_assoc_t assoc_id);
  int NodeConnected(const Node *node, sctp_assoc_t assoc_id, int stream,
                    SctpServer *server, FixedAllocator *allocator,
                    aos::monotonic_clock::time_point monotonic_now,
                    std::vector<sctp_assoc_t> *reconnected);

  // Adds a new peer.
  void AddPeer(const Connection *connection, int node_index,
               ServerConnection *server_connection_statistics,
               MessageBridgeServerStatus *server_status, bool logged_remotely,
               aos::Sender<RemoteMessage> *timestamp_logger);

  // Returns true if this channel has the same name and type as the other
  // channel.
  bool Matches(const Channel *other_channel);

  // Sends the data in context using the provided server.
  void SendData(SctpServer *server, FixedAllocator *allocator,
                const Context &context);

  // Packs a context into a size prefixed message header for transmission.
  flatbuffers::FlatBufferBuilder PackContext(FixedAllocator *allocator,
                                             const Context &context);

  // Handles reception of delivery times.
  void HandleDelivery(sctp_assoc_t rcv_assoc_id, uint16_t ssn,
                      absl::Span<const uint8_t> data,
                      uint32_t partial_deliveries,
                      MessageBridgeServerStatus *server_status);

 private:
  const int channel_index_;
  const Channel *const channel_;

  std::vector<Peer> peers_;

  // A fetcher to use to send the last message when a node connects and is
  // reliable.
  std::unique_ptr<aos::RawFetcher> last_message_fetcher_;
};

// This encapsulates the state required to talk to *all* the clients from this
// node.  It handles the session and dispatches data to the ChannelState.
class MessageBridgeServer {
 public:
  MessageBridgeServer(aos::ShmEventLoop *event_loop, std::string config_sha256);

  ~MessageBridgeServer() { event_loop_->epoll()->DeleteFd(server_.fd()); }

 private:
  // Reads a message from the socket.  Could be a notification.
  void MessageReceived();

  // Called when the server connection succeeds.
  void NodeConnected(sctp_assoc_t assoc_id);
  // Called when the server connection disconnects.
  void NodeDisconnected(sctp_assoc_t assoc_id);

  // Called when data (either a connection request or delivery timestamps) is
  // received.
  void HandleData(const Message *message);

  // Increments the invalid connection count overall, and per node if we know
  // which node (ie, node is not nullptr).
  void MaybeIncrementInvalidConnectionCount(const Node *node);

  // The maximum number of channels we support on a single connection. We need
  // to configure the SCTP socket with this before any clients connect, so we
  // need an upper bound on the number of channels any of them will use.
  int max_channels() const {
    return event_loop_->configuration()->channels()->size();
  }

  // Event loop to schedule everything on.
  aos::ShmEventLoop *event_loop_;

  ChannelTimestampSender timestamp_loggers_;
  SctpServer server_;

  MessageBridgeServerStatus server_status_;

  // ChannelState to send timestamps over the network with.
  ChannelState *timestamp_state_ = nullptr;

  // List of channels.  The entries that aren't sent from this node are left
  // null.
  std::vector<std::unique_ptr<ChannelState>> channels_;

  const std::string config_sha256_;

  // List of assoc_id's that have been found already when connecting.  This is a
  // member variable so the memory is allocated in the constructor.
  std::vector<sctp_assoc_t> reconnected_;

  FixedAllocator allocator_;
};

}  // namespace message_bridge
}  // namespace aos

#endif  // AOS_NETWORK_MESSAGE_BRIDGE_SERVER_LIB_H_
