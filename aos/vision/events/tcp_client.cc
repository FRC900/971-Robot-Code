#include "aos/vision/events/tcp_client.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <strings.h>
#include <sys/socket.h>

#include <cstdio>
#include <cstdlib>

#include "aos/logging/logging.h"

namespace aos::events {

namespace {
int MakeSocketNonBlocking(int sfd) {
  int flags;

  AOS_PCHECK(flags = fcntl(sfd, F_GETFL, 0));

  flags |= O_NONBLOCK;
  AOS_PCHECK(fcntl(sfd, F_SETFL, flags));

  return 0;
}

int OpenClient(const std::string &hostname, int portno) {
  int sockfd;
  struct sockaddr_in serveraddr;
  struct hostent *server;
  /* socket: create the socket */
  AOS_PCHECK(sockfd = socket(AF_INET, SOCK_STREAM, 0));

  /* gethostbyname: get the server's DNS entry */
  server = gethostbyname(hostname.c_str());
  if (server == NULL) {
    fprintf(stderr, "ERROR, no such host as %s\n", hostname.c_str());
    exit(-1);
  }

  /* build the server's Internet address */
  bzero((char *)&serveraddr, sizeof(serveraddr));
  serveraddr.sin_family = AF_INET;
  bcopy((char *)server->h_addr, (char *)&serveraddr.sin_addr.s_addr,
        server->h_length);
  serveraddr.sin_port = htons(portno);

  /* connect: create a connection with the server */
  AOS_PCHECK(connect(sockfd, (const struct sockaddr *)&serveraddr,
                     sizeof(serveraddr)));
  AOS_PCHECK(MakeSocketNonBlocking(sockfd));

  return sockfd;
}
}  // namespace

TcpClient::TcpClient(const std::string &hostname, int portno)
    : EpollEvent(OpenClient(hostname, portno)) {}

}  // namespace aos::events
