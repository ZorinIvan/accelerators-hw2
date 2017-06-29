CC=gcc
CFLAGS=-Wall -g
LDFLAGS=-libverbs
SOURCES=rpc_client.c rpc_server.c
OBJECTS=$(SOURCES:.c=.o)

all: rpc_client rpc_server

rpc_client: rpc_client.o
	$(CC) $^ $(LDFLAGS) -o $@

rpc_server: rpc_server.o
	$(CC) $^ $(LDFLAGS) -o $@

rpc_server.o: rpc_server.c
	$(CC) $(CFLAGS) -c -o $@ $<

rpc_client.o: rpc_client.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	/bin/rm -f *.o rpc_client rpc_server
