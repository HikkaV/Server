from threading import Thread
import logging
from Server import Server
from Settings import *
if __name__ == '__main__':
    logger = logging.getLogger("Server")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler("Server.log")
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(threadName)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    host = host_
    port = port_
    server = Server(host,port)
    logger.info('The program started')
    logger.info('Waiting for connection')
    print("Waiting for connection...")
    ACCEPT_THREAD = Thread(target=server.accept_incoming_connections)
    ACCEPT_THREAD.start()  # Starts the infinite loop.
    ACCEPT_THREAD.join()

   
   