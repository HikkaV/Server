import json
import socket
import keras, os
from keras.preprocessing.image import ImageDataGenerator
import Settings
import numpy as np
from threading import Thread
import tensorflow as tf
import logging


class Server(object):
    clients = {}
    buffer_size = 4096
    basename = "image%s.png"
    addresses = {}
    logger = logging.getLogger("Server.Server")
    i = 0


    def __init__(self, host, port):
        self.datagen = ImageDataGenerator(rescale=1. / 255)
        self.test_generator = self.datagen.flow_from_directory(
            directory=Settings.path_for_predict,
            target_size=(125, 125),
            color_mode="rgb",
            batch_size=1,
            class_mode=None,
            shuffle=False,
            seed=1
        )
        self.host = host
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((host, port))
        self.server_socket.listen(40)
        self.model = self.load_model()
        self.labels = self.init_label_dict()
        self.logger = logging.getLogger("Server.Server.add")

    def init_label_dict(self):
        json_data = open(Settings.path_to_labels).read()
        labels = json.loads(json_data)
        return labels

    def accept_incoming_connections(self):
        """Sets up handling for incoming clients."""
        while True:
            client, client_address = self.server_socket.accept()
            self.logger.info("%s:%s has connected." % client_address)
            print("%s:%s has connected." % client_address)
            self.addresses[client] = client_address
            Thread(target=self.handle_client, args=(client,)).start()

    def handle_client(self, client):  # Takes client socket as argument.
        """Handles a single client connection."""

        while True:
            command = client.recv(self.buffer_size).decode("utf8")
            self.logger.info('The command was received')
            if command == 'predict':
                self.logger.info('The command is "predict"')
                data = client.recv(4096)
                self.logger.info(
                    'The program obtained a minimal cluster of data :' + str(Server.buffer_size) + ' bytes')
                name = self.basename % Server.i
                abs_name = "/home/hikkav/PycharmProjects/ServerWithNN/pics/" + name
                myfile = open(abs_name, 'wb')
                myfile.write(data)
                self.logger.info('The program saved data to ' + name)
                data = client.recv(40960000)
                if not data:
                    self.logger.error('The trouble occurred while writing the last pieces of data into file')
                    client.send(bytes("The trouble occurred, please resend the pic", "utf8"))
                    raise Exception
                myfile.write(data)
                myfile.close()
                self.logger.info('The image was saved')
                client.send(bytes("GOT IMAGE", "utf8"))
                tmp = client.recv(4096).decode("utf8")
                if tmp == "let's go" and os.path.exists(abs_name):
                    self.logger.info('Wait for prediction')
                    client.send(bytes("Wait for prediction..", "utf8"))
                    msg = self.predict(abs_name)
                    self.logger.info('The prediction for ' + name + ' :' + msg)
                    client.send(bytes(msg, "utf8"))
                    self.logger.info('Sending prediction to client')
                else:
                    self.logger.error('The trouble occurred')
                    raise Exception
            elif command == 'quit':
                self.logger.info('The command is "quit" ')
                client.send(bytes("Sayonara^^", "utf8"))
                self.logger.info('Sending the farewell message to client')
                try:
                    del self.clients[client]
                except KeyError:
                    self.logger.info('The client %s:%s ' % self.addresses[client] + ' has left the server ')
                    print('The client %s:%s ' % self.addresses[client] + ' has left the server ')
                self.logger.info('Deleting the client from the list of clients')
                client.close()
                break

    def load_model(self):
        """load model if it exists"""

        if os.path.exists(Settings.path_to_model):
            model = keras.models.load_model(Settings.path_to_model)

            self.graph = tf.get_default_graph()
            return model
        else:
            self.logger.info('Model with such a path ' + Settings.path_to_model + ' doesn\'t exist')
            raise Exception

    def predict(self, path):
        self.logger.info('Making a prediction')
        with self.graph.as_default():
            if os.path.exists(path):
                tmp = self.model.predict_generator(self.test_generator, verbose=1)
                label = np.argmax(tmp, axis=1)
                prediction = self.labels[str(label[0])]
                return prediction
