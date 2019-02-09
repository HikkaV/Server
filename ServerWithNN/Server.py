import json
import socket
import keras, os
from PIL import Image
from skimage import transform
import Settings
import numpy as np
from threading import Thread
import tensorflow as tf
import logging
import datetime
import shutil


class Server(object):
    clients = {}
    buffer_size = 40960000
    basename = "image%s.png"
    addresses = {}
    logger = logging.getLogger("Server.Server")
    now = datetime.datetime.now()

    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((host, port))
        self.server_socket.listen(40)
        self.model = self.load_model()
        self.labels = self.init_label_dict()
        self.logger = logging.getLogger("Server.Server.add")

    def date_name(self):
        return ":" + str(self.now.year) + ":" + str(self.now.month) + ":" + str(self.now.day) + ":" + str(
            self.now.hour) + ":" + ":" + str(self.now.minute)

    def init_label_dict(self):
        json_data = open(Settings.path_to_labels).read()
        labels = json.loads(json_data)
        self.logger.info("Loading dict with labels for classes")
        return labels

    def save_locally(self, name):
        shutil.copyfile(self.abs_name, Settings.path_to_save_imgs + name)
        self.logger.info("Saving the img locally")

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
            self.load_img(command, client)
            self.get_prediction(client, command)
            flag = self.quit(command, client)
            if flag:
                break

    def quit(self, command, client):
        flag = False
        if command == 'quit':
            self.logger.info('The command is "quit" ')
            client.send(bytes("Sayonara^^", "utf8"))
            self.logger.info('Sending the farewell message to client')
            flag = True
        if flag:
            try:
                del self.clients[client]
            except KeyError:
                self.logger.info('The client %s:%s ' % self.addresses[client] + ' has left the server ')
                print('The client %s:%s ' % self.addresses[client] + ' has left the server ')
            self.logger.info('Deleting the client from the list of clients')
            client.close()

        return flag

    def load_img(self, command, client):

        if command == 'predict':
            self.logger.info('The command is "predict"')
            data = client.recv(self.buffer_size)
            if not data:
                self.logger.error('The trouble occurred while writing the last pieces of data into file')
                client.send(bytes("The trouble occurred, please resend the pic", "utf8"))
                raise Exception
            self.logger.info(
                'The program obtained all the data :' + str(data) + ' bytes')
            name = self.basename % self.date_name()
            self.abs_name = Settings.path_to_pic + name
            myfile = open(self.abs_name, 'wb')
            myfile.write(data)
            self.logger.info('The program saved data to ' + name)
            myfile.close()
            self.logger.info('The image was saved')
            client.send(bytes("Got image", "utf8"))
            self.save_locally(name)

    def get_prediction(self, client, command):
        if command == 'predict':
            if os.path.exists(self.abs_name):
                self.logger.info('Wait for prediction')
                client.send(bytes("Wait for prediction..", "utf8"))
                msg = self.predict(self.abs_name)
                self.logger.info('The prediction for ' + self.abs_name + ' :' + msg)
                client.send(bytes(msg, "utf8"))
                self.logger.info('Sending prediction to client')
                os.remove(self.abs_name)
            else:
                self.logger.error('The trouble occurred')
                client.send(bytes('The trouble occurred', 'utf8'))
                raise Exception

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
                np_image = Image.open(path)
                np_image = np.array(np_image).astype('float32') / 255
                np_image = transform.resize(np_image, (125, 125, 3))
                np_image = np.expand_dims(np_image, axis=0)
                tmp = self.model.predict(np_image)
                prediction = np.argmax(tmp, axis=1)
                prediction = self.labels[str(prediction[0])]
                return prediction
