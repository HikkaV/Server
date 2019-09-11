import json
import socket

import cv2
import keras, os
from PIL import Image
from skimage import transform
import settings
import numpy as np
from threading import Thread
import tensorflow as tf
import logging
import datetime
import shutil
import sys
import time


class Server(object):
    clients = {}
    buffer_size = 40960000
    basename = ".png"
    addresses = {}
    logger = logging.getLogger("Server.Server")
    now = datetime.datetime.now()

    def __init__(self, port, labels=None):
        self.host = ''
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('', port))
        self.server_socket.listen(settings.max_users)
        self.model = self.load_model()
        self.tf_model = self.read_tensorflow()
        self.labels = self.init_label_dict() if not labels else labels
        self.logger = logging.getLogger("Server.Server.add")
        sess = tf.Session()
        sess.graph.as_default()
        tf.import_graph_def(self.tf_model, name='')
        self.sess = sess
        self.CONF_THRESHOLD = 0.7
        self.class_list = self.get_classes()

    def get_classes(self):
        class_list = open('facepeople.names').read().strip().split('\n')
        return class_list

    def read_tensorflow(self):
        with tf.gfile.FastGFile('face_person.pb', 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        return graph_def

    def tf_detect(self, img, classes):

        faces_box = []
        people_box = []
        rows = img.shape[0]
        cols = img.shape[1]
        inp = cv2.resize(img, (300, 300))
        inp = inp[:, :, [2, 1, 0]]
        out = self.sess.run([self.sess.graph.get_tensor_by_name('num_detections:0'),
                             self.sess.graph.get_tensor_by_name('detection_scores:0'),
                             self.sess.graph.get_tensor_by_name('detection_boxes:0'),
                             self.sess.graph.get_tensor_by_name('detection_classes:0')],
                            feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

        # Visualize detected bounding boxes.
        num_detections = int(out[0][0])
        for i in range(num_detections):
            classId = int(out[3][0][i])
            score = float(out[1][0][i])
            bbox = [float(v) for v in out[2][0][i]]

            if score > self.CONF_THRESHOLD:

                x = int(bbox[1] * cols)
                y = int(bbox[0] * rows)
                right = int(bbox[3] * cols)
                bottom = int(bbox[2] * rows)
                boxes = [x, y, right, bottom]
                pred = classes[classId - 1]
                if pred == 'face':
                    faces_box.append(boxes)
                else:
                    people_box.append(boxes)

        return faces_box

    def crop_face(self, img):
        self.faces = self.tf_detect(img, classes=self.class_list)
        print(len(self.faces))
        crop_faces =[(self.faces[i][0], self.faces[i][1], self.faces[i][2],
                           self.faces[i][3]) for i in
                          range(len(self.faces))]

        img_ = img[crop_faces[0][1]:crop_faces[0][3], crop_faces[0][0]:crop_faces[0][2], :]
        return img_

    def date_name(self):
        """
        makes a new name for a pic based on current date
        :return:
        """
        return ":" + str(self.now.year) + ":" + str(self.now.month) + ":" + str(self.now.day) + ":" + str(
            self.now.hour) + ":" + str(self.now.minute)

    def init_label_dict(self):
        """

        loads dict with labels for prediction
        """
        json_data = open(settings.path_to_labels).read()
        labels = json.loads(json_data)
        self.logger.info("Loading dict with labels for classes")
        return labels

    def save_locally(self, name, img):
        """
        saves img to folder named pics
        :param name: the name of a saved img

        """
        cv2.imwrite('pics/' + name, img)
        print(name)
        self.logger.info("Saving the img locally")

    def accept_incoming_connections(self):
        """Sets up handling for incoming clients. """
        while True:
            client, client_address = self.server_socket.accept()
            self.logger.info("%s:%s has connected." % client_address)
            print("%s:%s has connected." % client_address)
            self.addresses[client] = client_address
            global thread
            thread = Thread(target=self.handle_client, args=(client,))
            thread.start()

    def handle_client(self, client):  # Takes client socket as argument.
        """
        Handles a single client connection.
        the func loads img from client using func load_img, then it sends a prediction
        about it , makes the copy of an img to the folder named pics and deletes it from the folder
        named ServerWithNN , so that the user can send another img and got a prediction exactly related to it

        """

        while True:

            command = client.recv(self.buffer_size).decode("utf8")
            self.logger.info('The command was received')

            abs_name, name = self.load_img(command, client)
            self.get_prediction(client, command, abs_name, name)

            if self.quit(client):
                break

    def quit(self, client):
        """
        the func deletes the client from the server if the command equals  'quit'
        :param client: the particular client

        """

        flag = True
        self.logger.info('Quitting ')

        try:
            del self.clients[client]
        except KeyError:
            self.logger.info('The client %s:%s ' % self.addresses[client] + ' has left the server ')
            print('The client %s:%s ' % self.addresses[client] + ' has left the server ')
            self.logger.info('Deleting the client from the list of clients')
            client.close()

        return flag

    def load_img(self, command, client):
        """
        loads an img to the server if the command equals 'predict'
       :param command: the command sent by a client
       :param client: the particular client

        """
        if command == 'predict':
            time.sleep(1)
            self.logger.info('Ready to load an img')
            data = client.recv(self.buffer_size)
            print(sys.getsizeof(data))
            if not data:
                self.logger.error('The trouble occurred while writing the last pieces of data into file')
                client.send(bytes("The trouble occurred, please resend the pic", "utf8"))
                raise Exception

            name = self.date_name() + '_' + str(self.addresses[client]) +str(time.localtime().tm_sec)+self.basename
            abs_name = settings.path_to_pic + name
            myfile = open(abs_name, 'wb')
            myfile.write(data)
            flag = True
            while flag:
                time.sleep(3)
                try:
                    img = np.array(Image.open(abs_name)).astype('float32') / 255
                    self.logger.info('that"s works!')
                    flag = False
                    self.logger.info('leaving from infinite loop')
                except Exception as e:

                    data = client.recv(self.buffer_size)
                    print(sys.getsizeof(data))
                    self.logger.info('waiting for additional data')
                    myfile.write(data)
                    print(e)
                    self.logger.info('going to the start')
                    if not data:
                        break
            self.logger.info(
                'The program obtained all the data :' + str(sys.getsizeof(data)) + ' bytes')
            self.logger.info('The program saved data to ' + name)
            myfile.close()
            self.logger.info('The image was saved')
            client.send(bytes("Got ", "utf8"))
            return abs_name, name

    def get_prediction(self, client, command, abs_name, name):
        """
        gets a prediction for an img to the server if the command equals 'predict'
       :param command: the command sent by a client
       :param client: the particular client

        """

        if command == 'predict':
            time.sleep(2)
            if os.path.exists(abs_name):
                msg = self.predict(abs_name, name)
                self.logger.info('The prediction for ' + abs_name + ' :' + msg)
                client.send(bytes(msg, "utf8"))
                self.logger.info('Sending prediction to client')
                os.remove(abs_name)
            else:
                self.logger.error('The trouble occurred')
                client.send(bytes('The trouble occurred', 'utf8'))
                raise Exception

    def load_model(self):
        """load model if it exists"""

        if os.path.exists(settings.path_to_model):
            model = keras.models.load_model(settings.path_to_model)

            self.graph = tf.get_default_graph()
            return model
        else:
            self.logger.info('Model with such a path ' + settings.path_to_model + ' doesn\'t exist')
            return None

    def predict(self, path, name):
        if self.model is None:
            msg = "As the model doesn't exist , the prediction is impossible"
            self.logger.info(msg)
            return msg
        self.logger.info('Making a prediction')
        with self.graph.as_default():
            if os.path.exists(path):
                np_image = cv2.imread(path)
                np_image = self.crop_face(np_image)
                self.save_locally(name, img=np_image)
                np_image = np.array(np_image).astype('float32') / 255
                np_image = transform.resize(np_image, (128, 128, 3))
                np_image = np.expand_dims(np_image, axis=0)
                tmp = self.model.predict(np_image)
                prediction = np.argmax(tmp, axis=1)
                prediction = self.labels[prediction[0]]
                return prediction
