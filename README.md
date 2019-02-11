# Server
link to trained NN : https://drive.google.com/open?id=1qyiD6lFpqCVpei87AzDEV8hW83cJ70iU
ubuntu ver : 18.10
there are two main comands which server can accept : predict and quit.
when the server is turned on it expects to get on of the above functions
if the command is predict the server is waiting for an image to load , after loading it sends a message 'Got image' to client so that user knows that no problems have occurred and copies the img to the folder named 'pics' so that it has the story of sended images. After that the NN is loaded into server and the prediction for the img is casted. Then the prediction is sended to the client and server continue waiting for the commands,
if the command is quit, the server sends farewall to client and deletes him from the connected users.
