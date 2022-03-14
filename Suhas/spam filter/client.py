import socket
import logging
import pickle
from PIL import Image


class Client:

    def __init__(self, server_addr, server_port):

        self.server_addr = server_addr
        self.server_port = server_port
        self.end = '<end>'.encode()

        self.log_path = 'logs/client.log'
        self.time_format = '%Y-%m-%d %H:%M:%S'

        self.logger = logging.getLogger('client')
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s :: %(name)s :: %(levelname)s :: %(message)s', datefmt=self.time_format)
        file_handler = logging.FileHandler(self.log_path)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def __call__(self, inps):

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        s.bind(('localhost', 0))
        self.logger.info(f'Sending Array via port:{s.getsockname()[1]}')

        s.connect((self.server_addr, self.server_port))

        self.send_img(s, inps)
        self.logger.info('Array sent awaiting results')

        data = self.recv_data(s, 1024)
        self.logger.info('Recieved Response')

        s.close()

        return data

    def recv_data(self, conn, buffer_size):

        data = b''
        cnt = 0
        while True:

            buffer = conn.recv(buffer_size)
            if self.end in buffer:
                data += buffer[:buffer.find(self.end)]
                break
            data += buffer
            cnt += 1

        data = pickle.loads(data)

        return data

    # def send_arr(self, conn, arr):

    #     arr = pickle.dumps(arr)
    #     conn.sendall(arr)
    #     conn.send(self.end)

    def send_img(self, conn, img_path):

        img = Image.open(img_path)
        img = pickle.dumps(img)
        conn.sendall(img)
        conn.send(self.end)


if __name__ == '__main__':

    client = Client('127.0.1.1', 5000)
    data = client('adv_fgsm.png')
    print(data)
