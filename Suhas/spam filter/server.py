import socket
import logging
import argparse
import pandas as pd
import os
from datetime import datetime
from datetime import timedelta
from pytesseract import image_to_string
import pickle


class Server:

    def __init__(self, port, limit):

        self.port = port
        self.limit = limit

        self.log_path = 'logs/server.log'
        self.req_path = 'logs/requests.csv'

        self.hostname = socket.gethostname()
        self.ipaddr = socket.gethostbyname(self.hostname)

        self.end = '<end>'.encode()
        self.time_format = '%Y-%m-%d %H:%M:%S'

        self.logger = logging.getLogger('server')
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s :: %(name)s :: %(levelname)s :: %(message)s', datefmt=self.time_format)
        file_handler = logging.FileHandler(self.log_path)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        if os.path.exists(self.req_path):
            self.df = pd.read_csv(self.req_path)

        else:
            self.df = pd.DataFrame(columns=['timestamp', 'ipaddr'])

        self.df.to_csv(self.req_path, index=False)
        self.df['timestamp'] = pd.to_datetime(
            self.df['timestamp'], format=self.time_format)

        self.logger.info(f'Server Initialized')
        print(f'Server Initialized')

    def __call__(self):

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.logger.info('Created Socket')

        s.bind((self.ipaddr, self.port))
        self.logger.info(f'Socket bind to port : {self.port}')

        s.listen()
        print(f'Server listening at: {self.ipaddr}:{self.port}')
        self.logger.info(f'Server listening at: {self.ipaddr}:{self.port}')

        try:
            while True:

                conn, addr = s.accept()
                dt = datetime.now()

                nreq = self.nrequests(dt, addr[0])

                dt = dt.strftime(self.time_format)

                self.df = self.df.append({'timestamp': dt,
                                          'ipaddr': f'{addr[0]}'}, ignore_index=True)
                self.df.to_csv(self.req_path, index=False)
                img = self.recv_img(conn, 1024)

                if nreq < self.limit:
                    status = 'pass'
                    # Compute with img
                    result = image_to_string(img, lang='eng')

                else:
                    status = 'fail'
                    result = 'Too many requests'
                self.send_result(conn, status, result)

        except KeyboardInterrupt as e:
            pass

        s.close()
        self.logger.info(f'Socket Closed')
        self.logger.info(f'Server stopped')
        print(f'Server stopped')

    # def recv_arr(self, conn, buffer_size):

    #     data = b''
    #     cnt = 0
    #     while True:

    #         buffer = conn.recv(buffer_size)
    #         if self.end in buffer:
    #             data += buffer[:buffer.find(self.end)]
    #             break
    #         data += buffer
    #         cnt += 1

    #     arr = pickle.loads(data)
    #     arr = torch.from_numpy(arr)

    #     return arr

    def recv_img(self, conn, buffer_size):

        data = b''
        cnt = 0
        while True:

            buffer = conn.recv(buffer_size)
            if self.end in buffer:
                data += buffer[:buffer.find(self.end)]
                break
            data += buffer
            cnt += 1

        img = pickle.loads(data)

        return img

    def nrequests(self, dt, ipaddr):

        deltime = dt - timedelta(hours=1)

        df2 = pd.to_datetime(
            self.df['timestamp'], format=self.time_format)

        self.df.drop(self.df[df2 <= deltime].index, inplace=True)

        return (self.df['ipaddr'] == ipaddr).sum()

    def send_result(self, conn, status, result=None):

        d = dict()
        d['status'] = status
        d['result'] = result

        if status not in ['pass', 'fail']:
            raise ValueError(f'No flag {status}')

        ds = pickle.dumps(d)
        conn.sendall(ds)
        conn.send(self.end)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="""python server.py [--port | -p] (int)""")
    parser.add_argument('--port', '-p', type=int, default=5000,
                        help='The port number on which the server runs')
    parser.add_argument('--limit', '-l', type=int, default=60,
                        help='The limit on number of requests')

    args = parser.parse_args()
    port = args.port
    limit = args.limit

    server = Server(port, limit)
    server()
