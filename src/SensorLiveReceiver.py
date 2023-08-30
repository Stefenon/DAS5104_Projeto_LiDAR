from struct import unpack
from math import cos, sin, pi
from multiprocessing import Process, Queue
from socketserver import UDPServer, DatagramRequestHandler

from src.Constants import Constants
from src.SensorManager import SensorManager


class SensorLiveReceiver():

    def __init__(self):
        self.sensor_front = SensorManager(Constants.SENSOR_IP_FRONT, Constants.SERVER_IP, Constants.SERVER_PORT)
        self.sensor_right = SensorManager(Constants.SENSOR_IP_RIGHT, Constants.SERVER_IP, Constants.SERVER_PORT)
        self.sensor_left = SensorManager(Constants.SENSOR_IP_LEFT, Constants.SERVER_IP, Constants.SERVER_PORT)
        self.sensor_top = SensorManager(Constants.SENSOR_IP_TOP, Constants.SERVER_IP, Constants.SERVER_PORT)

        self.server_port = Constants.SERVER_PORT
        self.server_ip = Constants.SERVER_IP
        self.queue = Queue()

    def start(self):
        self.process = Process(target=self.run, args=(self.queue, self.server_ip, self.server_port,), daemon=True)
        self.process.start()

        # print(self.sensor_front.set_parameters(samples_per_scan=600, scan_frequency=40))
        # print(self.sensor_right.set_parameters(samples_per_scan=600, scan_frequency=40))
        # print(self.sensor_left.set_parameters(samples_per_scan=600, scan_frequency=40))
        # print(self.sensor_top.set_parameters(samples_per_scan=600, scan_frequency=40))

        # self.sensor_front.request_handle_udp(max_num_points_scan=600, skip_scans=35)
        # self.sensor_right.request_handle_udp(max_num_points_scan=600, skip_scans=35)
        # self.sensor_left.request_handle_udp(max_num_points_scan=600, skip_scans=35)
        # self.sensor_top.request_handle_udp(max_num_points_scan=600, skip_scans=35)

        # self.sensor_front.start_scanoutput()
        # self.sensor_right.start_scanoutput()
        # self.sensor_left.start_scanoutput()
        # self.sensor_top.start_scanoutput()

    def stop(self):
        # self.sensor_front.stop_scanoutput()
        # self.sensor_right.stop_scanoutput()
        # self.sensor_left.stop_scanoutput()
        # self.sensor_top.stop_scanoutput()

        # self.sensor_front.release_handle()
        # self.sensor_right.release_handle()
        # self.sensor_left.release_handle()
        # self.sensor_top.release_handle()

        self.process.terminate()

    @staticmethod
    def run(queue: Queue, server_ip: str, server_port: int):
        server_udp = UDPServer((server_ip, server_port), Handler)
        server_udp.queue = queue
        server_udp.serve_forever()


class Handler(DatagramRequestHandler):

    def handle(self):
        data = self.rfile.read()

        if len(data) <= 10:
            print("Staring...")
            return

        # magic = unpack("H", data[:2])[0]
        # packet_type = unpack("H", data[2:4])[0]
        packet_size = unpack("I", data[4:8])[0]
        header_size = unpack("H", data[8:10])[0]
        # scan_number = unpack("H", data[10:12])[0]
        # packet_number = unpack("H", data[12:14])[0]
        # timestamp_raw = ...
        # timestamp_sync = ...
        # status_flags = unpack("I", data[30:34])[0]
        # scan_frequency = unpack("I", data[34:38])[0]
        # num_points_scan = unpack("H", data[38:40])[0]
        # num_points_packet = unpack("H", data[40:42])[0]
        # first_index = unpack("H", data[42:44])[0]
        first_angle = unpack("i", data[44:48])[0]
        angular_increment = unpack("i", data[48:52])[0]

        # print(f"magic: {hex(magic)}")
        # print(f"packet_type: {hex(packet_type)}")
        # print(f"packet_size: {packet_size}")
        # print(f"header_size: {header_size}")
        # print(f"scan_number: {scan_number}")
        # print(f"packet_number: {packet_number}")
        # print(f"status_flags: {status_flags}")
        # print(f"scan_frequency: {scan_frequency}")
        # print(f"num_points_scan: {num_points_scan}")
        # print(f"num_points_packet: {num_points_packet}")
        # print(f"first_index: {first_index}")
        # print(f"first_angle: {first_angle}")
        # print(f"angular_increment: {angular_increment}")
        # print("---------------------------------------")

        if len(data) != packet_size:
            print("corrupted package...")
            return

        payload = data[header_size:]  # list[uint32] - 4byte
        distances = unpack(f"{len(payload) // 4}I", payload[:len(payload) // 4 * 4])

        self.server.queue.put({
            "address": self.client_address[0],
            "xy": self.polar_to_xy(distances, first_angle, angular_increment),
        })

    def polar_to_xy(self, distances: list, first_angle: int, angular_increment: int) -> list[tuple[float, float]]:
        first_angle /= 10000
        angular_increment /= 10000

        xy = list()

        for i, distance in enumerate(distances):
            angle = (first_angle + i * angular_increment) * pi / 180.0

            x = round(distance * cos(angle))
            y = round(distance * sin(angle))

            xy.append((x, y))

        return xy