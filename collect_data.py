import socket
import struct
import csv
import time
import os
import signal
from datetime import datetime

# Vehicle: Hirochi Suburst 1.8 CVT

BAC = 0.0
UPDATE_RATE = 0.1 # Seconds
# COURSE = 'urban' # Garage to garage
# COURSE = 'highway'
COURSE = 'track' # starting line
outdir = "C:\\.Users.trizz.Documents.ai.data"


# Dictionary of avaliable data
field = {
    'time': 0,
    'car name': 1,
    'flags' : 2,
    'gear' : 3,
    'player id' : 4,
    'airspeed' : 5, # m/s
    'rpm' : 6,
    'abs' : 7, # bar
    'throttle input' : 8, # Deg. C
    'brake input' : 9,
    'steering input' : 10, # bar
    'wheelspeed' : 11, # Deg. C
    'dash lights' : 12,
    'dash lights on' : 13,
    'throttle' : 14,
    'brake' : 15,
    'steering': 16,
    'tcs' : 19,
}

car_fields_to_record = [
    'abs',
    'tcs',
    'airspeed',
    'wheelspeed',
    'rpm',
    'throttle input',
    'throttle',
    'brake input',
    'brake',
    'steering input',
    'steering',
    'gear',
]

other_fields_to_record = [
    'time',
    'BAC'
]

def write_data(writer, data):
    writer.writerow(data)

# Create UDP socket.
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# Bind to BeamNG OutGauge.
sock.bind(('127.0.0.1', 4444))

def handler(signum, frame):
    sock.close()
    exit()
signal.signal(signal.SIGINT, handler)

# Create CSV writer and file
datetime_str = datetime.now().strftime("%Y%m%d_%H-%M-%S")
fname = os.path.join(*outdir.split('.'), "{}_{}_{}.csv".format(COURSE, datetime_str, BAC))
print(fname)
csv_file = open(fname, 'w')
csv_writer = csv.writer(csv_file, lineterminator = '\n')

csv_writer.writerow(car_fields_to_record + other_fields_to_record)

time.sleep(5)

tic = time.perf_counter()
while True:
    # Receive data.
    data = sock.recv(96)
    toc = time.perf_counter()
    
    if toc - tic >= UPDATE_RATE:
        tic = toc
        # Unpack the data.
        outsim_pack = struct.unpack('I4sH2c7f2I3f16s16si', data)
        write_data(csv_writer,
            [outsim_pack[field[n]] for n in car_fields_to_record] +
            [tic, BAC]
        )
        # print("RPM: ", str(outsim_pack[6]))
        # print("Time: ", str(outsim_pack[field['airspeed']]))

# Release the socket.
sock.close()