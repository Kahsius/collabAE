import pymysql
import os
from time import time
from cassandra.cluster import Cluster

cluster = Cluster()
session = cluster.connect("youtube_multiview")

def sql_from_array(short_name, dimData):
    sql = "INSERT INTO " + short_name + " (id, label, "
    for i in range(dimData-1) :
        sql += "var" + str(i) + ", "
    sql += "var" + str(dimData-1) + ") VALUES (?, ?, " # id & label
    for i in range(dimData-1) :
        sql += "?, " 
    sql +=  "?);"
    return sql

sizes = [12185772, 2000, 838, 12183626, 1000, 1024, 512, 1024, 64, 647, 64, 4096, 7168]

path = "data/dir_data/test/"
DIM_BATCH = 1
filenames = os.listdir(path)
filenames = [f for f in filenames if os.path.isfile(os.path.join(path, f))]

for count, data_file_name in enumerate(filenames) :

    print("Loading " + data_file_name)
    short_name = data_file_name.split('.')[0]
    dimData = sizes[count]
    print(dimData)
    empty_arr = []
    for i in range(dimData+2):
        empty_arr+=[None]

    if dimData < 10000 :
        counter = 0

        request = "BEGIN BATCH "
        for i in range(DIM_BATCH):
            request += sql_from_array(short_name, dimData)
        request += "APPLY BATCH"
        len_request = 0
        request = session.prepare(request)
        args = []

        for line in open(path + data_file_name):
            if line[0] == "#":
                # Copy the model array
                arr_sql = list(empty_arr)
                k = int(line.split("\t")[1].strip("\n"))
                arr_sql[0] = k
            elif len(line) > 1:
                line = line.split(" ", 1)
                # In case an instance with all zero features
                if len(line) == 1: 
                    continue
                label, features = line
                arr_sql[1] = int(label)
                features = features.split(" ")
                for e in features:
                    ind, val = e.split(":")
                    arr_sql[int(ind)+1] = float(val.strip("\n"))
                args.extend(arr_sql)
                len_request += 1
                counter += 1
                if(len_request == DIM_BATCH) :
                    session.execute(request.bind(args))
                    len_request = 0
                    args = []
                    print(str(count) + " | " + short_name + " " + str(counter))

        if len_request > 0 :
            request = "BEGIN BATCH "
            for i in range(len_request):
                request += sql_from_array(short_name, dimData)
            request += "APPLY BATCH"
            request = session.prepare(request)
            session.execute(request.bind(args))