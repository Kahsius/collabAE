import os
from libCollabAE import get_dim_sparse_file
from cassandra.cluster import Cluster

cluster = Cluster()
session = cluster.connect("youtube_multiview")

path = "data/dir_data/test/"
filenames = os.listdir(path)
filenames = [f for f in filenames if os.path.isfile(os.path.join(path, f))]

TABLES = {}
sizes = [12185772, 2000, 838, 12183626, 1000, 1024, 512, 1024, 64, 647, 64, 4096, 7168]

for kn filename in enumerate(filenames) :
	print("Loading " + filename)
	short_name = filename.split('.')[0]
	dimData = sizes[k]
	print(dimData)
	if dimData < 10000 :
		TABLES[short_name] = "CREATE TABLE " + short_name + " ( id int PRIMARY KEY, label int ,"
		for i in range(dimData-1) :
			TABLES[short_name] += " var" + str(i) + " float,"
		TABLES[short_name] += " var" + str(dimData-1) + " float );"
		session.execute(TABLES[short_name])

connection.close()
print("Tables created")