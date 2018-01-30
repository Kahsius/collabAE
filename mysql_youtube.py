import pymysql
import os
from libCollabAE import get_dim_sparse_file

DB_NAME = 'youtube_multiview'

path = "data/dir_data/test/"
filenames = os.listdir(path)
filenames = [f for f in filenames if os.path.isfile(os.path.join(path, f))]

TABLES = {}

for filename in filenames :
	print("Loading " + filename)
	short_name = filename.split('.')[0]
	dimData = get_dim_sparse_file(path + filename)
	if False :
		TABLES[short_name] = "CREATE TABLE `" + short_name + "` ( `id` int(11) NOT NULL, "
		for i in range(dimData) :
			TABLES[short_name] += " `var" + str(i) + "`,"
		TABLES[short_name] += " PRIMARY KEY (`id`));"
		print(short_name)
		print(TABLES[short_name])
		break

print("Datasets saved")