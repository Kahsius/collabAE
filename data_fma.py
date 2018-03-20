from sklearn.preprocessing import scale
import pandas
import numpy as np

path = "data/fma/fma_metadata/"
features = pandas.read_csv(path + "features.csv", skiprows=[2,3])
# echonest = pandas.read_csv(path + "echonest.csv")

# print("{0} tracks described by {1} features".format(*features.shape))

columns = ['mfcc', 'chroma_cens', 'tonnetz', 'spectral_contrast',
'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff','rmse', 'zcr']

labels = pandas.read_csv(path + "tracks.csv", skiprows=[0,2]).genre_top
labels = labels.fillna("Unknown")
# genres = labels.unique()
# labels_index = pandas.DataFrame([genres.index(x) for x in labels])

keep = labels != "Unknown"
labels = labels.where(keep.values).dropna()


for column in columns :
    print("Column : " + column)
    # On récupère le type de feature "column"
    data = features.filter(regex=column+"*")
    keep_tmp = pandas.concat([keep]*data.shape[1], ignore_index=True, axis=1)
    # On renomme les colonnes et on enlève la première ligne (les anciens noms)
    data = data.rename(columns=data.loc[0,:]).loc[1:,:]
    data = data.where(keep_tmp.values).dropna()
    data = data.iloc[:,1:]
    columns_names = data.axes[1]
    data = pandas.DataFrame(data=scale(data), columns=columns_names)

    data.to_csv("data/fma/"+column+".csv", index=False)
    print("\tdata/fma/"+column+".csv")
    n = data.shape[0]
    data[:int(n*.9)].to_csv("data/fma/"+column+"_train.csv", index=False)
    print("\tdata/fma/"+column+"_train.csv")
    data[int(n*.9):].to_csv("data/fma/"+column+"_test.csv", index=False)
    print("\tdata/fma/"+column+"_test.csv")

labels[:int(n*.9)].to_csv("data/fma/labels_train.csv", index=False)
labels[int(n*.9):].to_csv("data/fma/labels_test.csv", index=False)