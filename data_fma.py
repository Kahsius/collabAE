import pandas

path = "data/fma/fma_metadata/"
features = pandas.read_csv(path + "features.csv")
echonest = pandas.read_csv(path + "echonest.csv")

print("{0} tracks described by {1} features".format(*features.shape))

columns = ['mfcc', 'chroma_cens', 'tonnetz', 'spectral_contrast']
columns.append(['spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff'])
columns.append(['rmse', 'zcr'])

for column in columns :
    data = features.filter(regex=column+"*")
    data.to_csv(column+".csv")