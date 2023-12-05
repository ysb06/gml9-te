import pandas as pd

a: pd.DataFrame = pd.read_pickle("./resources/IMCRTS_Mini/imcrts_data.pickle")
a.to_hdf("./resources/IMCRTS_Mini/imcrts_data_sample.h5", key="imc")
