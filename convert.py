import numpy as np
from scipy.sparse import coo_matrix
from scipy.io import mmwrite

def convert_to_matrix_market(input_file, output_file):
    data = np.loadtxt(input_file, dtype=int, comments='#')
    data = data[data[:,0].argsort()]
    rows = data[:, 0]
    cols = data[:, 1]
    values = np.ones(len(rows), dtype=int)
    matrix = coo_matrix((values, (rows, cols)))
    mmwrite(output_file, matrix)
# 'com-orkut', 'roadNet-CA', 'web-Stanford', 'soc-LiveJournal1', 
file_name = ["web-Google"]
for i in file_name:
    convert_to_matrix_market(i + '.txt', i + '.mtx')
