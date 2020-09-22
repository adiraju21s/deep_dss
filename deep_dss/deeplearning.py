# import numpy as np
# import healpy as hp
# from scipy import sparse
#
# import tensorflow as tf
# import tensorflow.keras as keras
# from tensorflow.keras import layers
#
#
# def nside2indexes(nsides, order):
#     """Return list of indexes from nside given a specific order.
#     This function return the necessary indexes for a deepsphere when
#     only a part of the sphere is considered.
#     Parameters
#     ----------
#     nsides : list of nside for the desired scale
#     order  : parameter specifying the size of the sphere part
#     """
#     nsample = 12 * order ** 2
#     indexes = [np.arange(hp.nside2npix(nside) // nsample) for nside in nsides]
#     return indexes
#
#
# def build_matrix_4_neighboors(nside, indexes, nest=True, dtype=np.float32):
#     assert nest
#
#     order = nside // hp.npix2nside(12 * (max(indexes) + 1))
#
#     npix = hp.nside2npix(nside) // hp.nside2npix(order)
#     new_indexes = list(range(npix))
#     assert (set(indexes) == set(new_indexes))
#
#     x, y, z = hp.pix2vec(nside, indexes, nest=True)
#     coords = np.vstack([x, y, z]).transpose()
#     coords = np.array(coords)
#
#     def all_or(d3, v):
#         v = np.array(v)
#         for d in d3:
#             if not (v == d).any():
#                 return False
#         return True
#
#     row_index = []
#     col_index = []
#     for index in indexes:
#         # A) Start with the initial square
#         d = index % 4
#         base = index - d
#         # 1) Add the next pixel
#         row_index.append(index)
#         if d == 0:
#             col_index.append(base + 1)
#         elif d == 1:
#             col_index.append(base + 3)
#         elif d == 2:
#             col_index.append(base)
#         elif d == 3:
#             col_index.append(base + 2)
#         else:
#             raise ValueError('Error in the code')
#         # 2) Add the previous pixel
#         row_index.append(index)
#         if d == 0:
#             col_index.append(base + 2)
#         elif d == 1:
#             col_index.append(base)
#         elif d == 2:
#             col_index.append(base + 3)
#         elif d == 3:
#             col_index.append(base + 1)
#         else:
#             raise ValueError('Error in the code')
#
#         # B) Connect the squares together...
#         for it in range(int(np.log2(nside) - np.log2(order) - 1)):
#
#             d2 = (index // (4 ** (it + 1))) % 4
#             d3 = [d]
#             for it2 in range(it):
#                 d3.append((index // (4 ** (it2 + 1)) % 4))
#             d3 = np.array(d3)
#             shift_o = []
#             for it2 in range(it + 1):
#                 shift_o.append(4 ** it2)
#             shift = 4 ** (it + 1) - sum(shift_o)
#             if d2 == 0:
#                 if all_or(d3, [1, 3]):
#                     row_index.append(index)
#                     col_index.append(index + shift)
#                 if all_or(d3, [2, 3]):
#                     row_index.append(index)
#                     col_index.append(index + 2 * shift)
#             elif d2 == 1:
#                 if all_or(d3, [0, 2]):
#                     row_index.append(index)
#                     col_index.append(index - shift)
#                 if all_or(d3, [2, 3]):
#                     row_index.append(index)
#                     col_index.append(index + 2 * shift)
#             elif d2 == 2:
#                 if all_or(d3, [0, 1]):
#                     row_index.append(index)
#                     col_index.append(index - 2 * shift)
#                 if all_or(d3, [1, 3]):
#                     row_index.append(index)
#                     col_index.append(index + shift)
#             elif d2 == 3:
#                 if all_or(d3, [0, 1]):
#                     row_index.append(index)
#                     col_index.append(index - 2 * shift)
#                 if all_or(d3, [0, 2]):
#                     row_index.append(index)
#                     col_index.append(index - shift)
#             else:
#                 raise ValueError('Error in the code')
#
#     # Compute Euclidean distances between neighbors.
#     distances = np.sum((coords[row_index] - coords[col_index]) ** 2, axis=1)
#     # slower: np.linalg.norm(coords[row_index] - coords[col_index], axis=1)**2
#
#     # Compute similarities / edge weights.
#     kernel_width = np.mean(distances)
#     weights = np.exp(-distances / (3 * kernel_width))
#
#     # Build the sparse matrix.
#     W = sparse.csr_matrix(
#         (weights, (row_index, col_index)), shape=(npix, npix), dtype=dtype)
#
#     return W
#
#
# def healpix_weightmatrix(nside=16, nest=True, indexes=None, dtype=np.float32):
#     """Return an unnormalized weight matrix for a graph using the HEALPIX sampling.
#     Parameters
#     ----------
#     nside : int
#         The healpix nside parameter, must be a power of 2, less than 2**30.
#     nest : bool, optional
#         if True, assume NESTED pixel ordering, otherwise, RING pixel ordering
#     indexes : list of int, optional
#         List of indexes to use. This allows to build the graph from a part of
#         the sphere only. If None, the default, the whole sphere is used.
#     dtype : data-type, optional
#         The desired data type of the weight matrix.
#     """
#     if not nest:
#         raise NotImplementedError()
#
#     if indexes is None:
#         indexes = range(nside ** 2 * 12)
#     npix = len(indexes)  # Number of pixels.
#     if npix >= (max(indexes) + 1):
#         # If the user input is not consecutive nodes, we need to use a slower
#         # method.
#         usefast = True
#         indexes = range(npix)
#     else:
#         usefast = False
#         indexes = list(indexes)
#
#     # Get the coordinates.
#     x, y, z = hp.pix2vec(nside, indexes, nest=nest)
#     coords = np.vstack([x, y, z]).transpose()
#     coords = np.asarray(coords, dtype=dtype)
#     # Get the 7-8 neighbors.
#     neighbors = hp.pixelfunc.get_all_neighbours(nside, indexes, nest=nest)
#     # if use_4:
#     #     print('Use 4')
#     #     col_index = []
#     #     row_index = []
#     #     for el,neighbor in zip(indexes,neighbors.T):
#     #         x, y, z = hp.pix2vec(nside, [el], nest=nest)
#     #         coords_1 = np.vstack([x, y, z]).transpose()
#     #         coords_1 = np.array(coords_1)
#
#     #         x, y, z = hp.pix2vec(nside, neighbor, nest=nest)
#     #         coords_2 = np.vstack([x, y, z]).transpose()
#     #         coords_2 = np.asarray(coords_2)
#     #         ind = np.argsort(np.sum((coords_2-coords_1)**2,axis=1),)[:4]
#     #         col_index = col_index + neighbor[ind].tolist()
#     #         row_index = row_index +[el]*4
#     #     col_index = np.array(col_index)
#     #     row_index = np.array(row_index)
#     # else:
#     # Indices of non-zero values in the adjacency matrix.
#     col_index = neighbors.T.reshape((npix * 8))
#     row_index = np.repeat(indexes, 8)
#
#     # Remove pixels that are out of our indexes of interest (part of sphere).
#     if usefast:
#         keep = (col_index < npix)
#         # Remove fake neighbors (some pixels have less than 8).
#         keep &= (col_index >= 0)
#         col_index = col_index[keep]
#         row_index = row_index[keep]
#     else:
#         col_index_set = set(indexes)
#         keep = [c in col_index_set for c in col_index]
#         inverse_map = [np.nan] * (nside ** 2 * 12)
#         for i, index in enumerate(indexes):
#             inverse_map[index] = i
#         col_index = [inverse_map[el] for el in col_index[keep]]
#         row_index = [inverse_map[el] for el in row_index[keep]]
#
#     # Compute Euclidean distances between neighbors.
#     distances = np.sum((coords[row_index] - coords[col_index]) ** 2, axis=1)
#     # slower: np.linalg.norm(coords[row_index] - coords[col_index], axis=1)**2
#
#     # Compute similarities / edge weights.
#     kernel_width = np.mean(distances)
#     weights = np.exp(-distances / (2 * kernel_width))
#
#     # Similarity proposed by Renata & Pascal, ICCV 2017.
#     # weights = 1 / distances
#
#     # Build the sparse matrix.
#     W = sparse.csr_matrix(
#         (weights, (row_index, col_index)), shape=(npix, npix), dtype=dtype)
#
#     # if use_4:
#     #     W = (W+W.T)/2
#
#     return W
#
#
# def build_laplacian(W, lap_type='normalized', dtype=np.float32):
#     """Build a Laplacian (tensorflow)."""
#     d = np.ravel(W.sum(1))
#     if lap_type == 'combinatorial':
#         D = sparse.diags(d, 0, dtype=dtype)
#         return (D - W).tocsc()
#     elif lap_type == 'normalized':
#         d12 = np.power(d, -0.5)
#         D12 = sparse.diags(np.ravel(d12), 0, dtype=dtype).tocsc()
#         return sparse.identity(d.shape[0], dtype=dtype) - D12 * W * D12
#     else:
#         raise ValueError('Unknown Laplacian type {}'.format(lap_type))
# 
#
# def healpix_graph(nside=16,
#                   nest=True,
#                   lap_type='normalized',
#                   indexes=None,
#                   use_4=False,
#                   dtype=np.float32):
#     """Build a healpix graph using the pygsp from NSIDE."""
#     from pygsp import graphs
#
#     if indexes is None:
#         indexes = range(nside ** 2 * 12)
#
#     # 1) get the coordinates
#     npix = hp.nside2npix(nside)  # number of pixels: 12 * nside**2
#     pix = range(npix)
#     x, y, z = hp.pix2vec(nside, pix, nest=nest)
#     coords = np.vstack([x, y, z]).transpose()[indexes]
#     # 2) computing the weight matrix
#     if use_4:
#         raise NotImplementedError()
#         W = build_matrix_4_neighboors(nside, indexes, nest=nest, dtype=dtype)
#     else:
#         W = healpix_weightmatrix(
#             nside=nside, nest=nest, indexes=indexes, dtype=dtype)
#     # 3) building the graph
#     G = graphs.Graph(
#         W,
#         lap_type=lap_type,
#         coords=coords)
#     return G
#
#
# def healpix_laplacian(nside=16,
#                       nest=True,
#                       lap_type='normalized',
#                       indexes=None,
#                       dtype=np.float32,
#                       use_4=False):
#     """Build a Healpix Laplacian."""
#     if use_4:
#         W = build_matrix_4_neighboors(nside, indexes, nest=nest, dtype=dtype)
#     else:
#         W = healpix_weightmatrix(
#             nside=nside, nest=nest, indexes=indexes, dtype=dtype)
#     L = build_laplacian(W, lap_type=lap_type)
#     return L
#
#
# def rescale_L(L, lmax=2, scale=1):
#     """Rescale the Laplacian eigenvalues in [-scale,scale]."""
#     M, M = L.shape
#     I = sparse.identity(M, format='csr', dtype=L.dtype)
#     L *= 2 * scale / lmax
#     L -= I
#     return L
#
#
# def build_laplacians(nsides, indexes=None, use_4=False):
#     """Build a list of Laplacians (and down-sampling factors) from a list of nsides."""
#     L = []
#     p = []
#     if indexes is None:
#         indexes = [None] * len(nsides)
#     for i, (nside, index) in enumerate(zip(nsides, indexes)):
#         if i > 0:  # First is input dimension.
#             p.append((nside_last // nside) ** 2)
#         nside_last = nside
#         if i < len(nsides) - 1:  # Last does not need a Laplacian.
#             laplacian = healpix_laplacian(nside=nside, indexes=index, use_4=use_4)
#             L.append(laplacian)
#     return L, p
#
#
# class ChebyshevGraphConv(layers.Layer):
#     def __init__(self, nside=1024, order=2, f_in=1, f_out=1, k=5):
#         super(ChebyshevGraphConv, self).__init__()
#         self.nside = nside
#         self.order = order
#         self.Fin = f_in
#         self.Fout = f_out
#         self.L = healpix_laplacian(nside=nside, indexes=None, use_4=False)
#         self.K = k
#
#     def build(self, input_shape):
#         """Xavier like weight initializer for Chebychev coefficients."""
#         stddev = 1 / np.sqrt(self.Fin * (self.K + 0.5) / 2)
#         self.W = self.add_weight(shape=(self.Fin * self.K, self.Fout),
#                                  initializer=tf.keras.initializers.random_normal(stddev=stddev), trainable=True)
#
#     def call(self, inputs):
#         print(inputs.get_shape())
#         N, M, Fin = inputs.get_shape()
#         N, M, Fin = int(N), int(M), int(Fin)
#         # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.
#         self.L = sparse.csr_matrix(self.L)
#         lmax = 1.02 * sparse.linalg.eigsh(
#             self.L, k=1, which='LM', return_eigenvectors=False)[0]
#         self.L = rescale_L(self.L, lmax=lmax, scale=0.75)
#         self.L = self.L.tocoo()
#         indices = np.column_stack((self.L.row, self.L.col))
#         self.L = tf.SparseTensor(indices, self.L.data, self.L.shape)
#         self.L = tf.sparse_reorder(self.L)
#         # Transform to Chebyshev basis
#         x0 = tf.transpose(inputs, perm=[1, 2, 0])  # M x Fin x N
#         x0 = tf.reshape(x0, [M, self.Fin * N])  # M x Fin*N
#         x = tf.expand_dims(x0, 0)  # 1 x M x Fin*N
#
#         def concat(x, x_):
#             x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
#             return tf.concat([x, x_], axis=0)  # K x M x Fin*N
#
#         if self.K > 1:
#             x1 = tf.sparse_tensor_dense_matmul(self.L, x0)
#             x = concat(x, x1)
#         for k in range(2, self.K):
#             x2 = 2 * tf.sparse_tensor_dense_matmul(self.L, x1) - x0  # M x Fin*N
#             x = concat(inputs, x2)
#             x0, x1 = x1, x2
#         x = tf.reshape(x, [self.K, M, self.Fin, N])  # K x M x Fin x N
#         x = tf.transpose(x, perm=[3, 1, 2, 0])  # N x M x Fin x K
#         x = tf.reshape(x, [N * M, self.Fin * self.K])  # N*M x Fin*K
#         # Filter: Fin*Fout filters of order K, i.e. one filterbank per output feature.
#         x = tf.matmul(x, self.W)  # N*M x Fout
#         return tf.reshape(x, [N, M, self.Fout])  # N x M x Fout
#
#
# def create_compile_model(nside=1024, order=2):
#     inputs = keras.Input(shape=(12*nside**2, 1), name="map")
#     outputs = ChebyshevGraphConv()(inputs)
#     model = keras.Model(inputs=inputs, outputs=outputs, name="single-conv-layer-model")
#
#     return model
#
#
# model = create_compile_model()
#
#
# def generate_data():
#     return tf.ones(shape=(2, 12*1024**2, 1))
#
#
# m = generate_data()
#
# results = model(m)
