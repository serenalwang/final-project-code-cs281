import ascdata
import numpy as np

# ascdata.save_asc_data()
X, y = ascdata.load_nonzero_asc_data()

# X_prog1, y_prog1, X_prog2, y_prog2 = ascdata.load_shrunken_noprogs()

X_prog1, y_prog1 = ascdata.get_bp_data(5, 0, X, y)
X_prog2, y_prog2 = ascdata.get_bp_data(1, 0, X, y)
X_prog3, y_prog3 = ascdata.get_bp_data(4, 0, X, y)

X_notprog1, y_notprog1 = ascdata.get_all_but_prog_data(5, X, y)
X_notprog2, y_notprog2 = ascdata.get_all_but_prog_data(1, X, y)
X_notprog3, y_notprog3 = ascdata.get_all_but_prog_data(4, X, y)

print "prog 1 y lengths", y_prog1.shape, y_notprog1.shape
print "prog 2 y lengths", y_prog2.shape, y_notprog2.shape
print "prog 3 y lengths", y_prog3.shape, y_notprog3.shape

np.save("X_good_nonzero.npy",X_prog1)
np.save("y_good_nonzero.npy",y_prog1)

np.save("X_bad_nonzero.npy",X_prog2)
np.save("y_bad_nonzero.npy",y_prog2)

np.save("X_superbad_nonzero.npy",X_prog3)
np.save("y_superbad_nonzero.npy",y_prog3)

np.save("X_good_noprog_nonzero.npy",X_notprog1)
np.save("y_good_noprog_nonzero.npy",y_notprog1)

np.save("X_bad_noprog_nonzero.npy",X_notprog2)
np.save("y_bad_noprog_nonzero.npy",y_notprog2)

np.save("X_superbad_noprog_nonzero.npy",X_notprog3)
np.save("y_superbad_noprog_nonzero.npy",y_notprog3)

# X_nonzero,y_nonzero = ascdata.remove_zeros(X_raw, y_raw)
# print "done removing zeros"
# del X_raw, y_raw
#X_noncrazy,y_noncrazy = ascdata.remove_crazy(X_raw,y_raw, 100)

#print "done removing crazy"
# X_shrunk, y_shrunk = ascdata.shrink_data(X_noncrazy, y_noncrazy, 10)
# print "done shrinking data"
#
# # X_nonzero, y_nonzero = ascdata.remove_zeros(X, y)
# #
# # X_multi, y_multi = ascdata.get_multitask_data(X_nonzero,y_nonzero)
#
# print X_shrunk, y_shrunk