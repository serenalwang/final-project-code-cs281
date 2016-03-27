import ascdata

# ascdata.save_asc_data()
X_raw, y_raw = ascdata.load_asc_data()

X_nonzero,y_nonzero = ascdata.remove_zeros(X_raw, y_raw)
print "done removing zeros"
del X_raw, y_raw
X_noncrazy,y_noncrazy = ascdata.remove_crazy(X_nonzero,y_nonzero, 100)
# print "done removing noncrazy"
X_shrunk, y_shrunk = ascdata.shrink_data(X_noncrazy, y_noncrazy, 10)
print "done shrinking data"

# X_nonzero, y_nonzero = ascdata.remove_zeros(X, y)
#
# X_multi, y_multi = ascdata.get_multitask_data(X_nonzero,y_nonzero)

print X_shrunk, y_shrunk