import ascdata

# ascdata.save_asc_data()
X, y = ascdata.load_asc_data()

X_nonzero, y_nonzero = ascdata.remove_zeros(X, y)

X_multi, y_multi = ascdata.get_multitask_data(X_nonzero,y_nonzero)

print X_multi, y_multi