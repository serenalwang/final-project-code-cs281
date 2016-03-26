import matplotlib.pyplot as plt
import ascdata

### IMPORT DATA ###
print "IMPORTING DATA"

X_raw, y_raw = ascdata.load_asc_data()
X, y = ascdata.remove_zeros(X_raw, y_raw)

### IP STATS ###

# Plots the cross entropy loss vs. the IP value for a given program.
def plot_loss_value(prog_num):
    plt.cla()
    X_prog, y_prog = ascdata.get_bp_data(prog_num, 0, X, y)
    ips = X_prog.T[1]
    loss_values = y_prog
    plt.plot(ips, loss_values, 'bx')
    plt.savefig("stats" + "_" + str(prog_num) + "_" + "ips" + "_" + "loss-values" + ".png")

# Plots the hamming distance vs. the IP value for a given program.
def plot_hamming_distance(prog_num):
    plt.cla()
    X_prog, y_prog = ascdata.get_bp_data(prog_num, 0, X, y)
    ips = X_prog.T[1]
    hamming_distances = X_prog.T[7]
    plt.plot(ips, hamming_distances, 'bx')
    plt.savefig("stats" + "_" + str(prog_num) + "_" + "ips" + "_" + "hamming-distances" + ".png")

# Plots the hamming distance vs. the IP value for a given program.
def plot_ir_counts(prog_num):
    plt.cla()
    X_prog, y_prog = ascdata.get_bp_data(prog_num, 0, X, y)
    ips = X_prog.T[1]
    ir_counts = X_prog.T[303]
    plt.plot(ips, ir_counts, 'bx')
    plt.savefig("stats" + "_" + str(prog_num) + "_" + "ips" + "_" + "ir-counts" + ".png")

# Plots the hamming distance vs. the IP value for a given program.
def plot_ir_percents(prog_num):
    plt.cla()
    X_prog, y_prog = ascdata.get_bp_data(prog_num, 0, X, y)
    ips = X_prog.T[1]
    ir_percents = X_prog.T[304]
    plt.plot(ips, ir_percents, 'bx')
    plt.savefig("stats" + "_" + str(prog_num) + "_" + "ips" + "_" + "ir-percents" + ".png")

print "GETTING IP STATS"
nprogs = 7
for prog_num in range(1,nprogs + 1):
    print "for program number", prog_num
    # plot_loss_value(prog_num)
    # plot_hamming_distance(prog_num)
    plot_ir_counts(prog_num)
    plot_ir_percents(prog_num)