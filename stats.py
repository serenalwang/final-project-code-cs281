import matplotlib.pyplot as plt
import ascdata

### IMPORT DATA ###
print "IMPORTING DATA"

#X, y = ascdata.load_nonzero_asc_data()
# X_nonzero, y_nonzero = ascdata.remove_zeros(X_raw, y_raw)
# X, y = ascdata.remove_crazy(X_nonzero,y_nonzero, 10)



### IP STATS ###

# Plots the cross entropy loss vs. the IP value for a given program.
def plot_loss_value(prog_num):
    plt.cla()
    X_prog, y_prog = ascdata.get_bp_data(prog_num, 0, X, y)
    ips = X_prog.T[1]
    loss_values = y_prog
    plt.plot(ips, loss_values, 'bx')
    plt.savefig("stats" + "_" + str(prog_num) + "_" + "ips" + "_" + "loss-values" + "_" + "notcrazy" + ".png")

# Plots the hamming distance vs. the IP value for a given program.
def plot_hamming_distance(prog_num):
    plt.cla()
    X_prog, y_prog = ascdata.get_bp_data(prog_num, 0, X, y)
    ips = X_prog.T[1]
    hamming_distances = X_prog.T[7]
    plt.plot(ips, hamming_distances, 'bx')
    plt.savefig("stats" + "_" + str(prog_num) + "_" + "ips" + "_" + "hamming-distances" + "_" + "notcrazy" + ".png")

# Plots the hamming distance vs. the IP value for a given program.
def plot_ir_counts(prog_num):
    plt.cla()
    X_prog, y_prog = ascdata.get_bp_data(prog_num, 0, X, y)
    ips = X_prog.T[1]
    ir_counts = X_prog.T[303]
    plt.plot(ips, ir_counts, 'bx')
    plt.savefig("stats" + "_" + str(prog_num) + "_" + "ips" + "_" + "ir-counts" + "_" + "notcrazy" + ".png")

# Plots the hamming distance vs. the IP value for a given program.
def plot_ir_percents(prog_num):
    plt.cla()
    X_prog, y_prog = ascdata.get_bp_data(prog_num, 0, X, y)
    ips = X_prog.T[1]
    ir_percents = X_prog.T[304]
    plt.plot(ips, ir_percents, 'bx')
    plt.savefig("stats" + "_" + str(prog_num) + "_" + "ips" + "_" + "ir-percents" + "_" + "notcrazy"+ ".png")

# Plots a histogram of all of the target values for a given program.
# If prog_num == 0, then plot the histogram for all target values.
def plot_target_histogram(prog_num):
    print "target histogram for prog num", prog_num
    plt.cla()
    if prog_num == 0:
        plt.hist(y)
    else:
        X_prog, y_prog = ascdata.get_bp_data(prog_num, 0, X, y)
        plt.hist(y_prog)
    plt.xlim([0,100000])
    plt.savefig("stats" + "_" + str(prog_num) + "_" + "target_hist" + "_" + "rawlim"+ ".png")

# type is a string representing the type of program.
def plot_program(prog_num, bp, inX, iny, type):
    plt.cla()
    X_bp, y_bp = ascdata.get_bp_data(prog_num, bp, inX, iny)
    plot_inputs = X_bp.T[2]
    plt.plot(plot_inputs, y_bp, 'b^')
    plt.savefig("stats" + "_" + "lossvalues" + "_" + type+ ".png")

print "GETTING STATS"
#plot_target_histogram(0)
#nprogs = 7
# for prog_num in range(1,nprogs + 1):
#     print "for program number", prog_num
    #plot_loss_value(prog_num)
    #plot_hamming_distance(prog_num)
    #plot_ir_counts(prog_num)
    #plot_ir_percents(prog_num)
    # plot_target_histogram(prog_num)

X_prog1_nz, y_prog1_nz, X_prog2_nz, y_prog2_nz, X_prog3_nz, y_prog3_nz = ascdata.load_nonzero_progs()
X_prog1, y_prog1, X_prog2, y_prog2 = ascdata.load_shrunken_progs()

# plot_program(5, int("40024f",16), X_prog1_nz, y_prog1_nz, "goodnonzero")
# plot_program(1, int("4014d2",16), X_prog2_nz, y_prog2_nz, "badnonzero")
plot_program(4, int("400649",16), X_prog3_nz, y_prog3_nz, "sbadnonzero")

# plot_program(5, int("40024f",16), X_prog1, y_prog1, "goodshrunk")
# plot_program(1, int("4014d2",16), X_prog2, y_prog2, "badshrunk")