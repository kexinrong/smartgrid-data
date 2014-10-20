# This file extracts voltages (real part) of each house, and outputs to
#  different files according to optimize model

from scipy.io import loadmat
import glob
import math
HOUSE_START = 525 # start index of house
HOUSE_END = 1933 # end index of hosue
NUM_HOUSE = HOUSE_END - HOUSE_START + 1
BIN_SIZE = 12 # bin every 12 data points => 1 hour interval
MODELS = ['baseline_cvr', 'baseline', 'optimized_cvr', 'optimized']
YEARS = ['2012', '2016', '2020']
MONTHS = ['_07_', '_10_', '_02_', '_03_']
INPUTS = glob.glob('../data/output_bfs_social/*/*.mat')

def rebin(raw_shape):
    '''Bin the raw data points according to BIN_SIZE'''
    accum = 0
    count = 0
    shape = []
    for v in raw_shape:
        accum += v
        count += 1
        if count % BIN_SIZE == 0:
            count = 0
            shape.append(str(accum / BIN_SIZE))
            accum = 0
    return shape

def get_phase(bus):
    '''Get the phase for each house from .network.bus'''
    phases = {}
    for i in range(HOUSE_START - 1, HOUSE_END):
        if int(bus[i][1].real):
            phases[i + 1] = 0 # phase A
        elif int(bus[i][2].real):
            phases[i + 1] = 1 # phase B
        elif int(bus[i][3].real):
            phases[i + 1] = 2 # phase C    
    return phases

def find_model(filename):
    '''Find model used for the mat file'''
    for i in range(len(YEARS)):
        for j in range(len(MODELS)):
            for k in range(len(MONTHS)):
                model = YEARS[i] + MODELS[j]
                if model in filename and MONTHS[k] in filename:
                    return (i, j, k)

def get_devices(network):
    devices = {}
    for i in range(2065):
        devices[i] = [0, 0, 0, 0]
    #hvac
    table = network['hvac'][0][0]
    for i in range(len(table)):
        devices[int(table[i][0].real)][0] = 1
    #pv
    table = network['pv'][0][0]
    for i in range(len(table)):
        devices[int(table[i][0].real)][1] = 1
    #ev
    table = network['ev'][0][0]
    for i in range(len(table)):
        devices[int(table[i][0].real)][2] = 1
    #pool
    table = network['pool'][0][0]
    for i in range(len(table)):
        devices[int(table[i][0].real)][3] = 1
    return devices


# Open one output file for each optimization model
outputs = []
for i in range(len(YEARS)):
    outputs.append([])
    for j in range(len(MODELS)):
        outputs[i].append([])
        for k in range(len(MONTHS)):
            f = open('../data/load/' + YEARS[i] + MODELS[j] + MONTHS[k][:-1] + '.txt', 'w')
            # Write bin size
            f.write(str(24 * 12 / BIN_SIZE) + '\n')
            outputs[i][j].append(f)

for mat in INPUTS:
    print "Processing: " + mat    
    m = loadmat(mat)
    # Get phase of each house
    phases = get_phase(m['network']['bus'][0][0])
    devices = get_devices(m['network'])
    # Find model used by the mat file
    (x, y, z) = find_model(mat)
    load = m['network']['bus'][0][0]
    for i in range(HOUSE_START - 1, HOUSE_END):
        index = phases[i + 1]
        raw_shape = []
        for t in range(288):
            # Get the real load of specified phase
            raw_shape.append(load[i][5 + index].real)
            index += 3   
        shape = rebin(raw_shape)
        # write to correpsonding output file        
        outputs[x][y][z].write(str(i + 1) + ' ')
        for d in devices[i + 1]:
            outputs[x][y][z].write(str(d) + ' ')
        outputs[x][y][z].write(' '.join(shape) + '\n')

# close files
for i in range(len(YEARS)):
    for j in range(len(MODELS)):
        for k in range(len(MONTHS)):
            outputs[i][j][k].close()