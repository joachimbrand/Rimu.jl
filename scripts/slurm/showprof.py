import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import re

parser = argparse.ArgumentParser(description='Plot CPU time')
parser.add_argument('--csv', default='', 
                    help='CSV file')
parser.add_argument('--title', default='',
                    help='Plot title')
parser.add_argument('--max', default=5, type=int,
                    help='Max number of functions')
parser.add_argument('--metric', default='CPU Time', 
	            help='Column name or metric')
parser.add_argument('--save', action='store_true', 
		    help='Save figure in PNG file')
args = parser.parse_args()

import matplotlib as mpl
if args.save:
    # required for batch jobs
    mpl.use('Agg')

if not args.csv:
    print('ERROR: must provide csv file')
    sys.exit(1)

def rgb(x):
    '''
    Return RGB color
    @param x normalized value in range [0, 1]
    '''
    quarterPi = np
    r = np.sin(np.pi*x/2.)**2
    g = np.cos(np.pi*x/2.)**2
    b = np.sin(np.pi*x)**2
    return (r, g, b)

df = pd.read_csv(args.csv, sep='\t')

n = min(args.max,df.shape[0])
df = df.loc[:n-1]
print(df.head)

xPos = np.arange(len(df['Function']))
vmax = df[args.metric].max()
vmin = df[args.metric].min()
clrs = [rgb((v - vmin)/(vmax - vmin)) for v in df[args.metric]]
plt.bar(xPos, df[args.metric], color=clrs)
plt.xticks(xPos, df['Function'])
plt.title(args.title)
plt.ylabel(args.metric)
if args.save:
    filename = re.sub(r'.csv$', '.png', args.csv)
    fig = plt.gcf()
    fig.set_size_inches(15, 3)
    print('Saving figure in file {}'.format(filename))
    plt.savefig(filename)
else:
    plt.show()
