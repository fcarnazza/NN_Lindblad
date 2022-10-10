import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
"""
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family: 'serif',
    'text.usetex: True,
    'pgf.rcfonts: False,
})
"""
ind = np.arange(10)
matplotlib.rcParams['text.usetex'] = True
matr = np.zeros(100).reshape(10,10)
matr2 = np.zeros(100).reshape(10,10)

b_dict = {'0.000000':0, '0.300000':1 ,'0.500000':2, '1.000000':3, '1.200000':4, '2.000000':5, '3.000000':6, '6.000000':7, '10.000000':8, '20.000000':9}
c_dict = {'0.100000':0,'0.200000':1,'0.300000':2,'0.400000':3, '0.500000':4, '0.600000':5, '0.800000':6, '1.000000':7,'1.500000':8, '2.000000':9}
filenames = [  'var_{6}_OBC_inter','var_{6}_OBC_extra', 'var_{8}_OBC_inter','var_{8}_OBC_extra','var_{10}_OBC_inter','var_{10}_OBC_extra' ]

for filen in filenames:
        matr = np.zeros(100).reshape(10,10)
        matr2 = np.zeros(100).reshape(10,10)
        filename = filen + '.txt'
        data = np.loadtxt(filename, delimiter=' ', skiprows=0, dtype=str)
        for i in range(len(data)):
         matr[c_dict[data[i][1]]][b_dict[data[i][3]] ] +=  float(data[i][6])
         matr2[c_dict[data[i][1]]][b_dict[data[i][3]] ] +=  float(data[i][6])/float(data[i][6])
        matr = matr/matr2

        b = ['0.0', '0.1', '0.3', '0.8', '1.0', '1.5', '2.0',  '8.0', '10.0']
        c = ['0.2', '0.3', '0.4', '0.5', '0.6', '0.8', '1.0', '1.5', '2.0']
        print(matr)
        print(matr2)

        fig, ax = plt.subplots()
        ax.set_xticks(ind)
        ax.set_yticks(ind)
        cax = ax.matshow( matr)#,interpolation='nearest')
        ax.set_xticklabels(c)
        ax.set_yticklabels(b)
        ax.set_ylabel(r'$\beta$')
        ax.set_xlabel(r'$V$')
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
        ax.imshow(matr, cmap='Greys',norm=norm)#,  interpolation='nearest')
        cmap = matplotlib.cm.binary

        fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap))

        path = os.path.join('/Users/francesco/Dropbox/PHYSICS_CORONA/Francesco_Carnazza/NN_docs/draft', filen+".pdf")
        plt.savefig(path,format ='pdf')#,dpi = 1200)


