#!/usr/bin/env python
#############################################################################
# Copyright (C) 2013-2015 OpenEye Scientific Software, Inc.
#############################################################################
# Plotting ROC curve
#############################################################################

import sys
import os
from operator import itemgetter
import matplotlib.pyplot as plt


def main(argv=[__name__]):

    if len(sys.argv) != 3:
        print "usage: <filePSafe> ] <image>"
        sys.exit(0)

    fname = sys.argv[1]
    ofname = sys.argv[2]

    f, ext = os.path.splitext(ofname)
    if not IsSupportedImageType(ext):
        print("Format \"%s\" is not supported!" % ext)
        sys.exit(0)

    print("Plotting ROC Curve ...")
    sfile = open(fname, 'r')
    sfile.readline()
    valores = []
    for line in sfile.readlines():
    	valores.append(float(line))


    DepictROCCurve(valores, ofname, 0)



def LoadData(fname):
	sfile = open(fname, 'r')
	sfile.readline()
	tpr = []
	fpr = []
	for line in sfile.readlines():
		fields = line.rstrip().split(' ')
		tpr.append(float(fields[0]))
		fpr.append(float(fields[1]))
	return tpr, fpr



def SetupROCCurvePlot(plt):

    plt.xlabel("Frame", fontsize=14)
    plt.ylabel("P safe", fontsize=14)
    plt.title("Probabilidade do Sistema estar Seguro", fontsize=14)


def SaveROCCurvePlot(plt, fname, randomline=True):

    if randomline:
        x = [0.0, 0.1]
        y = [0.75, 1.0]
        plt.plot(x, y, linestyle='dashed', color='red', linewidth=2, label='random')

    plt.xlim(1.0, 999)
    plt.ylim(0.00, 1.00)
    plt.legend(fontsize=10, loc='best')
    plt.tight_layout()
    plt.savefig(fname)


def AddROCCurve(plt, tpr, fpr, color, label):

    plt.plot(fpr, tpr, color=color, linewidth=2)


def DepictROCCurve(valores, fname, randomline=True):

    plt.figure(figsize=(10, 4), dpi=80)
    eixo = []
    for i in range(1,999):
    	eixo.append(i)
    print len(valores)
    print len(eixo)
    SetupROCCurvePlot(plt)
    AddROCCurve(plt, valores, eixo, "red", "System")
    SaveROCCurvePlot(plt, fname, randomline)


def IsSupportedImageType(ext):
    fig = plt.figure()
    return (ext[1:] in fig.canvas.get_supported_filetypes())


if __name__ == "__main__":
    sys.exit(main(sys.argv))
