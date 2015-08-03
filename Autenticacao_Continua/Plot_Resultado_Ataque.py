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

    if len(sys.argv) != 6:
        print "usage: <filePSafe1> <filePSafe1x2> <filePSafe1x3> <filePSafe1x4> ] <image>"
        sys.exit(0)

    fname1 = sys.argv[1]
    fname2 = sys.argv[2]
    fname3 = sys.argv[3]
    fname4 = sys.argv[4]
    ofname = sys.argv[5]

    f, ext = os.path.splitext(ofname)
    if not IsSupportedImageType(ext):
        print("Format \"%s\" is not supported!" % ext)
        sys.exit(0)

    print("Plotting ROC Curve ...")
    sfile = open(fname1, 'r')
    sfile.readline()
    valores1 = []
    valores2 = []
    valores3 = []
    valores4 = []
    valores = []
    for line in sfile.readlines():
    	valores1.append(float(line))

    sfile = open(fname2, 'r')
    sfile.readline()
    for line in sfile.readlines():
        valores2.append(float(line))

    sfile = open(fname3, 'r')
    sfile.readline()
    for line in sfile.readlines():
        valores3.append(float(line))

    sfile = open(fname4, 'r')
    sfile.readline()
    for line in sfile.readlines():
        valores4.append(float(line))

    valores.append(valores1)
    valores.append(valores2)
    valores.append(valores3)
    valores.append(valores4)

    DepictROCCurve(valores, ofname, 1)



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

    plt.xlabel("Quadro", fontsize=14)
    plt.ylabel("P seguro", fontsize=14)
    plt.title("Probabilidade do Sistema estar Seguro", fontsize=14)


def SaveROCCurvePlot(plt, fname, randomline=True):

    if randomline:
        x = [1000.0, 1000.0]
        y = [0.0, 1.0]
        plt.plot(x, y, linestyle='dashed', color='gray', linewidth=2)

    plt.xlim(1.0, 2000)
    plt.ylim(0.00, 1.00)
    plt.legend(fontsize=10, loc='best')
    plt.tight_layout()
    plt.grid(color='gray', linestyle='-', linewidth=0.5)
    plt.savefig(fname)


def AddROCCurve(plt, tpr, fpr, color, label):

    plt.plot(fpr, tpr, color=color, linewidth=2)


def DepictROCCurve(valores, fname, randomline=True):

    plt.figure(figsize=(15, 4), dpi=80)
    eixo1 = []
    eixo2 = []
    eixo3 = []
    eixo4 = []
    maximo = len(valores[0])
    for i in range(0, maximo):
        eixo1.append(i)
    for i in range(maximo,len(valores[1])+maximo):
        eixo2.append(i)
    for i in range(maximo,len(valores[2])+maximo):
        eixo3.append(i)
    for i in range(maximo,len(valores[3])+maximo):
        eixo4.append(i)
    print len(eixo1)
    print len(valores[0])
    print len(eixo2)
    print len(valores[1])
    print len(eixo3)
    print len(valores[2])
    print len(eixo4)
    print len(valores[3])

    SetupROCCurvePlot(plt)
    AddROCCurve(plt, valores[0], eixo1, "blue", "System")
    AddROCCurve(plt, valores[1], eixo2, "red", "System")
    AddROCCurve(plt, valores[2], eixo3, "yellow", "System")
    AddROCCurve(plt, valores[3], eixo4, "green", "System")
    SaveROCCurvePlot(plt, fname, randomline)


def IsSupportedImageType(ext):
    fig = plt.figure()
    return (ext[1:] in fig.canvas.get_supported_filetypes())


if __name__ == "__main__":
    sys.exit(main(sys.argv))
