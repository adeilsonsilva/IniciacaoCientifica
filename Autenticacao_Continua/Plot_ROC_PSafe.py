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
        print "usage: <scores>] <image>"
        sys.exit(0)

    FileName = sys.argv[1]
    ofname = sys.argv[2]

    f, ext = os.path.splitext(ofname)
    if not IsSupportedImageType(ext):
        print("Format \"%s\" is not supported!" % ext)
        sys.exit(0)

    # read id of actives

    actives = LoadActives(FileName)
    print("Loaded %d actives from %s" % (len(actives), FileName))

    # read molecule id - score pairs

    scores = LoadScores(FileName)
    print("Loaded %d scores from %s" % (len(scores), FileName))
    # sort scores by ascending order
    sortedScores = sorted(scores, key=itemgetter(1))
    sortedScores.reverse()
    print("Plotting ROC Curve ...")
    #color1 = "#008000"  # dark green
    DepictROCCurve(actives, sortedScores, ofname, 1)


def LoadActives(fname):

    actives = []
    for line in open(fname, 'r').readlines():
        id = line.strip()
        if(id[0] == '1'):
        	actives.append(id[0])

    return actives


def LoadScores(fname):

    sfile = open(fname, 'r')

    scores = []
    for line in sfile.readlines():
        id, score = line.strip().split()
        scores.append((id, float(score)))

    return scores


def GetRates(actives, scores):

    tpr = [0.0]  # true positive rate
    fpr = [0.0]  # false positive rate
    nractives = len(actives)
    nrdecoys = len(scores) - len(actives)
    foundactives = 0.0
    founddecoys = 0.0
    for idx, (id, score) in enumerate(scores):
        if id in actives:
            foundactives += 1.0
        else:
            founddecoys += 1.0

        tpr.append(foundactives / float(nractives))
        fpr.append(founddecoys / float(nrdecoys))
    return tpr, fpr


def SetupROCCurvePlot(plt):

    plt.xlabel("FPR", fontsize=14)
    plt.ylabel("TPR", fontsize=14)
    plt.title("ROC Curve", fontsize=14)


def SaveROCCurvePlot(plt, fname, randomline=True):

    if randomline:
        x = [0.0, 0.18]
	y = [1.00, 0.825]
        plt.plot(x, y, linestyle='dashed', color='gray', linewidth=2)

    plt.xlim(0.0, 0.18)
    plt.ylim(0.825, 1.00)
    plt.legend(fontsize=10, loc='best')
    plt.tight_layout()
    plt.savefig(fname)


def AddROCCurve(plt, actives, scores, color, labelEigen):

    tpr, fpr = GetRates(actives, scores)

    plt.plot(fpr, tpr, color=color, linewidth=2)


def DepictROCCurve(actives, sortedScores, fname, randomline=True):

    plt.figure(figsize=(6, 6), dpi=80)

    SetupROCCurvePlot(plt)
    AddROCCurve(plt, actives, sortedScores, "red", "System")
    SaveROCCurvePlot(plt, fname, randomline)


def IsSupportedImageType(ext):
    fig = plt.figure()
    return (ext[1:] in fig.canvas.get_supported_filetypes())


if __name__ == "__main__":
    sys.exit(main(sys.argv))
