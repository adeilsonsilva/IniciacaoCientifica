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

    if len(sys.argv) != 5:
        print "usage: <scoresEigen> <scoresFisher> <scoresLBPH>] <image>"
        sys.exit(0)

    EigenFileName = sys.argv[1]
    FisherFileName = sys.argv[2]
    LBPHFileName = sys.argv[3]
    ofname = sys.argv[4]

    f, ext = os.path.splitext(ofname)
    if not IsSupportedImageType(ext):
        print("Format \"%s\" is not supported!" % ext)
        sys.exit(0)

    # read id of actives

    activesEigen = LoadActives(EigenFileName)
    activesFisher = LoadActives(FisherFileName)
    activesLBPH = LoadActives(LBPHFileName)
    print("Loaded %d actives from %s" % (len(activesEigen), EigenFileName))
    print("Loaded %d actives from %s" % (len(activesFisher), FisherFileName))
    print("Loaded %d actives from %s" % (len(activesLBPH), LBPHFileName))

    # read molecule id - score pairs

    labelEigen, scoresEigen = LoadScores(EigenFileName)
    labelFisher, scoresFisher = LoadScores(FisherFileName)
    labelLBPH, scoresLBPH = LoadScores(LBPHFileName)
    print("Loaded %d %s scores from %s" % (len(scoresEigen), labelEigen, EigenFileName))
    print("Loaded %d %s scores from %s" % (len(scoresFisher), labelFisher, FisherFileName))
    print("Loaded %d %s scores from %s" % (len(scoresLBPH), labelLBPH, LBPHFileName))

    # sort scores by ascending order
    sortedscoresEigen = sorted(scoresEigen, key=itemgetter(1))
    sortedscoresFisher = sorted(scoresFisher, key=itemgetter(1))
    sortedscoresLBPH = sorted(scoresLBPH, key=itemgetter(1))

    print("Plotting ROC Curve ...")
    #color1 = "#008000"  # dark green
    DepictROCCurve(activesEigen, activesFisher, activesLBPH, sortedscoresEigen, sortedscoresFisher, sortedscoresLBPH, labelEigen, labelFisher, labelLBPH, ofname)


def LoadActives(fname):

    actives = []
    for line in open(fname, 'r').readlines():
        id = line.strip()
        if(id[0] == '1'):
        	actives.append(id[0])

    return actives


def LoadScores(fname):

    sfile = open(fname, 'r')
    labelEigen = sfile.readline()
    labelEigen = labelEigen.strip()

    scores = []
    for line in sfile.readlines():
        id, score = line.strip().split()
        scores.append((id, float(score)))

    return labelEigen, scores


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
        x = [0.0, 1.0]
        y = [0.75, 1.0]
        plt.plot(x, y, linestyle='dashed', color='red', linewidth=2, label='random')

    plt.xlim(0.0, 1.0)
    plt.ylim(0.75, 1.0)
    plt.legend(fontsize=10, loc='best')
    plt.tight_layout()
    plt.savefig(fname)


def AddROCCurve(plt, actives, scores, color, labelEigen):

    tpr, fpr = GetRates(actives, scores)

    plt.plot(fpr, tpr, color=color, linewidth=2, label=labelEigen)


def DepictROCCurve(activesEigen, activesFisher, activesLBPH, sortedscoresEigen, sortedscoresFisher, sortedscoresLBPH, labelEigen, labelFisher, labelLBPH, fname, randomline=True):

    plt.figure(figsize=(4, 4), dpi=80)

    SetupROCCurvePlot(plt)
    AddROCCurve(plt, activesEigen, sortedscoresEigen, "green", labelEigen)
    AddROCCurve(plt, activesFisher, sortedscoresFisher, "blue", labelFisher)
    AddROCCurve(plt, activesLBPH, sortedscoresLBPH, "yellow", labelLBPH)
    SaveROCCurvePlot(plt, fname, randomline)


def IsSupportedImageType(ext):
    fig = plt.figure()
    return (ext[1:] in fig.canvas.get_supported_filetypes())


if __name__ == "__main__":
    sys.exit(main(sys.argv))
