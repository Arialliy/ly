import sys, runpy, importlib.metadata as m
sys.path.insert(0, '/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/official_deeppro')
import skimage, sklearn, thop, matplotlib.pyplot
from data_utils.TestDataLoader import TestIRSeqDataLoader
from ShootingRules import ShootingRules
from write_results import writeNUDTMIRSDT_ROC
print({p: m.version(p) for p in ["torch", "numpy", "scikit-image", "scikit-learn", "thop", "pyparsing"]}, flush=True)
runpy.run_path('/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/runtime/gpu_witness.py')
