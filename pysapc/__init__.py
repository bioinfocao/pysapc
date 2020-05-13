# For relative imports to work in Python 3.6
import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from SparseAPCluster import SAP
from SparseAPCluster import sparseAffinityPropagation
import sparseAP_cy