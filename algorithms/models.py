import sys

from algorithms.baseline.LinearRegression import *
from algorithms.baseline.ConfidenceBall import *
from algorithms.baseline.LinUCB import *
from algorithms.baseline.Optimal import *
from algorithms.baseline.Random import *
from algorithms.baseline.ThompsonSampling import *

from algorithms.fjlt.LinearRegression_FJLT import *
from algorithms.fjlt.ConfidenceBall_FJLT import *
from algorithms.fjlt.ConfidenceBall_FJLT_Cholesky import *

from algorithms.fjlt.LinUCB_FJLT import *
from algorithms.fjlt.ThompsonSampling_FJLT import *

from algorithms.other.SOFUL import *
from algorithms.other.SOFUL_2m import *
from algorithms.other.CBSCFD import *

from algorithms.other.LRP import *
from algorithms.other.CBRAP import *
from algorithms.other.PCA import *
from algorithms.other.CBRAP_V0 import *
from algorithms.other.CBRAP_V1 import *





def select(name):
    return getattr(sys.modules[__name__], name)







