import unittest
import numpy as np
import physipandas

from physipy import m, Quantity, Dimension, units
W = units["W"]
import pandas as pd
from physipandas import QuantityDtype, QuantityArray


class TestClassQuantityDtype(unittest.TestCase):


    @classmethod
    def setUp(cls):        
        arr = np.arange(10)
        arr[0]=2
        cls.arr = arr
        qarr = arr*m
        cls.qarr = qarr
        s = pd.Series(arr)
        cls.s = s
        #sq = pd.Series(qarr, dtype='physipy[m]')
        #cls.sq = sq

    def test_type(self):
        self.assertEqual(QuantityDtype.type, Quantity)
        
    def test_creation_from_None(self):
        self.assertEqual(QuantityDtype().unit, Quantity(1, Dimension(None)))
        
    def test_creation_from_Q(self):
        self.assertEqual(QuantityDtype(m).unit, m)
    def test_creation_from_Q(self):
        self.assertEqual(QuantityDtype(W).unit, W)
    
        
        
    def test_creation_SI_unit_from_string(self):
        QuantityDtype("physipy[m]")
        
    def test_creation_SI_unit2_from_string(self):
        QuantityDtype("physipy[m**2]")
        
    def test_creation_dimless_from_string(self):
        QuantityDtype("physipy[]")
        
    def test_is_numeric(self):
        self.assertTrue(QuantityDtype._is_numeric)
        
    def test_name_SI_unit(self):
        self.assertEqual(QuantityDtype("physipy[m]"), "physipy[m]")
    def test_name_SI_unit2(self):
        self.assertEqual(QuantityDtype("physipy[m**2]"), "physipy[m**2]")
    def test_name_dimesless(self):
        self.assertEqual(QuantityDtype("physipy[]"), "physipy[]")
    def test_name_W(self):
        self.assertEqual(QuantityDtype("physipy[W]"), "physipy[W]")
    
    
    def test_dim_SI_unit(self):
        self.assertEqual(QuantityDtype("physipy[m]").dimension, Dimension("L"))
    def test_dim_SI_unit2(self):
        self.assertEqual(QuantityDtype("physipy[m**2]").dimension, Dimension("L**2"))
    def test_dim_dimesless(self):
        self.assertEqual(QuantityDtype("physipy[]").dimension, Dimension(None))
    def test_dim_W(self):
        self.assertEqual(QuantityDtype("physipy[W]").dimension, W.dimension)
        
    
    def test_array_twin(self):
        self.assertEqual(QuantityDtype.construct_array_type(), QuantityArray)
    
    
    def test_na_value(self):
        # make sure the na_value has same dimension
        self.assertEqual(QuantityDtype("physipy[m]").na_value.dimension, Dimension("L"))
        # make sure the value is a nan
        self.assertTrue(np.isnan(QuantityDtype("physipy[m]").na_value))
        