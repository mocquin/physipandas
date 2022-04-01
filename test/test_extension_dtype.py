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

    ## Inherited attributes/methods
    def test_kind(self):
        # see https://github.com/pandas-dev/pandas/blob/06d230151e6f18fdb8139d09abf539867a8cd481/pandas/core/dtypes/base.py#L167
        self.assertEqual(QuantityDtype().kind, "O")
    
    def test_holdna(self):
        self.assertEqual(QuantityDtype()._can_hold_na, True)
    
    def test_is_boolean(self):
        self.assertFalse(QuantityDtype()._is_boolean)
        
    def test_names(self):
        self.assertEqual(None, QuantityDtype().names)
        
    ## Actualy implemented by physipandas
    def test_type(self):
        self.assertEqual(QuantityDtype.type, Quantity)
        
    def test_creation_from_None(self):
        self.assertEqual(QuantityDtype().unit, Quantity(1, Dimension(None)))
        
    def test_creation_from_Q(self):
        self.assertEqual(QuantityDtype(m).unit, m)
    def test_creation_from_Q(self):
        self.assertEqual(QuantityDtype(W).unit, W)
        
    def test_is_dtype(self):
        self.assertTrue(QuantityDtype.is_dtype(QuantityDtype()))
    def test_is_dtype2(self):
        self.assertTrue(QuantityDtype.is_dtype(QuantityDtype(m)))
    def test_is_dtype3(self):
        self.assertFalse(QuantityDtype.is_dtype(m))

        

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
        