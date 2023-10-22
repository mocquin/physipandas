import unittest
import numpy as np
import physipandas

from physipy import m, s, units, Quantity, Dimension
import pandas as pd
from physipandas import QuantityDtype, QuantityArray

W = units["W"]

class TestClassQuantityArray(unittest.TestCase):

    def test_creation(self):
        qa = QuantityArray(np.arange(10)*m)
        self.assertEqual(qa.dtype, QuantityDtype(m))
        
    def test_creation_nonbasic(self):
        qa = QuantityArray(np.arange(10)*W)
        self.assertEqual(qa.dtype, QuantityDtype(W))
        
    def test_len(self):
        qa = QuantityArray(np.arange(10)*W)
        self.assertEqual(len(qa), 10)
        
    def test_eq(self):
        qa = QuantityArray(np.arange(10)*W)
        s = pd.Series(np.arange(10))
        i = pd.Index(np.arange(10))
        df = pd.DataFrame()
        # these first go throug __eq__ that returns NotImplemented, then pandas
        # extracts the actual value
        #self.assertFalse(np.all(qa == s))
        self.assertFalse(np.all(qa == i))
        #self.assertRaises(qa == df)
        #self.assertRaises(pd.Series() == qa)
        #self.assertRaises(pd.Index() == qa)
        #self.assertRaises(pd.DataFrame() == qa)
        
        
    def test_dtype(self):
        qa = QuantityArray(np.arange(10)*W)
        self.assertEqual(qa.dtype, QuantityDtype(W))
        
    def test_isna(self):
        qa = QuantityArray(np.arange(10)*W)
        self.assertTrue(type(qa.isna())==np.ndarray)
        self.assertTrue(np.all(qa.isna()==np.zeros(10)))
        
        
    def test_unique(self):
        qa = QuantityArray(np.arange(10)*W)
        self.assertTrue(np.all(qa==qa.unique()))
                           
    def test_from_sequence(self):
        qa = QuantityArray(np.arange(10)*W)
        self.assertTrue(np.all(qa == QuantityArray._from_sequence([i*W for i in range(10)])))
    
    def test_take(self):
        qa = QuantityArray(np.arange(10)*W)        
        self.assertTrue(np.all(qa.take([1,2,3])==QuantityArray([1,2,3]*W)))
        
    def test_concat_same_type(self):
        qa = QuantityArray(np.arange(10)*W)        
        # same dimension is ensured by pandas using Dtype before calling
        to_concat = [qa, qa]
        concat = QuantityArray._concat_same_type(to_concat)
        self.assertTrue(np.all(concat==QuantityArray(np.concatenate([np.arange(10), np.arange(10)])*W)))      
        
        
    def test_get_item(self):
        qa = QuantityArray(np.arange(10)*W)        
        
        self.assertEqual(qa[3], 3*W)
        self.assertTrue(np.all(qa[:]==qa))
        get = np.array([True, False, True, False, True,
                        False, True, False, True, False])
        self.assertTrue(np.all(qa[get] == QuantityArray([0, 2, 4, 6, 8]*W)))
        
        
    def test_argsort(self):
        qa = QuantityArray(np.arange(10)[::-1]*W)        
        self.assertTrue(np.all(np.arange(10)[::-1] == qa.argsort()))
        
    def test_dropna(self):
        qa = QuantityArray(np.arange(10)[::-1]*W)
        qa[5] = np.nan*W
        self.assertTrue(np.all(qa.dropna()==QuantityArray([0, 1, 2, 3, 5, 6, 7, 8, 9][::-1]*W)))
        
    def test_tolist(self):
        qa = QuantityArray(np.arange(10)*W)        
        self.assertTrue(np.all(qa.tolist()==[i*W for i in range(10)]))
        
    def test_shift(self):
        qa = QuantityArray(np.arange(10)*W)   
        exp = QuantityArray(np.arange(-1,9)*W)
        exp[0] = np.nan*W
        self.assertTrue(np.all(qa.shift()[1:]==exp[1:]))
        self.assertTrue(qa.shift()[0].is_nan())
        
        
    def test_equals(self):
        qa = QuantityArray(np.arange(10)*W)   
        self.assertTrue(np.all(qa.equals(qa)))

    def test_clip(self):
        s_ = np.arange(10)
        s = pd.Series(s_)
        sq = pd.Series(s_*m, dtype='physipy[m]')

    def test_shift(self):
        s_ = np.arange(10)
        s = pd.Series(s_)
        sq = pd.Series(s_*m, dtype='physipy[m]')
        sq.shift(2)

    def test_le(self):
        s_ = np.arange(10)
        s = pd.Series(s_)
        sq = pd.Series(s_*m, dtype='physipy[m]')
        exp = s.le(2)
        res = sq.le(2*m)
        self.assertTrue((exp==res).all())

    def test_std(self):
        s_ = np.arange(10)
        s = pd.Series(s_)
        sq = pd.Series(s_*m, dtype='physipy[m]')
        
        res = sq.std()
        exp = s.std()*m
        self.assertEqual(res, exp)

    def test_abs(self):
        s_ = np.arange(10)
        s = pd.Series(s_)
        sq = pd.Series(s_*m, dtype='physipy[m]')
        res = (-1*sq).abs()
        self.assertTrue((sq==res).all())