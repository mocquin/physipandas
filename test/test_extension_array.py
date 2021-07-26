import unittest
import numpy as np
import physipandas

from physipy import m
import pandas as pd


class TestClassQuantityArray(unittest.TestCase):


    @classmethod
    def setUp(cls):        
        arr = np.arange(10)
        arr[0]=2
        cls.arr = arr
        qarr = arr*m
        cls.qarr = qarr
        s = pd.Series(arr)
        cls.s = s
        sq = pd.Series(qarr, dtype='physipy[m]')
        cls.sq = sq

    def test_add(self):
        # add
        res = self.sq.add(1*m)
        exp = self.s.add(1)*m
        self.assertTrue(all(res==exp))
    def test_add_2(self):
        res = self.sq.add(self.sq)
        exp = self.s.add(self.s)*m
        self.assertTrue(all(res==exp))
    def test_add_3(self):
        res = self.sq.add(np.arange(10)*m)
        exp = self.s.add(np.arange(10))*m
        self.assertTrue(all(res==exp))
    def test_any(self):
        exp = self.s.any()
        res = self.sq.any()
        self.assertEqual(exp, res)
    def test_append(self):
        exp = self.s.append(self.s)*m
        res = self.sq.append(self.sq)
        self.assertTrue(all(res==exp))
    def test_all(self):
        # all elements are True
        res = self.s.all()
        exp = self.sq.all()
        self.assertEqual(exp, res)
    def test_argmax(self):
        # position of max value
        res = self.s.argmax()
        exp = self.sq.argmax()
        self.assertEqual(exp, res)
    def test_argmin(self):
        # position of min value
        res = self.s.argmin()
        exp = self.sq.argmin()
        self.assertEqual(exp, res)
    def test_autocorr(self):
        # autocorrelation
        res = self.sq.autocorr()
        exp = self.s.autocorr()
        self.assertEqual(exp, res)
    def test_dot(self):
        # dot
        exp = self.s.dot(self.s)*m**2
        res = self.sq.dot(self.sq)
        self.assertEqual(exp, res)
    def test_dot(self):
        exp = self.s.dot(self.s)*m
        res = self.sq.dot(self.s)
        self.assertEqual(exp, res)
    def test_dot(self):
        exp = self.s.dot(self.s)*m
        res = self.s.dot(self.sq)
        self.assertEqual(exp, res)
    def test_drop(self):
        # drop
        exp = self.s.drop_duplicates()*m
        res = self.sq.drop_duplicates()
        self.assertTrue(all(res==exp))
    def test_argsort(self):
        # sort index
        res = self.s.argsort()
        exp = self.sq.argsort()
        self.assertTrue(all(res==exp))
    def test_clip(self):
        # clip
        res = self.sq.clip(2*m, 5*m)
        exp = self.s.clip(2, 5)*m
        self.assertTrue(all(res==exp))
    def test_clip(self):
        res = self.sq.clip(3*m, 6*m)
        exp = self.s.clip(3, 6)*m
        print(all(res == exp))
    def test_corr(self):
        # correlation
        exp = self.s.corr(self.s)
        res = self.sq.corr(self.sq)
        self.assertEqual(exp, res)
    def test_corr(self):
        res = self.s.corr(self.sq)
        exp = self.s.corr(self.s)
        self.assertEqual(exp, res)
    def test_count(self):
        # count
        res = self.sq.count()
        exp = self.s.count()
        self.assertEqual(exp, res)
    def test_cov(self):
        # covariance
        res = self.sq.cov(self.sq)
        exp = self.s.cov(self.s)
        self.assertEqual(exp, res)
    def test_cov(self):
        res = self.sq.cov(self.s)
        exp = self.s.cov(self.s)
        self.assertEqual(exp, res)
    def test_cummax(self):
        #cummax : rolling max
        exp = self.s.cummax()*m
        res = self.sq.cummax()
        self.assertEqual(exp, res)
    def test_cummin(self):
        # cummin : rolling min
        exp = self.s.cummin()*m
        res = self.sq.cummin()
        self.assertEqual(exp, res)
    def test_cumprod(self):
        # cumprod : rolling prod
        exp = self.s.cumprod()
        res = self.sq.cumprod() #Fails because would lead to dimension (m, m**2, ...) in an array
        self.assertEqual(exp, res)
    def test_cumsum(self):
        #cumsum : rolling sum
        exp = self.s.cumsum()*m
        res = self.sq.cumsum()
        self.assertEqual(exp, res)
    def test_diff(self):
        pass
        #self.sq.diff()
    def test_eq(self):
        exp = self.s.eq(3)
        res = self.sq.eq(3*m)
        self.assertTrue(all(res==exp))
    def test_equals(self):
        exp = self.s.equals(self.s)
        res = self.sq.equals(self.sq)
        self.assertEqual(exp, res)
    def test_equals(self):
        exp = self.s.equals(self.s)
        res = self.sq.equals(self.s*m)
        self.assertEqual(exp, res) # fails because s*m returns a series of objects, not a QuantityArray
    def test_floordiv(self):
        exp = self.s.floordiv(2)*m
        res = self.sq.floordiv(2*m)
        self.assertTrue(all(res==exp))
    def test_ge(self):
        exp = self.s.ge(2)
        res = self.sq.ge(2*m)
        self.assertTrue(all(res==exp))
    def test_gt(self):
        exp = self.s.gt(2)
        res = self.sq.gt(2*m)
        self.assertTrue(all(res==exp))
    def test_iat(self):
        exp = self.s.iat[0]*m
        res = self.sq.iat[0]
        self.assertEqual(exp, res)
    def test_iat(self):
        exp = self.s.iat[0]*m
        res = self.sq.iat[0]
        self.assertEqual(exp, res)
    def test_idxmax(self):
        res = self.sq.idxmax()
        exp = self.s.idxmax()
        self.assertEqual(exp, res)
    def test_idxmin(self):
        res = self.sq.idxmin()
        exp = self.s.idxmin()
        self.assertEqual(exp, res)
    def test_is_monotonic(self):
        res = self.sq.is_monotonic
        exp = self.s.is_monotonic
        self.assertEqual(exp, res)
    def test_is_monotonic_decreasing(self):
        res = self.sq.is_monotonic_decreasing
        exp = self.s.is_monotonic_decreasing
        self.assertEqual(exp, res)
    def test_is_monotonic_increasing(self):
        res = self.sq.is_monotonic_increasing
        exp = self.s.is_monotonic_increasing
        self.assertEqual(exp, res)
    def test_is_unique(self):
        res = self.sq.is_unique
        exp = self.s.is_unique
        self.assertEqual(exp, res)
    def test_kurt(self):
        exp = self.s.kurt()
        res = self.sq.kurt()
        self.assertEqual(exp, res)
    def test_kurtosis(self):
        exp = self.s.kurtosis()
        res = self.sq.kurtosis()
        self.assertEqual(exp, res)
    def test_le(self):
        exp = self.s.le(2)
        res = self.sq.le(2*m)
        self.assertTrue(all(res==exp))
    def test_loc(self):
        res = self.sq.loc[3:5]
        exp = self.s.loc[3:5]*m
        self.assertTrue(all(res==exp))
    def test_lt(self):
        exp = self.s.lt(2)
        res = self.sq.lt(2*m)
        self.assertTrue(all(res==exp))
    def test_mad(self):
        exp = self.s.mad()*m
        res = self.sq.mad()
        self.assertEqual(exp, res)
    def test_max(self):
        exp = self.s.max()*m
        res = self.sq.max()
        self.assertEqual(exp, res)
    def test_mean(self):
        exp = self.s.mean()*m
        res = self.sq.mean()
        self.assertEqual(exp, res)
    def test_median(self):
        exp = self.s.median()*m
        res = self.sq.median()
        self.assertEqual(exp, res)
    def test_min(self):
        exp = self.s.min()*m
        res = self.sq.min()
        self.assertEqual(exp, res)
    def test_mod(self):
        exp = self.s.mod(3)*m
        res = self.sq.mod(3*m)
        self.assertTrue(all(res==exp))
    def test_mode(self):
        exp = self.s.mode(3)*m
        res = self.sq.mode(3*m)
        self.assertTrue(all(res==exp))
    def test_nunique(self):
        exp = self.s.nunique()
        res = self.sq.nunique()
        self.assertEqual(exp, res)
    #def test_nsmallest(self):
    #    Fails because a check is done on the dtype, and
    #    isinstance(np.dtype(sq.dtype.type).type, np.number) is False
    #    exp = self.s.nsmallest()
    #    res = self.sq.nsmallest()
    #    self.assertEqual(exp, res)
    #def test_nlargest(self):
    #    Fails because a check is done on the dtype, and
    #    isinstance(np.dtype(sq.dtype.type).type, np.number) is False
    #    exp = self.s.nlargest()*m
    #    res = self.sq.nlargest()
    #    self.assertEqual(exp, res)
    def test_pct_change(self):
        exp = self.s.pct_change()
        res = self.sq.pct_change()
        self.assertTrue(all(exp==res))
    def test_pop(self):
        scopy = self.s.copy()
        exp = scopy.pop(5)
        sq_copy = self.sq.copy()
        res = sq_copy.pop(5*m)
        self.assertTrue(all(res==exp))
    def test_pow(self):
        res = self.sq.pow(2)
        exp = self.s.pow(2)*m
        self.assertTrue(all(res==exp))
    def test_prod(self):
        res = self.sq.prod()
        exp = self.s.prod()*m**(len(self.s))
        self.assertEqual(exp, res)
    def test_product(self):
        res = self.sq.product()
        exp = self.s.product()*m**(len(self.s))
        self.assertEqual(exp, res)
    def test_quantile(self):
        res = self.sq.quantile()
        exp = self.s.quantile()*m
        self.assertEqual(exp, res)
    def test_radd(self):
        res = self.sq.radd(1*m)
        exp = self.s.radd(1)*m
        self.assertTrue(all(res==exp))
    def test_radd(self):
        res = self.sq.radd(self.sq)
        exp = self.s.radd(self.s)*m
        self.assertTrue(all(res==exp))
    def test_rank(self):
        res = self.sq.rank()
        exp = self.s.rank()
        self.assertTrue(all(res==exp))
    def test_ravel(self):
        res = self.sq.ravel()
        exp = self.s.ravel()*m
        self.assertTrue(all(res==exp))
    def test_rdiv(self):
        res = self.sq.rdiv(2)
        exp = self.s.rdiv(2)/m
        self.assertTrue(all(res==exp))
    def test_repeat(self):
        exp = self.s.repeat(2)*m
        res = self.sq.repeat(2)
        self.assertTrue(all(res==exp))
    def test_round(self):
        exp = self.s.round()*m
        res = self.sq.round()
        self.assertTrue(all(res==exp))
    def test_searchsorted(self):
        exp = self.s.searchsorted(5)
        res = self.sq.searchsorted(5*m)
        self.assertEqual(exp, res)
    def test_sem(self):
        res = self.sq.sem()
        exp = self.s.sem()
        self.assertEqual(exp, res)
    def test_shift(self):
        res = self.sq.shift()
        exp = self.s.shift()*m
        self.assertTrue(all(res==exp))
    def test_skew(self):
        res = self.sq.skew()
        exp = self.s.skew()
        self.assertEqual(exp, res)
    def test_sort_values(self):
        res = self.sq.sort_values()
        exp = self.s.sort_values()*m
        self.assertTrue(all(res==exp))
    def test_sort_index(self):
        res = self.sq.sort_index()
        exp = self.s.sort_index()*m
        self.assertTrue(all(res==exp))
    def test_std(self):
        res = self.sq.std()
        exp = self.s.std()*m
        self.assertEqual(exp, res)
    def test_sum(self):
        res = self.sq.sum()
        exp = self.s.sum()*m
        self.assertEqual(exp, res)
    def test_tail(self):
        res = self.sq.tail()
        exp = self.s.tail()*m
        self.assertTrue(all(res==exp))
    def test_truncate(self):
        res = self.sq.truncate(2, 8)
        exp = self.s.truncate(2, 8)*m
        self.assertTrue(all(res==exp))
    def test_unique(self):
        res = self.sq.unique()
        exp = self.s.unique()*m
        self.assertTrue(all(res==exp))
    def test_value_counts(self):
        res = self.sq.value_counts()
        exp = self.s.value_counts()
        self.assertEqual(exp, res)
    def test_var(self):
        res = self.sq.var()
        exp = self.s.var()*m**2
        self.assertEqual(exp, res)
    def test_xs(self):
        exp = self.s.xs(3)*m
        res = self.sq.xs(3)
        self.assertEqual(exp, res)
    def test_where(self):
        res = self.sq.where(self.sq>4*m)
        exp = self.s.where(self.s>4)*m
        self.assertEqual(exp, res)
    def test_abs(self):
        # asbolute value
        res = self.sq.abs()
        exp = pd.Series(self.s.abs()*m, dtype='physipy[m]')
        self.assertTrue(all(res==exp))

        
        
if __name__ == "__main__":
    unittest.main()