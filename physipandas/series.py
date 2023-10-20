from pandas.api.extensions import register_series_accessor

from .extension import QuantityDtype


@register_series_accessor("physipy")
class PhysipySeriesAccessor(object):
    """
    TODO : could remove the validate and rely on quantity conversion.
    Eg, if the series is numerical could be used to automaticaly quantify.
    
    Series accessor for accesing physipy methods on the series' quantity value.
    Only work on physipy dtype series.
    
    Examples : 
    =========
    import pandas as pd
    from physipy import m
    from physipandas import QuantityDtype
    
    c = pd.Series(np.arange(10)*m, 
                  dtype=QuantityDtype(m))
    try:
        print(c.dimension)
    except Exception as e:
        print("Raised ", e)
        print(c.physipy.dimension)
    try:
        print(c._SI_unitary_quantity)
    except Exception as e:
        print("Raised ", e)
        print(c.physipy._SI_unitary_quantity)
    try:
        print(c.mean())
    except Exception as e:
        print("Raised ", e)
        print(c.physipy.values.mean())
        
    c.physipy.values.is_length()

        
    Raised  'Series' object has no attribute 'dimension'
    L
    Raised  'Series' object has no attribute '_SI_unitary_quantity'
    1 m
    Raised  cannot perform mean with type physipy[m]
    4.5 m
    True
    
    """
    def __init__(self, pandas_obj):
        # anything that shouldn't be used through the accessor should be made private
        self._validate(pandas_obj)
        # self._pandas_obj = pandas_obj
        self._quantity = pandas_obj.values.quantity
        #self.dimension = pandas_obj.values.dimension
        #self.SI_unitary_quantity = pandas_obj.values.quantity._SI_unitary_quantity
        #self._index = pandas_obj.index
        #self._name = pandas_obj.name
    
    @property
    def dimension(self):
        return self._quantity.dimension
    @property
    def quantity(self):
        return self._quantity

    def to_quantity(self):
        return self._quantity 

    @property
    def SI_unitary_quantity(self):
        return self._quantity._SI_unitary_quantity

    @staticmethod
    def _validate(obj):
        if not is_physipy_type(obj):
            raise AttributeError(
                "Cannot use 'physipy' accessor on objects of "
                "dtype '{}'.".format(obj.dtype)
            )

def is_physipy_type(obj):
    t = getattr(obj, "dtype", obj)
    try:
        return isinstance(t, QuantityDtype) or issubclass(t, QuantityDtype)
    except Exception:
        return False