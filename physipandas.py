import physipy
from physipy import Quantity, Dimension, quantify, units, DimensionError
from physipy.quantity.utils import asqarray

import pandas as pd
import re

import numpy as np

from pandas import DataFrame
from pandas.core.arrays import ExtensionArray
from pandas.core.dtypes.base import ExtensionDtype

from pandas.api.types import is_integer, is_list_like, is_object_dtype, is_string_dtype
from pandas.compat import set_function_name
from pandas.core.arrays.base import ExtensionOpsMixin
from pandas.core.indexers import check_array_indexer
from pandas.api.extensions import register_extension_dtype, register_dataframe_accessor, register_series_accessor


# This enables operations like ``.astype('toto')`` for the name
# of the ExtensionDtype. Example : 
# register_extension_dtype
# class MyExtensionDtype(ExtensionDtype):
#     name = "myextension"
@register_extension_dtype
class QuantityDtype(ExtensionDtype):
    """A custom data type, to be paired with an ExtensionArray.
    This basically wraps a dimension using a unitary quantity.
    Hence type = Quantity
    """

    # each instance of QuantityDtype has a "unit" attribute that is 
    # a Quantity object, usually with 1 value, only the dimension will be used
    # It has a "name" attribute that is the dimension string
    # and a "dimension" attribute, a Dimension Object
    
    # The construct_array_type method gives the object to call
    # when creating a padnas object, with signature
    # self, values, dtype=None, copy=False)
    
    type = Quantity
    #name = "quantity" # see register_extension_dtype at the end
    _metadata = ("unit",)
    # for construction from string
    _match = re.compile(r"physipy\[(?P<unit>.+)\]")
    
    # this will be used by QuantityArray when accessing indexes that do not
    # exist (see .take)
    na_value = Quantity(np.nan, Dimension(None))

    @classmethod
    def construct_array_type(cls):
        """Return the array type associated with this dtype."""
        return QuantityArray
    
    
    #############################
    #######  To print the unit under the values when repr-ed
    def __new__(cls, unit=None):
        print("QDTYPE : new with", unit, "of type", type(unit))
        if isinstance(unit, QuantityDtype):
            print("QDTYPE : new already a QDTYPE, returning it", unit)
            return unit
        elif isinstance(unit, str):
            unit = cls._parse_dtype_strict(unit)
        elif unit is None:
            print("QDTYPE : new is None, using quantify(1)")
            unit = quantify(1)
            
        if isinstance(unit, Quantity):
            #qdtype_unit = QuantityDtype(unit)
            u = object.__new__(cls)
            u.unit = unit
            print("returning ", u, "with .unit", u.unit)
            return u
        else:
            raise ValueError

    
    @property
    def name(self):
        """
        What to print below 
        
            df["quanti"].values
            df["quanti"].dtype
            df["quanti"]
        
        """
        return f"physipy[{self.unit.dimension.str_SI_unit()}]"
    
    @property
    def dimension(self):
        return self.unit.dimension
    
    def __repr__(self):
        """
        Return a string representation for this object.
        Invoked by unicode(df) in py2 only. Yields a Unicode String in both
        py2/py3.
        """
        return self.name
    
    ####################
    
    ###### Construction from string, to allow both series.astype("quantity")
    #### and pretty printing
    @classmethod
    def _parse_dtype_strict(cls, string_unit):
        """
        Parses the unit, which should be a string like 
            'physipy[ANYTHIN]'
        """
        eval_dict_units = physipy.units

        if isinstance(string_unit, str):
            if string_unit.startswith("physipy["):# or units.startswith("Pint["):
                if not string_unit[-1] == "]":
                    raise ValueError("could not construct QuantityDtype")
                m = cls._match.search(string_unit)
                if m is not None:
                    string_unit = m.group("unit")
                    actual_unit_quantity = eval_dict_units[string_unit]
            if actual_unit_quantity is not None:
                return actual_unit_quantity

        raise ValueError("could not construct QuantityDtype")

    @classmethod
    def construct_from_string(cls, string):
        """
        Strict construction from a string, raise a TypeError if not
        possible
        """
        print("QDTYPE : construct_from_string with", string)
        eval_dict_units = physipy.units
        
        if not isinstance(string, str):
            raise TypeError(
                f"'construct_from_string' expects a string, got {type(string)}"
            )
        if isinstance(string, str) and (
            string.startswith("physipy[")# or string.startswith("Pint[")
        ):
            # do not parse string like U as pint[U]
            # avoid tuple to be regarded as unit
            try:
                actual_unit_quantity = cls._parse_dtype_strict(string)
                print("QDTYPE : construct_from_string, actual unit found", actual_unit_quantity, f". Returning QuantityDtype({actual_unit_quantity})")
                return cls(unit=actual_unit_quantity)
            except ValueError:
                pass
        raise TypeError(f"Cannot construct a 'QuantityType' from '{string}'")

    @classmethod
    def construct_from_quantity_string(cls, string):
        """
        For use in QuantityArray
         
        Strict construction from a string, raise a TypeError if not
        possible
        """
        print("QDTYPE : construct_from_quantity_string")
        if not isinstance(string, str):
            raise TypeError(
                f"'construct_from_quantity_string' expects a string, got {type(string)}"
            )

        quantity = cls.ureg.Quantity(string)
        return cls(unit=quantity.unit)
    

    
from pandas.core.arrays.base import ExtensionOpsMixin
# for ExtensionOpsMixin
from pandas.api.types import is_list_like


class QuantityArray(ExtensionArray,
                    # for operations between series for ex
                    # see https://github.com/pandas-dev/pandas/blob/21d61451ca363bd4c67dbf284b29daacc30302b1/pandas/core/arrays/base.py#L1330
                    # must define _create_arithmetic_method(cls, op) for ex
                   ExtensionOpsMixin):
    """Abstract base class for custom 1-D array types."""

    def __init__(self, values, dtype=None, copy=False):
        """Instantiate the array.
        If you're doing any type coercion in here, you will also need
        that in an overwritten __settiem__ method.
        But, here we coerce the input values into Decimals.
        """
        # check if we have a list like [<Quantity:[1, 2, 3], m>]
        if isinstance(values, list) and len(values) == 1 and isinstance(values[0], Quantity):
            values = values[0]
        print("QARRAY : init with", values, 'of type', type(values))
        values = quantify(values)
        print("QARRAY : init quantified values ", values, "with len", len(values))
        self._data = values
        # Must pass the dimension to create a "custom" QuantityDtype, that displays with the proper unit
        #self._dtype = QuantityDtype()
        if dtype is None:
            dtype = QuantityDtype(values._SI_unitary_quantity)
        else:
            if isinstance(dtype, QuantityDtype) or isinstance(dtype, Quantity):
                if dtype.dimension != values.dimension:
                    raise DimensionError(dtype.dimension, values.dimension)
            dtype = QuantityDtype(values._SI_unitary_quantity)
        self._dtype = dtype

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        """Construct a new ExtensionArray from a sequence of scalars."""
        #values = asarray(scalars)
        values = asqarray(scalars)
        return cls(scalars, dtype=dtype)

    #@classmethod
    #def _from_factorized(cls, values, original):
    #    """Reconstruct an ExtensionArray after factorization."""
    #    return cls(values)

    def __getitem__(self, item):
        """Select a subset of self.
        
        Called by : print(df["quanti"].values)
        """
        #return self._data[item]
        if is_integer(item):
            return self._data[item]# * self.units

        item = check_array_indexer(self, item)

        return self.__class__(self._data[item], self.dtype)
    
    def __setitem__(self, key, value):
        print("in setitem")
        

    def __len__(self) -> int:
        """Length of this array."""
        return len(self._data)

    @property
    def nbytes(self):
        """The byte size of the data."""
        return self._itemsize * len(self)

    @property
    def dtype(self):
        """An instance of 'ExtensionDtype'."""
        return self._dtype


    def _formatter(self, boxed=False):
        """Formatting function for scalar values.
        This is used in the default '__repr__'. The returned formatting
        function receives scalar Quantities.
        # type: (bool) -> Callable[[Any], Optional[str]]
        Parameters
        ----------
        boxed: bool, default False
            An indicated for whether or not your array is being printed
            within a Series, DataFrame, or Index (True), or just by
            itself (False). This may be useful if you want scalar values
            to appear differently within a Series versus on its own (e.g.
            quoted or not).
        Returns
        -------
        Callable[[Any], str]
            A callable that gets instances of the scalar type and
            returns a string. By default, :func:`repr` is used
            when ``boxed=False`` and :func:`str` is used when
            ``boxed=True``.
        """


        def formatting_function(quantity):
            return "{}".format(
                quantity
            )

        return formatting_function
    
    
    def isna(self):
        """A 1-D array indicating if each value is missing."""
        #return np.array([x.is_nan() for x in self._data], dtype=bool)
        return self._data.is_nan() # Quantity implements this directly
        
    def take(self, indexer, allow_fill=False, fill_value=None):
        """Take elements from an array.
        Relies on the take method defined in pandas:
        https://github.com/pandas-dev/pandas/blob/e246c3b05924ac1fe083565a765ce847fcad3d91/pandas/core/algorithms.py#L1483
        """
        from pandas.api.extensions import take
        print("into take")

        data = self._data
        if allow_fill and fill_value is None:
            fill_value = self.dtype.na_value

        result = take(
            data, indexer, fill_value=fill_value, allow_fill=allow_fill)
        return self._from_sequence(result)

    def copy(self):
        """Return a copy of the array.
        
        Used on :
        res = df["a"] + df["b"]
        df["c"] = res
        """
        print("into copy")
        print("data is ", self._data.copy())
        print("dtype is", self._dtype)
        return type(self)(self._data.copy(), self._dtype)

    @classmethod
    def _concat_same_type(cls, to_concat):
        """Concatenate multiple arrays."""
        print("QARRAY : _concat_same_type with", to_concat)
        return cls(np.concatenate([x._data for x in to_concat]))
    
    
    
    
    ###################################
    ##### ExtensionOpsMixin
    ###################################
    @property
    def quantity(self):
        return self._data
    
    @property
    def dimension(self):
        return self._data.dimension
    
    @classmethod
    def _create_method(cls, op, coerce_to_dtype=True):
        """
        A class method that returns a method that will correspond to an
        operator for an ExtensionArray subclass, by dispatching to the
        relevant operator defined on the individual elements of the
        ExtensionArray.
        Parameters
        ----------
        op : function
            An operator that takes arguments op(a, b)
        coerce_to_dtype :  bool
            boolean indicating whether to attempt to convert
            the result to the underlying ExtensionArray dtype
            (default True)
        Returns
        -------
        A method that can be bound to a method of a class
        Example
        -------
        Given an ExtensionArray subclass called MyExtensionArray, use
        >>> __add__ = cls._create_method(operator.add)
        in the class definition of MyExtensionArray to create the operator
        for addition, that will be based on the operator implementation
        of the underlying elements of the ExtensionArray
        """
        from pandas.compat import set_function_name

        def _binop(self, other):
            def validate_length(obj1, obj2):
                # validates length and converts to listlike
                try:
                    if len(obj1) == len(obj2):
                        return obj2
                    else:
                        raise ValueError("Lengths must match")
                except TypeError:
                    return [obj2] * len(obj1)

            def convert_values(param):
                # convert to a quantity or listlike
                if isinstance(param, cls):
                    return param.quantity
                elif isinstance(param, Quantity):
                    return param
                elif is_list_like(param) and isinstance(param[0], Quantity):
                    return type(param[0])([p.value for p in param], param[0].dimension)
                else:
                    return param

            if isinstance(other, (pd.Series, pd.DataFrame)):
                return NotImplemented
            lvalues = self.quantity
            other = validate_length(lvalues, other)
            rvalues = convert_values(other)
            # Pint quantities may only be exponented by single values, not arrays.
            # Reduce single value arrays to single value to allow power ops
            if isinstance(rvalues, Quantity):
                if len(set(np.array(rvalues.data))) == 1:
                    rvalues = rvalues[0]
            elif len(set(np.array(rvalues))) == 1:
                rvalues = rvalues[0]
            # If the operator is not defined for the underlying objects,
            # a TypeError should be raised
            res = op(lvalues, rvalues)

            if op.__name__ == "divmod":
                return (
                    cls.from_1darray_quantity(res[0]),
                    cls.from_1darray_quantity(res[1]),
                )

            if coerce_to_dtype:
                try:
                    #res = cls.from_1darray_quantity(res)
                    # enforce a Quantity object when a calculation leads to dimensionless
                    # hence physipy returns an array, not a dimensionless quantity (df["a"]/df["a"])
                    res = cls.from_1darray_quantity(quantify(res))
                except TypeError:
                    pass

            return res

        op_name = f"__{op}__"
        return set_function_name(_binop, op_name, cls)

    @classmethod
    def _create_arithmetic_method(cls, op):
        #return cls._create_method(op)
        return cls._create_method(op, coerce_to_dtype=True)

    @classmethod
    def _create_comparison_method(cls, op):
        return cls._create_method(op, coerce_to_dtype=False)
    
    @classmethod
    def from_1darray_quantity(cls, quantity):
        if not is_list_like(quantity.value):
            raise TypeError("quantity's magnitude is not list like")
        return cls(quantity)
    
    #################### END ExtensionOpsMixin
    
    
    ##########################
    ##### For repr : must be convertible to numpy array
    def astype(self, dtype, copy=True):
        """Cast to a NumPy array with 'dtype'.
        Parameters
        ----------
        dtype : str or dtype
            Typecode or data-type to which the array is cast.
        copy : bool, default True
            Whether to copy the data, even if not necessary. If False,
            a copy is made only if the old dtype does not match the
            new dtype.
        Returns
        -------
        array : ndarray
            NumPy ndarray with 'dtype' for its dtype.
        """
        #if isinstance(dtype, str) and (
        #    dtype.startswith("physipy[")):
        #    dtype = QuantityDtype(dtype)
        if isinstance(dtype, QuantityDtype):
            if dtype == self._dtype and not copy:
                return self
            else:
                return QuantityArray(self.quantity.value, dtype)
        elif isinstance(dtype, Quantity):
            return QuantityArray(self.quantity.value, QuantityDtype(dtype))
        #return self.__array__(dtype, copy)
        return Quantity(self.quantity.value.astype(dtype), self.quantity.dimension)
    
    def __array__(self, dtype=None, copy=False):
        #if dtype is None or is_object_dtype(dtype):
        #    return self._to_array_of_quantity(copy=copy)
        #if (isinstance(dtype, str) and dtype == "string") or isinstance(
        #    dtype, pd.StringDtype
        #):
        #    return pd.array([str(x) for x in self.quantity], dtype=pd.StringDtype())
        #if is_string_dtype(dtype):
        #    return np.array([str(x) for x in self.quantity], dtype=str)
        return np.array(self._data.value, dtype=dtype, copy=copy)

    
# FOR ExtensionOpsMixin to work    
QuantityArray._add_arithmetic_ops()
QuantityArray._add_comparison_ops()



@register_dataframe_accessor("physipy")
class PhysipyDataFrameAccessor(object):
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def quantify(self, level=-1):
        df = self._obj
        df_columns = df.columns.to_frame()
        unit_col_name = df_columns.columns[level]
        units = df_columns[unit_col_name]
        df_columns = df_columns.drop(columns=unit_col_name)

        df_new = DataFrame(
            {i: QuantityArray(df.values[:, i], unit) for i, unit in enumerate(units.values)}
        )

        df_new.columns = df_columns.index.droplevel(unit_col_name)
        df_new.index = df.index

        return df_new

    def dequantify(self):
        def formatter_func(units):
            #formatter = "{:" + units._REGISTRY.default_format + "}"
            #formatter = "{:"+str(units.str_SI_unit())+"}"
            formatter = "{:}"
            return formatter.format(units)

        df = self._obj

        df_columns = df.columns.to_frame()
        df_columns["units"] = [
            formatter_func(df[col].values.dimension) for col in df.columns
        ]
        from collections import OrderedDict

        data_for_df = OrderedDict()
        for i, col in enumerate(df.columns):
            data_for_df[tuple(df_columns.iloc[i])] = df[col].values._data
        df_new = DataFrame(data_for_df, columns=data_for_df.keys(),
                          index=range(len(data_for_df)))

        df_new.columns.names = df.columns.names + ["unit"]
        df_new.index = df.index

        return df_new

    def to_base_units(self):
        obj = self._obj
        df = self._obj
        index = object.__getattribute__(obj, "index")
        # name = object.__getattribute__(obj, '_name')
        return DataFrame(
            {col: df[col].physipy.to_base_units() for col in df.columns}, index=index
        )


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
        self._validate(pandas_obj)
        self.pandas_obj = pandas_obj
        self.values = pandas_obj.values.quantity
        self.dimension = pandas_obj.values.dimension
        self._SI_unitary_quantity = pandas_obj.values.quantity._SI_unitary_quantity
        self._index = pandas_obj.index
        self._name = pandas_obj.name

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