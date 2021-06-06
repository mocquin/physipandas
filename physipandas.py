import physipy
from physipy import Quantity, Dimension, quantify, units
from physipy.quantity.utils import asqarray

import pandas as pd
m = units["m"]
s = units["s"]


import decimal

import numpy as np

from pandas.core.arrays import ExtensionArray
from pandas.core.dtypes.base import ExtensionDtype


class QuantityDtype(ExtensionDtype):
    """A custom data type, to be paired with an ExtensionArray.
    
    This basically wraps a dimension using a unitary quantity.
    Hence type = Quantity
    """

    type = Quantity
    name = "quantitydtype"
    _metadata = ("unit",)
    na_value = Quantity(np.nan, Dimension(None))

    @classmethod
    def construct_array_type(cls):
        """Return the array type associated with this dtype."""
        return QuantityArray
    
    
    #############################
    #######  To print the unit under the values when repr-ed
    
    def __new__(cls, unit=None):
        print(f"New dtype with :{unit}")
        if isinstance(unit, QuantityDtype):
            return unit
        elif isinstance(unit, Quantity):
            #qdtype_unit = QuantityDtype(unit)
            u = object.__new__(cls)
            u.unit = unit
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
    
    ####################
    

    
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
        values = quantify(values)
        self._data = values
        # Must pass the dimension to create a "custom" QuantityDtype, that displays with the proper unit
        #self._dtype = QuantityDtype()
        self._dtype = QuantityDtype(values._SI_unitary_quantity)

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        """Construct a new ExtensionArray from a sequence of scalars."""
        values = asarray(scalars)
        return cls(scalars, dtype=dtype)

    @classmethod
    def _from_factorized(cls, values, original):
        """Reconstruct an ExtensionArray after factorization."""
        return cls(values)

    def __getitem__(self, item):
        """Select a subset of self.
        
        Called by : print(df["quanti"].values)
        """
        return self._data[item]

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
        return np.array([x.is_nan() for x in self._data], dtype=bool)

    def take(self, indexer, allow_fill=False, fill_value=None):
        """Take elements from an array.
        Relies on the take method defined in pandas:
        https://github.com/pandas-dev/pandas/blob/e246c3b05924ac1fe083565a765ce847fcad3d91/pandas/core/algorithms.py#L1483
        """
        from pandas.api.extensions import take

        data = self._data
        if allow_fill and fill_value is None:
            fill_value = self.dtype.na_value

        result = take(
            data, indexer, fill_value=fill_value, allow_fill=allow_fill)
        return self._from_sequence(result)

    def copy(self):
        """Return a copy of the array."""
        return type(self)(self._data.copy())

    @classmethod
    def _concat_same_type(cls, to_concat):
        """Concatenate multiple arrays."""
        return cls(np.concatenate([x._data for x in to_concat]))
    
    
    
    
    
    ###################################
    ##### ExtensionOpsMixin
    ###################################
    @property
    def quantity(self):
        return self._data
    
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
        return cls._create_method(op)

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
        print(f"astype not caught : {dtype}")
        return self.__array__(dtype, copy)
    
    def __array__(self, dtype=None, copy=False):
        #if dtype is None or is_object_dtype(dtype):
        #    return self._to_array_of_quantity(copy=copy)
        #if (isinstance(dtype, str) and dtype == "string") or isinstance(
        #    dtype, pd.StringDtype
        #):
        #    return pd.array([str(x) for x in self.quantity], dtype=pd.StringDtype())
        #if is_string_dtype(dtype):
        #    return np.array([str(x) for x in self.quantity], dtype=str)
        print("into array")
        return np.array(self._data.value, dtype=dtype, copy=copy)

    
# FOR ExtensionOpsMixin to work    
QuantityArray._add_arithmetic_ops()
QuantityArray._add_comparison_ops()
    


#quantity_series = pd.Series(QuantityArray([0.1, 0.2, 0.3]*m))
#df = pd.DataFrame({
#    "quanti":quantity_series,
#    "q":pd.Series(QuantityArray(np.arange(3)*s)),
#    "qq": QuantityArray([2,.3, 4], dtype=m),
#})
#print(df)