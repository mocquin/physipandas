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



@register_extension_dtype
class QuantityDtype(ExtensionDtype):
    """A custom data type, to be paired with an ExtensionArray.
    This basically wraps a dimension using a unitary quantity.

    For guidelines on how to write this class : 
    https://pandas.pydata.org/pandas-docs/stable/reference/api/
    pandas.api.extensions.ExtensionDtype.html
    
    We subclass from https://github.com/pandas-dev/pandas/blob/
    f00ed8f47020034e752baf0250483053340971b0/pandas/core/dtypes/base.py#L35
    
    The interface includes the following abstract methods that must be implemented by subclasses:
     - [X] : type : The scalar type for the array, ut’s expected ExtensionArray[item] returns an
     instance of ExtensionDtype.type for scalar item : hence we use Quantity
     - [X] : name : property that returns the string 
     f"physipy[{self.unit.dimension.str_SI_unit()}]", a string identifying the data type.
     Will be used for display in, e.g. Series.dtype
     - [X] : construct_array_type : return QuantityArray


    The following attributes and methods influence the behavior of the dtype in pandas operations
     - [X] : _is_numeric : returns True for now, but should it since we a
     re not a plain number ?. If not overriden , returns False by inheritance of ExtensionDtype
     - [X] : _is_boolean : returns False by inheritance of ExtensionDtype
     - [X] : _get_common_dtype : for now inherite from ExtensionDtype at
     https://github.com/pandas-dev/pandas/blob/f00ed8f47020034e752baf0250483053340971b0/
     pandas/core/dtypes/base.py#L335

    The na_value class attribute can be used to set the default NA value for this type.
    numpy.nan is used by default.
     - [X] : na_value : we overide this with na_value = Quantity(np.nan, Dimension(None))


    ExtensionDtypes are required to be hashable. The base class provides a default implementation,
    which relies on the _metadata class attribute. _metadata should be a tuple containing the 
    strings that define your data type. For example, with PeriodDtype that’s the freq attribute.
    If you have a parametrized dtype you should set the ``_metadata`` class property.
    Ideally, the attributes in _metadata will match the parameters to your ExtensionDtype.__init__ 
    (if any). If any of the attributes in _metadata don’t implement the standard __eq__ or __hash__, 
    the default implementations here will not work.
    - [X] : _metadata : QuantityDtype are parametrized by a physical quantity, so 
    we rely on the hash of the quantity to hash the Dtype.
    
    
    Methods
     - [X] : construct_array_type() : Return the array type associated with this dtype : QuantityArray
     - [X] : construct_from_string(string) : Construct this type from a string. See 
     [the doc of ExtensionDtype.construct_from_string]
     (https://pandas.pydata.org/pandas-docs/stable/reference
     /api/pandas.api.extensions.ExtensionDtype.construct_from_string.
     html#pandas.api.extensions.ExtensionDtype.construct_from_string.)
    This is useful mainly for data types that accept parameters. For example, a period dtype 
    accepts a frequency parameter that can be set as period[H] (where H means hourly frequency).
    For this we use a string parsing of the style `physipy[m]` for meter.
     - [X] : is_dtype(dtype) : Check if we match ‘dtype’. For now we use the default behaviour 
     given [here](https://pandas.pydata.org/pandas-docs/stable/reference
     /api/pandas.api.extensions.ExtensionDtype.is_dtype.html#pandas.api.extensions.ExtensionDtype.is_dtype).
    
     - kind [X] : use the default from inheritance : This should match the NumPy dtype used when the array is
        converted to an ndarray, which is probably 'O' for object if
        the extension type cannot be represented as a built-in NumPy
        type. See : https://github.com/pandas-dev/pandas/blob/
        f00ed8f47020034e752baf0250483053340971b0/pandas/core/dtypes/base.py#L163

    """
    
    # The scalar type for the array, it’s expected ExtensionArray[item] returns an
    # instance of ExtensionDtype.type for scalar item : hence we use Quantity
    type = Quantity
    
    
    # ExtensionDtypes are required to be hashable. The base class provides a default implementation,
    # which relies on the _metadata class attribute. _metadata should be a tuple containing the 
    # strings that define your data type. For example, with PeriodDtype that’s the freq attribute.
    # If you have a parametrized dtype you should set the ``_metadata`` class property.
    # Ideally, the attributes in _metadata will match the parameters to your ExtensionDtype.__init__ 
    # (if any). If any of the attributes in _metadata don’t implement the standard __eq__ or __hash__, 
    # the default implementations here will not work. QuantityDtype are parametrized by a
    # physical quantity, so we rely on the hash of the quantity to hash the Dtype.
    _metadata = ("unit",)
    
    
    # for construction from string
    _match = re.compile(r"physipy\[(?P<unit>.+)\]")
    
    
    # this will be used by QuantityArray when accessing indexes that do not
    # exist (see .take)
    # The na_value class attribute can be used to set the default NA value for this type.
    # numpy.nan is used by default : we overide this with na_value = Quantity(np.nan, Dimension(None))
    # could be na_value = Quantity(np.nan, Dimension(None)) but this then fails for concatenation
    # with object.
    @property
    def na_value(self):
        """
        this will be used by QuantityArray when accessing indexes that do not
        exist (see .take), as well as shifting to create new empty element.
        """
        return Quantity(np.nan, self.dimension)

    
    #  Return the array type associated with this dtype : QuantityArray
    @classmethod
    def construct_array_type(cls, *args):
        """Return the array type associated with this dtype."""
        return QuantityArray
    
    
    #############################
    #######  To print the unit under the values when repr-ed
    def __new__(cls, unit=None):
        if isinstance(unit, QuantityDtype):
            return unit
        elif isinstance(unit, str):
            unit = cls._parse_dtype_strict(unit)
        elif unit is None:
            unit = quantify(1)
            
        if isinstance(unit, Quantity):
            #qdtype_unit = QuantityDtype(unit)
            u = object.__new__(cls)
            u.unit = unit
            return u
        else:
            raise ValueError


    # property that returns a string identifying the data type.
    # Will be used for display in, e.g. Series.dtype
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

        
        
    # Construct this type from a string. Construct this type from a string.
    # This is useful mainly for data types that accept parameters. For example, a period dtype 
    # accepts a frequency parameter that can be set as period[H] (where H means hourly frequency).
    @classmethod
    def construct_from_string(cls, string):
        """
        Strict construction from a string, raise a TypeError if not
        possible
        """
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
        if not isinstance(string, str):
            raise TypeError(
                f"'construct_from_quantity_string' expects a string, got {type(string)}"
            )

        quantity = cls.ureg.Quantity(string)
        return cls(unit=quantity.unit)
    
    
    # returns True for now, but should it since we are not a plain number ?. If not over
    # riden , returns False by inheritance of ExtensionDtype
    @property
    def _is_numeric(self):
        return True
    
    
    
from pandas.core.arrays.base import ExtensionOpsMixin
from pandas.api.types import is_list_like



# ExtensionOpsMixin : for operations between series for ex
# see https://github.com/pandas-dev/pandas/blob/
# 21d61451ca363bd4c67dbf284b29daacc30302b1/pandas/core/arrays/base.py#L1330
# must define _create_arithmetic_method(cls, op) for ex
class QuantityArray(ExtensionArray, ExtensionOpsMixin):
    """Abstract base class for custom 1-D array types."""

    def __init__(self, values, dtype=None, copy=False):
        """Instantiate the array.
        If you're doing any type coercion in here, you will also need
        that in an overwritten __settiem__ method.
        But, here we coerce the input values into Decimals.
        """
        # check if we have a list like [<Quantity:[1, 2, 3], m>]
        if (isinstance(values, list) or isinstance(values, np.ndarray)) and len(values) == 1 and isinstance(values[0], Quantity):
            values = values
        values = quantify(values)
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

        
    # Shortcut to access the actual data
    @property
    def quantity(self):
        return self._data
    
    
    # Shortcut to access the dimension - should be equivalent to the QuantityDtype
    @property
    def dimension(self):
        return self._data.dimension
    
        
    ###################################
    ########   ExtensionArray interface
    ###################################
    # Atributes 
    #     dtype
    #     nbytes
    # Methods
    #     _from_sequence
    #     __getitem__
    #     __len__
    #     __eq__
    #     isna
    #     take
    #     copy
    #     _concat_same_type
    #     TODO : _from_factorized
    
    @property
    def dtype(self):
        """An instance of 'QuantityDtype'."""
        return self._dtype
        
        
    @property
    def nbytes(self):
        """
        The number of bytes needed to store this object in memory.
        """        
        # If this is expensive to compute, return an approximate lower bound
        # on the number of bytes needed.
        return self._itemsize * len(self)

    
    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        """
        Construct a new QuantityArray from a sequence of scalars.
        
        Parameters
        ----------
        scalars : Sequence
            Each element will be an instance of the scalar type for this
            array, ``cls.dtype.type`` or be converted into this type in this method.
        dtype : dtype, optional
            Construct for this particular dtype. This should be a Dtype
            compatible with the ExtensionArray.
        copy : bool, default False
            If True, copy the underlying data.
        Returns
        -------
        QuantityArray
        
        """
        values = asqarray(scalars)        
        return cls(values, dtype=dtype)

    
    def __getitem__(self, item):
        """
        Select a subset of self.
        Parameters
        ----------
        item : int, slice, or ndarray
            * int: The position in 'self' to get.
            * slice: A slice object, where 'start', 'stop', and 'step' are
              integers or None
            * ndarray: A 1-d boolean NumPy ndarray the same length as 'self'
        Returns
        -------
        item : scalar or ExtensionArray
        Notes
        -----
        For scalar ``item``, return a scalar value suitable for the array's
        type. This should be an instance of ``self.dtype.type``.
        For slice ``key``, return an instance of ``ExtensionArray``, even
        if the slice is length 0 or 1.
        For a boolean mask, return an instance of ``ExtensionArray``, filtered
        to the values where ``item`` is True.
        """
        # for scalar item
        if is_integer(item):
            # we use the __getitem__ of the underlying Quantity
            # to return a QuantityType.type == Quantity instance scalar
            return self._data[item] 

        # seems to be a good practice
        item = check_array_indexer(self, item)

        # return another QuantityArray with the subset
        # again using __getitem__ of the Quantity
        return self.__class__(self._data[item], self.dtype)

    
    #def __setitem__(self, key, value):
    #    """
    #    Necessary for some funtions.
    #    """
    #    self._data.value[key] = value
    
    def round(self, decimals=0, *args, **kwargs):
        """
        Used by round.
        """
        return type(self)(np.around(self.quantity.value, 
                                    decimals)*self.quantity._SI_unitary_quantity, 
                          self.dtype)
        
        
    def __len__(self) -> int:
        """
        Length of this array
        Returns
        -------
        length : int
        """
        return len(self._data)

    
    def __eq__(self, other):
        """
        Return for `self == other` (element-wise equality).
        """
        # Implementer note: this should return a boolean numpy ndarray or
        # a boolean ExtensionArray.
        # When `other` is one of Series, Index, or DataFrame, this method should
        # return NotImplemented (to ensure that those objects are responsible for
        # first unpacking the arrays, and then dispatch the operation to the
        # underlying arrays)
        if instance(other, pd.DataFrame) or isintance(other, pd.Series) or isinstance(pd.Index):
            raise NotImplemented
        # rely on Quantity comparison that will return a boolean array
        return self._data == other._data
            

    def isna(self):
        """
        A 1-D array indicating if each value is missing.
        Returns
        -------
        na_values : Union[np.ndarray, ExtensionArray]
            In most cases, this should return a NumPy ndarray. For
            exceptional cases like ``SparseArray``, where returning
            an ndarray would be expensive, an ExtensionArray may be
            returned.
        Notes
        -----
        If returning an ExtensionArray, then
        * ``na_values._is_boolean`` should be True
        * `na_values` should implement :func:`ExtensionArray._reduce`
        * ``na_values.any`` and ``na_values.all`` should be implemented
        """
        # Quantity implements this directly
        return self._data.is_nan() 
        

    def take(self, indices, *,allow_fill=False, fill_value=None):
            """
            Take elements from an array.
            Parameters
            ----------
            indices : sequence of int
                Indices to be taken.
            allow_fill : bool, default False
                How to handle negative values in `indices`.
                * False: negative values in `indices` indicate positional indices
                  from the right (the default). This is similar to
                  :func:`numpy.take`.
                * True: negative values in `indices` indicate
                  missing values. These values are set to `fill_value`. Any other
                  other negative values raise a ``ValueError``.
            fill_value : any, optional
                Fill value to use for NA-indices when `allow_fill` is True.
                This may be ``None``, in which case the default NA value for
                the type, ``self.dtype.na_value``, is used.
                For many ExtensionArrays, there will be two representations of
                `fill_value`: a user-facing "boxed" scalar, and a low-level
                physical NA value. `fill_value` should be the user-facing version,
                and the implementation should handle translating that to the
                physical version for processing the take if necessary.
            Returns
            -------
            ExtensionArray
            Raises
            ------
            IndexError
                When the indices are out of bounds for the array.
            ValueError
                When `indices` contains negative values other than ``-1``
                and `allow_fill` is True.
            See Also
            --------
            numpy.take : Take elements from an array along an axis.
            api.extensions.take : Take elements from an array.
            Notes
            -----
            ExtensionArray.take is called by ``Series.__getitem__``, ``.loc``,
            ``iloc``, when `indices` is a sequence of values. Additionally,
            it's called by :meth:`Series.reindex`, or any other method
            that causes realignment, with a `fill_value`.
            Examples
            --------
            Here's an example implementation, which relies on casting the
            extension array to object dtype. This uses the helper method
            :func:`pandas.api.extensions.take`.
            .. code-block:: python
               def take(self, indices, allow_fill=False, fill_value=None):
                   from pandas.core.algorithms import take
                   # If the ExtensionArray is backed by an ndarray, then
                   # just pass that here instead of coercing to object.
                   data = self.astype(object)
                   if allow_fill and fill_value is None:
                       fill_value = self.dtype.na_value
                   # fill value should always be translated from the scalar
                   # type for the array, to the physical storage type for
                   # the data, before passing to take.
                   result = take(data, indices, fill_value=fill_value,
                                 allow_fill=allow_fill)
                   return self._from_sequence(result, dtype=self.dtype)
            """
            # Implementer note: The `fill_value` parameter should be a user-facing
            # value, an instance of self.dtype.type. When passed `fill_value=None`,
            # the default of `self.dtype.na_value` should be used.
            # This may differ from the physical storage type your ExtensionArray
            # uses. In this case, your implementation is responsible for casting
            # the user-facing type to the storage type, before using
            # pandas.api.extensions.take
            # raise AbstractMethodError(self)
        
            # Base Pandas implementation example
            # from pandas.core.algorithms import take
            # # If the ExtensionArray is backed by an ndarray, then
            # # just pass that here instead of coercing to object.
            # data = self.astype(object)
            # if allow_fill and fill_value is None:
            #     fill_value = self.dtype.na_value
            # # fill value should always be translated from the scalar
            # # type for the array, to the physical storage type for
            # # the data, before passing to take.
            # result = take(data, indices, fill_value=fill_value,
            #               allow_fill=allow_fill)
            # return self._from_sequence(result, dtype=self.dtype)
            #
            # Pintpandas implementation
            # data = self._data
            # if allow_fill and fill_value is None:
            #     fill_value = self.dtype.na_value
            # if isinstance(fill_value, _Quantity):
            #     fill_value = fill_value.to(self.units).magnitude
            # result = take(data, indices, fill_value=fill_value, 
            #               allow_fill=allow_fill)
            # return PintArray(result, dtype=self.dtype)
            
            from pandas.core.algorithms import take
            if allow_fill and fill_value is None:
                fill_value = self.dtype.na_value
            return QuantityArray(take(self._data,
                                      indices, 
                                      fill_value=fill_value,
                                      allow_fill=allow_fill,
                                     ),
                                 self.dtype)
            
            
    def copy(self):
        """
        Return a copy of the array.
        Returns
        -------
        QuantityArray
        """
        return type(self)(self._data.copy(), self._dtype)

    
    @classmethod
    def _concat_same_type(cls, to_concat):
        """
        Concatenate multiple array of this dtype.
        Parameters
        ----------
        to_concat : sequence of this type
        Returns
        -------
        ExtensionArray
        """
        # Implementer note: this method will only be called with a sequence of
        # ExtensionArrays of this class and with the same dtype as self. This
        # should allow "easy" concatenation (no upcasting needed), and result
        # in a new ExtensionArray of the same dtype.
        # Note: this strict behaviour is only guaranteed starting with pandas 1.1
        return cls(np.concatenate([x._data for x in to_concat]), cls.dtype)
    
    
    ##############################
    ##############################
    
    #########################
    # Performance methods 
    #########################
    # Some methods require casting the ExtensionArray to an ndarray of Python
    # objects with ``self.astype(object)``, which may be expensive. When
    # performance is a concern, we highly recommend overriding the following
    # methods:
    #    fillna
    #    dropna
    #    unique
    #    factorize / _values_for_factorize
    #    argsort / _values_for_argsort
    #    searchsorted


    def dropna(self):
        """
        Return QuantityArray without NA values.
        Returns
        -------
        valid : QuantityArray
        """
        return self[~self.isna()]  
    
    def unique(self):
        """
        Compute the QuantityArray of unique values.
        Returns
        -------
        uniques : QuantityArray
        """
        from pandas.core.algorithms import unique
        uniques = unique(self.quantity.value)
        return QuantityArray(Quantity(uniques, self.quantity.dimension), self.dtype)
    
    def searchsorted(self, value, side="left", sorter=None):
        """
        Find indices where elements should be inserted to maintain order.
        Find the indices into a sorted array `self` (a) such that, if the
        corresponding elements in `value` were inserted before the indices,
        the order of `self` would be preserved.
        Assuming that `self` is sorted:
        ======  ================================
        `side`  returned index `i` satisfies
        ======  ================================
        left    ``self[i-1] < value <= self[i]``
        right   ``self[i-1] <= value < self[i]``
        ======  ================================
        Parameters
        ----------
        value : array-like
            Values to insert into `self`.
        side : {'left', 'right'}, optional
            If 'left', the index of the first suitable location found is given.
            If 'right', return the last such index.  If there is no suitable
            index, return either 0 or N (where N is the length of `self`).
        sorter : 1-D array-like, optional
            Optional array of integer indices that sort array a into ascending
            order. They are typically the result of argsort.
        Returns
        -------
        array of ints
            Array of insertion points with the same shape as `value`.
        See Also
        --------
        numpy.searchsorted : Similar method from NumPy.
        """
        # Note: the base tests provided by pandas only test the basics.
        # We do not test
        # 1. Values outside the range of the `data_for_sorting` fixture
        # 2. Values between the values in the `data_for_sorting` fixture
        # 3. Missing values.
        return np.searchsorted(self.quantity, value, side=side, sorter=sorter)
    
    def _values_for_argsort(self):
        """
        Return values for sorting.
        Returns
        -------
        ndarray
            The transformed values should maintain the ordering between values
            within the array.
        See Also
        --------
        ExtensionArray.argsort : Return the indices that would sort this array.
        """
        # Note: this is used in `ExtensionArray.argsort`.
        return np.array(self.quantity.value)
    
    ####################
    # Array reduction
    ####################
    # One can implement methods to handle array reductions.
    # * _reduce    
    def _reduce(self, name, *, skipna=True, **kwargs):
        """
        Return a scalar result of performing the reduction operation.
        Parameters
        ----------
        name : str
            Name of the function, supported values are:
            { any, all, min, max, sum, mean, median, prod,
            std, var, sem, kurt, skew }.
        skipna : bool, default True
            If True, skip NaN values.
        **kwargs
            Additional keyword arguments passed to the reduction function.
            Currently, `ddof` is the only supported kwarg.
        Returns
        -------
        scalar
        Raises
        ------
        TypeError : subclass does not define reductions
        """
        functions = {
            "all": all,
            "any": any,
            "min": min,
            "max": max,
            "sum": sum,
            "mean": np.mean,
            "median": np.median,
            "prod": np.prod,
            "std": np.std, 
            "var": np.var,
            # sem
            # kurt
            # skew
        }
        if name not in functions:
            raise TypeError(f"cannot perform {name} with type {self.dtype}")
        
        if skipna:
            quantity = self.dropna()._data
        else:
            quantity = self._data    
            
        return functions[name](quantity)

    
    ################
    ### REPR
    ###############
    # A default repr displaying the type, (truncated) data, length,
    # and dtype is provided. It can be customized or replaced by
    # by overriding:
    #    __repr__ : A default repr for the ExtensionArray.
    #    _formatter : Print scalars inside a Series or DataFrame.
    
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
    
   
    
    ################### 
    # Stats helper
    ###############
    def value_counts(self, dropna=True):
        """
        From https://github.com/hgrecco/pint-pandas/blob/04d4ed7befb42d3c830885d7f39997eac5392af3/pint_pandas/pint_array.py
        """
        #"""
        #Returns a Series containing counts of each category.
        #Every category will have an entry, even those with a count of 0.
        #Parameters
        #----------
        #dropna : boolean, default True
        #    Don't include counts of NaN.
        #Returns
        #-------
        #counts : Series
        #See Also
        #--------
        #Series.value_counts
        #"""

        from pandas import Series
        from pandas.core.describe import describe_ndframe

        # compute counts on the data with no nans
        #data = self._data
        #if dropna:
        #    data = data[~np.isnan(data)]
        #print(data, type(data))
#
        #data_list = data.tolist()
        #index = list(set(data))
        #array = [data_list.count(item) for item in index]
#
        raw_res = describe_ndframe(self._data)
        return raw_res#pd.Series(array, index=index)
    
    

    ###################################
    ##### ExtensionOpsMixin
    ###################################
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
        if isinstance(dtype, str) and (dtype.startswith("physipy[")):
            dtype = QuantityDtype(dtype)
        if isinstance(dtype, QuantityDtype):
            if dtype == self._dtype and not copy:
                return self
            else:
                return self.copy()
        elif isinstance(dtype, Quantity):
            return QuantityArray(self.quantity.value, QuantityDtype(dtype))
        #return self.__array__(dtype, copy)
        return Quantity(self.quantity.value.astype(dtype), self.quantity.dimension)

    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Series implements __array_ufunc__. As part of the implementation, pandas unboxes the ExtensionArray
        from the Series, applies the ufunc, and re-boxes it if necessary.
 
        If applicable, we highly recommend that you implement __array_ufunc__ in your extension array to avoid
        coercion to an ndarray. See the NumPy documentation for an example.
       
        As part of your implementation, we require that you defer to pandas when a pandas container (Series, DataFrame, Index)
        is detected in inputs. If any of those is present, you should return NotImplemented. pandas will take care of unboxing
        the array from the container and re-calling the ufunc with the unwrapped input.
        """
        if any(map(lambda x:isinstance(x, pd.DataFrame) or isinstance(x, pd.Series), inputs)):
            raise NotImplemented
        if not (method == "__call__" or method =="reduce"):
            raise NotImplementedError(f"array ufunc {ufunc} with method {method} not implemented")
 
        # first compute the raw quantity result
        inputs = map(lambda x:x.quantity, inputs)
        if method == "__call__":
            qres = ufunc.__call__(*inputs)
        elif method == "reduce":
            qres = ufunc.reduce(*inputs, **kwargs)
 
        # then wrap it in a QuantityArray, with the corresponding dtype
        # good thing is that the dimensions work is handled in Quantity
        try:
            n = len(qres)
            return QuantityArray(qres, QuantityDtype(qres))
        except:
            return qres
        
        
    
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



