# physipandas : a pandas accessor and types for physipy


[physipy](https://github.com/mocquin/physipy) allows you to handle physical quantities in python. Using pandas [extension interface](https://pandas.pydata.org/docs/development/extending.html), physipandas provides additionnal features for great integration of physipy with pandas Series and Dataframe.

## Series Accessor

Using [pandas accessor interface](https://pandas.pydata.org/docs/development/extending.html#registering-custom-accessors), define an accessor for physipy.
In physipy, quantities have usefull attributes and methods, like `.dimension` or `.mean()`. Using physipandas series accessor allows you to interpret a Series kinda like a 1D quantity array, and access such attributes. Use this if you want your Series to behave like a regular `physipy.Quantity` object.

```python
import pandas as pd
import numpy as np
from physipy import m
from physipandas import QuantityDtype, QuantityArray

c = pd.Series(QuantityArray(np.arange(10)*m), 
              dtype=QuantityDtype(m))

print(type(c))
# <class 'pandas.core.series.Series'>
print(c.physipy.dimension)     # --> : L
print(c.physipy.values.mean()) # --> : 4.5 m
```

See the notebook on [Accessors Series](./docs/Acessors Series.ipynb).

## QuantityDtype

The first object is the `QuantityDtype`. You probably won't need to interact with this directly. This is an [`pandas.api.extensions.ExtensionDtype`](https://pandas.pydata.org/docs/development/extending.html#extensiondtype), just like [`pandas.CaterogicalDtype`](https://pandas.pydata.org/docs/reference/api/pandas.CategoricalDtype.html). 
See also [the extension types](https://pandas.pydata.org/docs/development/extending.html#extension-types).

```python
from physipy import m
from physipandas import QuantityDtype

qd = QuantityDtype()

```

## QuantityArray

This is the second part of the extension interface. An [`ExtensionArray`](https://pandas.pydata.org/docs/development/extending.html#extensionarray).
Citing [pandas docs](https://pandas.pydata.org/docs/development/extending.html#extensionarray) : 
> This class provides all the array-like functionality. ExtensionArrays are limited to 1 dimension. An ExtensionArray is linked to an ExtensionDtype via the dtype attribute. pandas makes no restrictions on how an extension array is created via its __new__ or __init__, and puts no restrictions on how you store your data. We do require that your array be convertible to a NumPy array, even if this is relatively expensive (as it is for Categorical).




```python
import numpy as np
from physipy import m, units

# get the milliseconds units
ms = units["ms"] 

t_samples = np.arange(10) * ms # array of milliseconds
x_samples = np.arange(20) * m  # array of meters

speeds = x_samples / t_samples
print(speeds)
```


# Usefull ressources for pandas extensions

Pandas introduction to Categorical series : https://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html

https://medium.com/ibm-data-ai/text-extensions-for-pandas-tips-and-techniques-for-extending-pandas-e0c745cc9dbb


PintArrays : https://www.youtube.com/watch?v=xx7H5EkzQH0
Extending pandas — pandas 1.2.4 documentation (pydata.org) : https://pandas.pydata.org/docs/development/extending.html

The Easy Way to Extend Pandas API | by Eyal Trabelsi | Towards Data Science : https://towardsdatascience.com/ready-the-easy-way-to-extend-pandas-api-dcf4f6612615

https://github.com/pandas-dev/pandas/blob/21d61451ca363bd4c67dbf284b29daacc30302b1/pandas/core/dtypes/base.py#L34
 
Example wih decimal array
pandas-extension-dtype/decimal_array.py at master · tomharvey/pandas-extension-dtype · GitHub
https://github.com/tomharvey/pandas-extension-dtype/blob/master/decimal_array.py
 
support for decimal.Decimal
 
 
 
 
__setitem__ pour quantityarray
concat_same_type(cls, to_concat) : self not defined
 