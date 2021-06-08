Pandas introduction to Categorical series : 
https://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html

https://medium.com/ibm-data-ai/text-extensions-for-pandas-tips-and-techniques-for-extending-pandas-e0c745cc9dbb


PS3_Error_Propagation_sp13.pdf (harvard.edu)
http://ipl.physics.harvard.edu/wp-uploads/2013/03/PS3_Error_Propagation_sp13.pdf

Microsoft Word - 2.Propagation (mit.edu)
http://web.mit.edu/fluids-modules/www/exper_techniques/2.Propagation_of_Uncertaint.pdf

 
PintArrays
https://www.youtube.com/watch?v=xx7H5EkzQH0
Extending pandas — pandas 1.2.4 documentation (pydata.org)
https://pandas.pydata.org/docs/development/extending.html

The Easy Way to Extend Pandas API | by Eyal Trabelsi | Towards Data Science
https://towardsdatascience.com/ready-the-easy-way-to-extend-pandas-api-dcf4f6612615

https://github.com/pandas-dev/pandas/blob/21d61451ca363bd4c67dbf284b29daacc30302b1/pandas/core/dtypes/base.py#L34
 
Example wih decimal array
pandas-extension-dtype/decimal_array.py at master · tomharvey/pandas-extension-dtype · GitHub
https://github.com/tomharvey/pandas-extension-dtype/blob/master/decimal_array.py
 
support for decimal.Decimal
 
 
 
 
making Dimension hashable :   
 def __hash__(self):
        return hash(str(self))
 
Pandas accessor :
import pandas as pd
@pd.api.extensions.register_dataframe_accessor("opt")
class MonteCarloAccessor:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj
 
    @staticmethod
    def _validate(obj):
        # verify there is a column latitude and a column longitude
        if "toto" not in obj.columns or "PSA" not in obj.columns:
            raise AttributeError("Must have 'N' and 'PSA'.")
 
    @property
    def mean_gain(self):
        # return the geographic center point of this DataFrame
        N = self._obj.N
        PSA = self._obj.PSA
        return float(N.mean()) * float(PSA.mean())
 
    def plot(self):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2)
        axes[0].hist(self._obj.N)
        axes[0].hist(self._obj.PSA)
      
after this is executed, any created DataFrame will have the ability to be used as : “df.opt.mean_gain”
_validate is called at time of accession of df.opt.*