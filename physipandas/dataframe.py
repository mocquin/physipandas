from pandas.api.extensions import register_dataframe_accessor
from pandas import DataFrame

from .extension import QuantityDtype, QuantityArray


@register_dataframe_accessor("physipy")
class PhysipyDataFrameAccessor(object):
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def quantify(self, level=-1):
        """
        
        """
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

    def show(self):
        """Alias for df.physipy.dequantify()"""
        return self.dequantify()
    
    def dequantify(self):
        # TODO : add the dtype as string for non quantity columns ?
        def formatter_func(units):
                    #formatter = "{:" + units._REGISTRY.default_format + "}"
                    #formatter = "{:"+str(units.str_SI_unit())+"}"
            formatter = "{:}"
            return formatter.format(units.str_SI_unit())
        df = self._obj
        df_columns = df.columns.to_frame()
        units = []
        for col in df.columns:
            if isinstance(df[col].dtype, QuantityDtype):
                units.append(formatter_func(df[col].values.dimension))
            else:
                units.append("-")
        df_columns["units"] = units
        from collections import OrderedDict
        
        data_for_df = OrderedDict()
        for i, col in enumerate(df.columns):
            if isinstance(df[col].dtype, QuantityDtype):
                data_for_df[tuple(df_columns.iloc[i])] = QuantityArray(df[col].values._data)
            else:
                data_for_df[tuple(df_columns.iloc[i])] = df[col]
        df_new = DataFrame(
            data_for_df, 
            columns=data_for_df.keys(),
        )

        df_new.columns.names = df.columns.names + ["unit"]
        df_new.index = df.index
        return df_new

