import matplotlib.pyplot as plt
import pandas as pd

def build_pie_chart(serie):  # pd.Series

    plt.pie(serie.value_counts().values,
            labels=serie.value_counts().index, shadow=True, autopct='%.0f%%')
    plt.title('{} Pie Chart'.format(serie.name))
    plt.legend(loc='upper right')
    plt.show()


def build_stacked_bars_chart(row_serie, col_serie):
    pd.crosstab(row_serie,
                col_serie,
                normalize='index').plot(kind='bar', stacked=True)
    plt.legend(loc='upper right')
    plt.show()