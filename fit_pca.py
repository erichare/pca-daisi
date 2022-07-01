import os
import random
import pandas as pd
import numpy as np
import streamlit as st

from sklearn import datasets
from sklearn.decomposition import PCA
from bokeh.plotting import figure, show, ColumnDataSource, output_file
from bokeh.layouts import gridplot


def _recode_iris_variety(species: str):
    '''
        Internal function to recode from the 0/1/2 species encoding to human readable labels

        :param int species: The species integer code
        :return: String recoded
    '''
    if species == 0:
        return "Setosa"
    elif species == 1:
        return "Versicolor"
    elif species == 2:
        return "Virginica"
    else:
        return None


def plot_pca(pca_data, x_component=1, y_component=2, split_by=None):
    '''
    Plot a PCA in Bokeh of the given data
    
    This function takes a pre-computed PCA dataset, along with parameters
    to specify the components to plot, and optionally a split variable,
    and renders a Bokeh visualization of the given data

    :param pd.DataFrame pca_data: The PCA data which to plot
    :param int x_component: The component number to plot on the x axis
    :param int y_component: The component number to plot on the y axis
    :param str split_by: Optionally, the variable to split/color by 
    
    :return: Bokeh Visualization of PCA Results
    '''

    fig = figure(plot_width=1000, plot_height=1000)

    if not split_by:
        pca_data["split"] = 1
    else:
        pca_data["split"] = pca_data[split_by]
        pca_data.drop(split_by, axis=1, inplace=True)

    for pop in pca_data["split"].unique().tolist():
        
        pca_sub = pca_data.loc[pca_data['split'] == pop]
        source = ColumnDataSource(pca_sub)
        
        fig.circle(
            'PC' + str(x_component), 'PC' + str(y_component), 
            source=source,
            line_color='black',
            line_width=0.5,
            size=6,
            legend_label=str(pop),
            fill_color="#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
        )
    
    return fig

def fit_pca(df: pd.DataFrame=None, vars=None, n_components=2):
    '''
    Fit a PCA to the given dataset

    This function takes a data frame and, given the specified variables
    and number of components, fits a PCA to the data. If the input data
    frame is not specified, the iris data is used by default

    :param pd.DataFrame df: The data with which to generate Prinicpal Components
    :param str vars: Comma-separated list of variables to include for the PCA
    :param int n_components: The number of components to generate

    :return: DataFrame of PCA Results
    '''
    if type(df) == str and os.path.isfile(df):
        df = pd.read_csv(df)

    if df is None:
        iris = datasets.load_iris()

        df = pd.DataFrame(iris.data)
        df["variety"] = iris.target
        df["variety"] = df["variety"].apply(_recode_iris_variety)

        df.columns = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width", "Variety"]

    df.dropna(axis=0, how='any', inplace=True)

    other_cols = []
    numeric_vars = df.select_dtypes([np.number]).columns
    if not vars:
        vars = numeric_vars
    else:
        vars = [x.strip() for x in vars.split(",")]
    other_cols = list(set(df.columns) - set(numeric_vars))

    pca_df = df[vars]

    pca = PCA(n_components=n_components)
    pca.fit(pca_df)

    print("Explained Variance Ratio:")
    print(pca.explained_variance_ratio_)

    pca_data = pd.DataFrame(pca.fit_transform(pca_df))
    for col in other_cols:
        pca_data[col] = df[col]

    pca_data.columns = ["PC" + str(x + 1) for x in range(pca_data.shape[1] - 1)] + other_cols

    return pca_data

if __name__ == "__main__":
    st.set_page_config(layout = "wide")
    st.title("Generalized Principal Components Analysis")

    st.write("This Daisi, powered by Streamlit, allows for a generalized specification of a Principal Components Analysis. Upload your data to get started!")
    with st.sidebar:
        uploaded_file = st.file_uploader("Choose a file")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        iris = datasets.load_iris()

        df = pd.DataFrame(iris.data)
        df["variety"] = iris.target
        df["variety"] = df["variety"].apply(_recode_iris_variety)

        df.columns = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width", "Variety"]

    numeric_vars = list(df.select_dtypes([np.number]).columns)

    with st.sidebar:
        vars = st.multiselect("Choose Variables", numeric_vars, default=numeric_vars)
        split = st.multiselect("Choose Split", list(set(df.columns) - set(numeric_vars)))
        n_components = st.number_input("Choose Number of Components", min_value=1, max_value=8, step=1, value=2)

    pca_data = fit_pca(df, vars=",".join(vars))
    p = plot_pca(pca_data, split_by=split)

    st.bokeh_chart(p, use_container_width=True)
