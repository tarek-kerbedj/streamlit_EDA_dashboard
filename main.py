import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

#setting the title for the app
st.title("Dataframe exploration webapp")
#allowing the user to choose a classifier through a selectbox
classifier_name = st.sidebar.selectbox("Choose Classifier", ("Knn", "SVM", "Random forest"))
#first section called quick exploration
st.sidebar.header("quick exploration")
dfhead = st.sidebar.checkbox("Display head")
dfshape = st.sidebar.checkbox("Display Shape")
dfdescribe = st.sidebar.checkbox("Display summary stats")
dfcolumnnames = st.sidebar.checkbox("Display column names")
st.sidebar.markdown("""-------""")
#second section called quick visualization
st.sidebar.header("quick visualization ")
#prompt the user to choose a a csv file
st.subheader("Choose a custom dataset")
custom_dataset = st.file_uploader("", type=['csv'])
if custom_dataset:
    df = pd.read_csv(custom_dataset)

if dfhead:
    st.write(df.head())

if dfshape:
    st.write("the shape of your data is "+str(df.shape))

if dfdescribe:
    st.write(df.describe())

if dfcolumnnames:

    st.write(df.columns)

def plot_line(dataframe):
    figure = px.line(dataframe, x=str(x_axis), y=str(y_axis), markers=True)
    figure.update_layout(title_text=title, title_x=0.5)
    figure.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    figure.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    return figure
if st.sidebar.checkbox("plot line "):

    x_axis = st.sidebar.selectbox("X axis", df.columns.tolist())
    y_axis = st.sidebar.selectbox("Y axis", df.columns.tolist())
    title = st.sidebar.text_input("Title")

    x_axis= st.sidebar.selectbox("X axis",df.columns.tolist())
    y_axis=st.sidebar.selectbox("Y axis",df.columns.tolist())
    title=st.sidebar.text_input("Title")
    st.write(plot_line(df))





    fig = px.line(df, x=str(x_axis), y=str(y_axis), markers=True)
    fig.update_layout(title_text=title, title_x=0.5)
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    st.write(fig)
if st.sidebar.checkbox("Scatter plot"):
    xsc_axis = st.sidebar.selectbox("X scatter axis", df.columns.tolist())
    ysc_axis = st.sidebar.selectbox("Y  scatter axis", df.columns.tolist())
    title = st.sidebar.text_input("Title scatter")
    fig = px.scatter(
        x=df[xsc_axis],
        y=df[ysc_axis],
    )
    fig.update_layout(xaxis_title=str(xsc_axis), yaxis_title=str(ysc_axis),)
    fig.update_xaxes(showgrid=False, mirror=True)
    fig.update_yaxes(showgrid=False, mirror=True)
    st.write(fig)
if st.sidebar.checkbox("bar chart"):
    xbar_axis = st.sidebar.selectbox("X bar axis", df.columns.tolist())
    ybar_axis = st.sidebar.selectbox("Y bar axis", df.columns.tolist())
    bar_title = st.sidebar.text_input("Title")
    data_canada = px.data.gapminder().query("country == 'Canada'")
    fig = px.bar(df, x=str(xbar_axis), y=str(ybar_axis))
    st.plotly_chart(fig)
if st.sidebar.checkbox("Histogram"):
    xhist_axis = st.sidebar.selectbox("X axis", df.columns.tolist())
    fig, ax = plt.subplots()
    nbbins = st.sidebar.slider("choose number of bins", min_value=1, step=1)
    fig = px.histogram(df, x=xhist_axis, nbins=nbbins)
    st.plotly_chart(fig)
dfcorr = st.sidebar.checkbox("visualize corrolation matrix")
if dfcorr:
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), ax=ax)
    st.write(fig)

def get_classifier(clf_name):

    if clf_name == 'SVM':
        clf = SVC()
    elif clf_name == 'KNN':
        clf = KNeighborsClassifier()
    else:
        clf = RandomForestClassifier()
    return clf
st.sidebar.header('classification')
if st.sidebar.checkbox('classify'):

    classifier = get_classifier(classifier_name)
    data_X = st.sidebar.multiselect("select your data", df.columns.tolist())
    labels_Y = st.sidebar.selectbox("select your labels", df.columns.tolist())
    testsize=st.sidebar.slider("pick test size",0.1,0.8,0.2,0.1)
    if data_X and labels_Y:
        with st.spinner('please wait...'):

            X_train, X_test, y_train, y_test = train_test_split(df[data_X], df[labels_Y],\
             test_size=testsize, random_state=1234)

            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1=f1_score(y_test,y_pred,average='weighted')
            #st.write(f'Classifier = {classifier_name}')
            st.subheader(classifier_name)
            st.write('Testing Accuracy : ', acc)
            st.write('F1 score: ', f1)
            #st.markdown("testing accuracy")
            #st.metric('',acc)

