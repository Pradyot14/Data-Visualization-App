import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title='Data Visualization')
#st.write("Data visualization project")

st.markdown(
    f"""
    <h1 style='text-align: center; padding: 20px; background: linear-gradient(to right, violet, indigo, blue, green, magenta, orange, red); color: white;'>Insights from 120 Years of Olympic Games</h1>
    """,
    unsafe_allow_html=True
)




def page1():
    st.header("120 years Olympic Dataset", divider='rainbow')
    df = pd.read_csv('datasets/120years.csv')
    st.write(df)
    st.caption(
        'In the Medal Column :blue[0 represent no Medal] , :red[1 represent Gold] , :red[2 represent Silver] , :red[3 represent Bronze] .')

    st.bar_chart(df, x="Age", y="Height", color="Sex")

    st.scatter_chart(
        df,
        x='Weight',
        y='Height',
        color='Sex'
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(df['Weight'], df['Height'], c=df['Sex'].map({'M': 'blue', 'F': 'red'}))

    st.header("Height and Weight Scatterplot", divider='rainbow')
    # Customize plot
    ax.set_xlabel('Weight')
    ax.set_ylabel('Height')
    ax.set_title('Scatter Plot')
    ax.legend(*scatter.legend_elements(), title='Sex')
    # Display plot in Streamlit
    st.pyplot(fig)
    st.header("Map of Cities", divider='rainbow')
    st.image('datasets/m1.png', width=700)

    st.header("Madel Tally", divider='rainbow')
    st.image('datasets/a1.png', width=700)

    st.header("Year Wise participation", divider='rainbow')
    st.image('datasets/a2.png', width=700)

    st.header("Age Histogram", divider='rainbow')
    st.image('datasets/a3.png', width=700)

    st.header("Country madel tally", divider='rainbow')
    st.image('datasets/a4.png', width=700)

    st.header("Country Top Atheletes", divider='rainbow')
    st.image('datasets/a5.png', width=700)

    st.header("Country Top Atheletes", divider='rainbow')
    st.image('datasets/a6.png', width=700)

    st.header("Age distribution of atheletes", divider='rainbow')
    st.image('datasets/a7.png', width=700)

    st.header("Height distribution of atheletes", divider='rainbow')
    st.image('datasets/a8.png', width=700)

    st.header("Weight distribution of atheletes", divider='rainbow')
    st.image('datasets/a9.png', width=700)

    st.header("Trends of Age, Height, and Weight Over Different Olympic Games", divider='rainbow')
    st.image('datasets/a10.png', width=700)

    st.header("Olympic Performance of India Over the Years", divider='rainbow')
    st.image('datasets/a11.png', width=700)

    st.header("Male-Female Participation for Top 10 Countries and India Over the Years", divider='rainbow')
    st.image('datasets/a12.png', width=700)






def page2():
    st.header("Predictive Analytics", divider='rainbow')

    df = pd.read_csv('datasets/x_train.csv')
    # Add a download button for the uploaded CSV file
    st.download_button(
        label="Download x_train.csv file",
        data=df.to_csv().encode('utf-8'),
        file_name='x_train.csv',
        mime='text/csv')
    df = pd.read_csv('datasets/y_train.csv')
    # Add a download button for the uploaded CSV file
    st.download_button(
        label="Download y_train.csv file",
        data=df.to_csv().encode('utf-8'),
        file_name='y_train.csv',
        mime='text/csv')
    df = pd.read_csv('datasets/x_test.csv')
    # Add a download button for the uploaded CSV file
    st.download_button(
        label="Download x_test.csv file",
        data=df.to_csv().encode('utf-8'),
        file_name='x_test.csv',
        mime='text/csv')
    df = pd.read_csv('datasets/y_test.csv')
    # Add a download button for the uploaded CSV file
    st.download_button(
        label="Download y_test.csv file",
        data=df.to_csv().encode('utf-8'),
        file_name='y_test.csv',
        mime='text/csv')

    st.header('Neural Network Code')
    code = '''from keras.layers import Dropout
    from keras.optimizers import Adam
    from keras.callbacks import EarlyStopping
    from keras.models import Sequential
    from keras.layers import Flatten, Dense


    model = Sequential()
    model.add(Flatten(input_shape=(9,)))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(4, activation='softmax'))


    optimizer = Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])


    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)


    history = model.fit(x_train, ytrain,
                        validation_split=0.1,
                        batch_size=15,
                        epochs=20,
                        callbacks=[early_stopping],
                        verbose=1)


    test_loss, test_acc = model.evaluate(x_test, ytest)
    print("Test accuracy:", test_acc * 100)
    print('Test loss:', test_loss)
    '''
    st.code(code, language='python')
    st.write('Accuracy : 85.23%')

    st.title('Machine Learning Model Code')
    code = '''# Random Forest
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score


    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(x_train, y_train)
    predictions = rf_classifier.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy:", accuracy)

    '''
    st.code(code, language='python')
    st.write('Accuracy : 88.64%')
    st.write('Confusion matrix : ')
    st.image('datasets/c.png', width=400)

    code = '''# Gradient Boosting
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score

    gb_classifier = GradientBoostingClassifier(n_estimators = 100, max_depth = 5)


    gb_classifier.fit(x_train, y_train)
    train_predictions = gb_classifier.predict(x_train)
    train_accuracy = accuracy_score(y_train, train_predictions)

    print("Training Accuracy:", train_accuracy)

    test_predictions = gb_classifier.predict(x_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    print("Testing Accuracy:", test_accuracy)

    '''
    st.code(code, language='python')
    st.write('Accuracy : 86.23% ')
    st.write('Confusion matrix : ')
    st.image('datasets/c1.png', width=400)

    code = '''# XG Boost
    import xgboost as xgb

    x_train['Sex'] = label_encoder.fit_transform(x_train['Sex'])
    x_test['Sex'] = label_encoder.fit_transform(x_test['Sex'])


    xgb_classifier = xgb.XGBClassifier(n_estimators=100, max_depth=5)

    xgb_classifier.fit(x_train, y_train)

    train_predictions = xgb_classifier.predict(x_train)
    train_accuracy = accuracy_score(y_train, train_predictions)
    print("Training Accuracy:", train_accuracy)

    test_predictions = xgb_classifier.predict(x_test)


    test_accuracy = accuracy_score(y_test, test_predictions)
    print("Testing Accuracy:", test_accuracy)

    '''
    st.code(code, language='python')
    st.write('Accuracy : 86.37%')
    st.write('Confusion matrix : ')
    st.image('datasets/c2.png', width=400)


# Create a sidebar navigation bar to switch between pages
page = st.sidebar.selectbox("Select your choice", ["Analytics", "Predictive"])

# Depending on the selected page, display the corresponding content
# Apply CSS to customize the selectbox
if page == "Analytics":
    page1()
elif page == "Predictive":
    page2()




