import streamlit as st
import pickle

st.title('Which subreddit does your investment advice belong in?')
st.header('Distinguish between r/stocks and r/wallstreetbets')

page = st.sidebar.selectbox(
'Select a page:',
('About', 'Check Investment Advice', 'Contact Me')
)

if page == 'About':
    st.image('./stocks-wsb.png')
    st.write('In late January 2021, Gamestop\'s stock prices rose dramatically despite the company\'s shares being shorted at high rates. The forces behind the movement were a handful of r/WallStreetBets users who convinced Redditors en masse to buy up a bunch of GameStop stock, therefore increasing its value. As with any day-trade, this resulted in large profits for some and large losses for others, and the risks were especially magnified when the stock value was driven by the whims of a small group of Redditors rather than market inputs.')
    st.write('This project seeks to create a model that will help every day investors find less risky, long-term investments backed by financial and business results. In other words, it helps investors distinguish investment advice from r/Stocks vs r/WallStreetBets. The goal is to minimize instances of investors being told to buy a regular stock when it is actually a "meme stock" (minimize false negatives and optimize for recall/sensitivity).')
    link = '[Check out the presentation here!](https://docs.google.com/presentation/d/1CAfvf7QoDfEKiaFfA5aUh1ZLSaGdQ9uKJ8dO710Wg5A/edit#slide=id.g35f391192_00)'
    st.markdown(link, unsafe_allow_html=True)

if page == "Check Investment Advice":
    st.write('Generate predictions here! ')

    with open('finalized_model.sav', mode='rb') as pickle_in:
        pipe = pickle.load(pickle_in)

    user_text = st.text_input('Please input your investment advice:', value='hold on to gme')

    predicted_subreddit = pipe.predict([user_text])[0]

    output = ''
    if predicted_subreddit == 1:
        output+='r/wallstreetbets'
    else:
        output+='r/stocks'

    st.write(f'Your investment advice is from {output}')

if page == 'Contact Me':
    name = st.text_input('What is your name?')
    email = st.text_input('What is your email?')
    question = st.text_input('What is your question?')
    user_query_dict = {
        'name': name,
        'email': email,
        'question': question
    }

    if name and email and question:
        st.write(f'Thanks for reaching out {name}! I will get back to you on your question: {question} at {email} within 3 business days!')
