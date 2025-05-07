import streamlit as st
import json
import requests
import pandas as pd
from pathlib import Path

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ''

def load_credentials():
    cred_file = Path('GUI/credentials.json')
    if not cred_file.exists():
        with open(cred_file, 'w') as f:
            json.dump({'users': []}, f)
    with open(cred_file, 'r') as f:
        return json.load(f)

def save_credentials(credentials):
    with open('GUI/credentials.json', 'w') as f:
        json.dump(credentials, f, indent=4)

def login():
    st.title('House Hunt - Login')
    username = st.text_input('Username')
    password = st.text_input('Password', type='password')
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button('Login'):
            credentials = load_credentials()
            for user in credentials['users']:
                if user['username'] == username and user['password'] == password:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success('Login successful!')
                    st.rerun()
                    return
            st.error('Invalid username or password')
    
    with col2:
        if st.button('Create Account'):
            st.session_state.page = 'create_account'
            st.rerun()

def create_account():
    st.title('Create Account')
    username = st.text_input('Username')
    password = st.text_input('Password', type='password')
    confirm_password = st.text_input('Confirm Password', type='password')
    
    if st.button('Create Account'):
        if not username or not password:
            st.error('Please fill all fields')
            return
        
        if password != confirm_password:
            st.error('Passwords do not match')
            return
        
        credentials = load_credentials()
        
        # Check if username exists
        for user in credentials['users']:
            if user['username'] == username:
                st.error('Username already exists')
                return
        
        # Add new user
        credentials['users'].append({
            'username': username,
            'password': password
        })
        save_credentials(credentials)
        
        st.success('Account created successfully!')
        st.session_state.page = 'login'
        st.rerun()
    
    if st.button('Back to Login'):
        st.session_state.page = 'login'
        st.rerun()

def recommendations_page():
    st.title('House Recommendations')
    
    with st.form('recommendations_form'):
        col1, col2 = st.columns(2)
        with col1:
            min_price = st.number_input('Minimum Price', min_value=0.0)
            bedrooms = st.number_input('Number of Bedrooms', min_value=1, max_value=10, value=2)
        with col2:
            max_price = st.number_input('Maximum Price', min_value=0.0)
            city = st.text_input('City')
        
        st.subheader('Amenities')
        col1, col2 = st.columns(2)
        amenities = []
        with col1:
            if st.checkbox('Gymnasium'): amenities.append('Gymnasium')
            if st.checkbox('SwimmingPool'): amenities.append('SwimmingPool')
            if st.checkbox('LandscapedGardens'): amenities.append('LandscapedGardens')
            if st.checkbox('JoggingTrack'): amenities.append('JoggingTrack')
        with col2:
            if st.checkbox('RainWaterHarvesting'): amenities.append('RainWaterHarvesting')
            if st.checkbox('ClubHouse'): amenities.append('ClubHouse')
            if st.checkbox('CarParking'): amenities.append('CarParking')
        
        if st.form_submit_button('Search'):
            try:
                preferences = {
                    'min_price': min_price,
                    'max_price': max_price,
                    'bedrooms': int(bedrooms),
                    'city': city,
                    'amenities': amenities
                }
                
                response = requests.post('http://localhost:8001/recommend', json=preferences)
                recommendations = response.json()['recommendations']
                
                if not recommendations:
                    st.warning('No recommendations found.')
                else:
                    st.success('Found recommendations!')
                    df = pd.DataFrame(recommendations)
                    st.dataframe(df[['Price', 'Area', 'No. of Bedrooms', 'Location', 'City']])
            except Exception as e:
                st.error(f'Error: {str(e)}')

def price_prediction_page():
    st.title('Price Prediction')
    
    with st.form('price_prediction_form'):
        col1, col2 = st.columns(2)
        with col1:
            area = st.number_input('Area (sq ft)', min_value=0.0)
            bedrooms = st.number_input('Number of Bedrooms', min_value=1, max_value=10, value=2)
        with col2:
            city = st.text_input('City')
        
        st.subheader('Amenities')
        col1, col2 = st.columns(2)
        amenities = []
        with col1:
            if st.checkbox('Gymnasium'): amenities.append('Gymnasium')
            if st.checkbox('SwimmingPool'): amenities.append('SwimmingPool')
            if st.checkbox('LandscapedGardens'): amenities.append('LandscapedGardens')
            if st.checkbox('JoggingTrack'): amenities.append('JoggingTrack')
        with col2:
            if st.checkbox('RainWaterHarvesting'): amenities.append('RainWaterHarvesting')
            if st.checkbox('ClubHouse'): amenities.append('ClubHouse')
            if st.checkbox('CarParking'): amenities.append('CarParking')
        
        if st.form_submit_button('Predict Price'):
            try:
                preferences = {
                    'area': area,
                    'bedrooms': int(bedrooms),
                    'city': city,
                    'amenities': amenities
                }
                
                response = requests.post('http://localhost:8001/predict_price', json=preferences)
                predicted_price = response.json()['predicted_price']
                
                st.success(f'Predicted Price: ₹{predicted_price:,.2f}')
            except Exception as e:
                st.error(f'Error: {str(e)}')

def main():
    if 'page' not in st.session_state:
        st.session_state.page = 'login'
    
    # Sidebar navigation when logged in
    if st.session_state.logged_in:
        with st.sidebar:
            st.title(f'Welcome, {st.session_state.username}!')
            page = st.radio('Navigation', ['Recommendations', 'Price Prediction'])
            if st.button('Logout'):
                st.session_state.logged_in = False
                st.session_state.username = ''
                st.session_state.page = 'login'
                st.rerun()
        
        if page == 'Recommendations':
            recommendations_page()
        else:
            price_prediction_page()
    else:
        if st.session_state.page == 'login':
            login()
        else:
            create_account()

if __name__ == '__main__':
    st.set_page_config(page_title='House Hunt', layout='wide')
    main()