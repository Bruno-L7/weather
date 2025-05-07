import streamlit as st

st.title("Weather")

import streamlit as st

# Function to create footer
def footer():
    st.markdown(
        """
        <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 50%;
            text-align: right;
            padding: 10px;
            background-color: #0E1117;
        }
        .linkedin-logo {
            height: 20px;  /* Adjust the height of the logo */
            vertical-align: middle;  /* Align the logo with the text */
        }
        </style>
        <div class="footer">
            <a href="https://www.linkedin.com/in/brunolombardolamasalvarado/" target="_blank">
                <img src="https://upload.wikimedia.org/wikipedia/commons/0/01/LinkedIn_Logo.svg" class="linkedin-logo" alt="LinkedIn Logo">
                Bruno Lamas
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )

# Call the footer function
footer()