import streamlit as st
import os
import json
from datetime import datetime
from vocalock import enroll_user, authenticate_user, process_access_request, LOG_DIR

st.set_page_config(page_title="Vocalock", layout="centered")
st.title("🔐 Vocalock – Voice Access Control")

# Sidebar menu
menu = st.sidebar.radio("Navigation", ["Home", "Enroll", "Authenticate", "View Logs"])

if menu == "Home":
    st.markdown("""
    Welcome to **Vocalock** – a secure voice-activated access control system.
    
    Use the sidebar to:
    - 🎙 Enroll new users
    - 🔑 Authenticate using voice
    - 📜 View access logs
    """)

elif menu == "Enroll":
    st.header("🎙 Enroll New User")
    user_id = st.text_input("Enter your desired username")

    if st.button("Start Enrollment"):
        if not user_id:
            st.warning("Please enter a username first.")
        else:
            with st.spinner("Recording and processing..."):
                uid, passphrase, features = enroll_user(user_id)
            if uid:
                st.success(f"Enrollment successful for `{uid}`")
                st.info(f"Your passphrase is: `{passphrase}`")
            else:
                st.error("Enrollment failed. Please try again.")

elif menu == "Authenticate":
    st.header("🔑 Authenticate User")
    user_id = st.text_input("Enter your username")

    if st.button("Start Authentication"):
        if not user_id:
            st.warning("Please enter your username.")
        else:
            with st.spinner("Authenticating..."):
                auth_result, auth_details = authenticate_user(user_id)
                access_granted, message = process_access_request(auth_result, auth_details)

            if access_granted:
                st.success("🔓 ACCESS GRANTED")
                st.balloons()
            else:
                st.error("🔒 ACCESS DENIED")

            st.markdown("#### Authentication Summary")
            st.json(auth_details)
            st.info(message)

elif menu == "View Logs":
    st.header("📜 Daily Access Logs")

    today = datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(LOG_DIR, f"{today}_access_log.json")

    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            logs = json.load(f)

        st.success(f"Showing last {min(10, len(logs))} access attempts for `{today}`:")
        for log in logs[-10:][::-1]:
            status = "✅ GRANTED" if log["access_granted"] else "❌ DENIED"
            st.write(f"- **{log['timestamp']}** — `{log['user_id']}` — {status}")
    else:
        st.warning("No logs available for today.")
