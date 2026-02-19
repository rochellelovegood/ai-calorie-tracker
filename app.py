import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os

# ----------------------------
# Load Gemini (Google) AI
# ----------------------------
from google import genai

# Initialize Gemini client using streamlit secrets
gemini_client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
# ----------------------------
# Load YOLO model (cached)
# ----------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ----------------------------
# Simple Calorie Database (expandable)
# ----------------------------
calorie_db = {
    "banana": 89,
    "apple": 52,
    "pizza": 266,
    "sandwich": 250,
    "cake": 350,
    "broccoli": 34,
    "carrot": 41,
    "rice": 130,
    "fried rice": 250,
    "noodles": 220,
    "egg": 155,
    "chicken": 239,
    "fish": 206,
    "beef": 250
}

# ----------------------------
# Detect food function
# ----------------------------
def detect_food(image):
    results = model(image)
    detected_items = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            detected_items.append(label)

    return list(set(detected_items))

# ----------------------------
# Initialize Session State
# ----------------------------
if "daily_calories" not in st.session_state:
    st.session_state.daily_calories = 0

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ----------------------------
# APP UI
# ----------------------------
st.title("🥗 AI Nutrition Assistant with Gemini")

# ============================
# STEP 1 — Body Info
# ============================
st.header("Step 1: Enter Your Body Info")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", 10, 100, 25)
weight = st.number_input("Weight (kg)", 30.0, 200.0, 60.0)
height = st.number_input("Height (cm)", 100.0, 250.0, 165.0)
goal = st.selectbox("Goal", ["Maintain", "Lose Weight", "Gain Weight"])

if st.button("Calculate Daily Target"):
    height_m = height / 100
    bmi = weight / (height_m ** 2)

    if gender == "Male":
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161

    if goal == "Lose Weight":
        daily_target = bmr - 500
    elif goal == "Gain Weight":
        daily_target = bmr + 500
    else:
        daily_target = bmr

    st.session_state.daily_target = int(daily_target)
    st.session_state.bmi = round(bmi, 2)

    st.success(f"BMI: {st.session_state.bmi}")
    st.success(f"Daily Calorie Target: {st.session_state.daily_target} kcal")

# ============================
# STEP 2 — Upload Food
# ============================
st.header("Step 2: Upload Food Image")

uploaded_file = st.file_uploader("Upload meal image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, width=400)

    with st.spinner("Detecting food..."):
        detected_foods = detect_food(image)

    if detected_foods:
        st.success(f"Detected: {', '.join(detected_foods)}")

        meal_calories = 0
        for food in detected_foods:
            if food in calorie_db:
                meal_calories += calorie_db[food]

        st.write(f"Estimated Meal Calories: {meal_calories} kcal")

        if st.button("Add to Daily Intake"):
            st.session_state.daily_calories += meal_calories
            st.success("Added to daily intake!")
    else:
        st.warning("No recognizable food detected.")

# ============================
# STEP 3 — Daily Summary
# ============================
st.header("Step 3: Daily Summary")

if "daily_target" in st.session_state:
    st.write(f"Daily Target: {st.session_state.daily_target} kcal")
    st.write(f"Calories Consumed: {st.session_state.daily_calories} kcal")

    remaining = st.session_state.daily_target - st.session_state.daily_calories
    st.write(f"Remaining Calories: {remaining} kcal")
else:
    st.info("Please calculate your daily target first.")

# ============================
# STEP 4 — AI Meal Recommendation
# ============================
st.header("Step 4: AI Meal Suggestion")

if st.button("Get Meal Suggestion"):
    if "daily_target" not in st.session_state:
        st.warning("Please calculate body info first.")
    else:
        prompt = f"""
        You are a certified nutritionist.
        User profile:
        - Gender: {gender}
        - Age: {age}
        - BMI: {st.session_state.bmi}
        - Goal: {goal}
        - Daily target: {st.session_state.daily_target}
        - Calories eaten today: {st.session_state.daily_calories}
        Suggest:
        1. Next meal recommendation
        2. Approximate calories
        3. Macro balance
        4. Keep suggestions suitable for Asian diet
        """

        # Create a chat session with a Gemini model (free tier model like flash)
        chat = gemini_client.chats.create(model="gemini-2.5-flash")
        gemini_resp = chat.send_message(prompt)
        st.write(gemini_resp.text)

# ============================
# STEP 5 — AI Chat Assistant
# ============================
st.header("💬 Nutrition Chat Assistant")

user_input = st.chat_input("Ask about your diet...")

if user_input:
    # Append new user query into history
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Build conversation text for Gemini
    chat_text = ""
    for msg in st.session_state.chat_history:
        chat_text += f"{msg['role']}: {msg['content']}\n"

    # Use Gemini chat session
    chat = gemini_client.chats.create(model="gemini-2.5-flash")
    gemini_resp = chat.send_message(chat_text)
    reply = gemini_resp.text

    st.session_state.chat_history.append({"role": "assistant", "content": reply})

# Display chat messages
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
