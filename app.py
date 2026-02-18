import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# ---------------------------
# CONFIG
# ---------------------------

st.set_page_config(page_title="AI Calorie Tracker", layout="centered")

st.title("Smart AI Calorie Tracker")

# ---------------------------
# CALORIE DATABASE
# ---------------------------

calorie_db = {
    "banana": {"small": 80, "medium": 105, "large": 130},
    "apple": {"small": 70, "medium": 95, "large": 120},
    "pizza": {"small": 200, "medium": 285, "large": 400},
    "cheeseburger": {"small": 250, "medium": 300, "large": 450},
    "hotdog": {"small": 150, "medium": 250, "large": 350},
    "ice_cream": {"small": 130, "medium": 210, "large": 300},
}

# ---------------------------
# STEP 1: BODY INFORMATION
# ---------------------------

st.header("Step 1: Body Information")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=10, max_value=100, value=20)
height = st.number_input("Height (cm)", min_value=100, max_value=220, value=160)
weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=55)

activity_level = st.selectbox(
    "Activity Level",
    ["Sedentary", "Lightly Active", "Moderately Active", "Very Active"]
)

goal = st.selectbox(
    "Your Goal",
    ["Lose Weight", "Maintain Weight", "Gain Weight"]
)

if st.button("Calculate Results"):

    # BMI
    height_m = height / 100
    bmi = weight / (height_m ** 2)

    st.subheader(f"Your BMI: {bmi:.2f}")

    if bmi < 18.5:
        category = "Underweight"
    elif bmi < 25:
        category = "Normal weight"
    elif bmi < 30:
        category = "Overweight"
    else:
        category = "Obese"

    st.write(f"Category: {category}")

    # BMR (Mifflin-St Jeor)
    if gender == "Male":
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161

    # Activity multiplier
    activity_multipliers = {
        "Sedentary": 1.2,
        "Lightly Active": 1.375,
        "Moderately Active": 1.55,
        "Very Active": 1.725
    }

    daily_calories = bmr * activity_multipliers[activity_level]

    # Goal adjustment
    if goal == "Lose Weight":
        daily_calories -= 500
    elif goal == "Gain Weight":
        daily_calories += 500

    st.session_state["daily_calories"] = int(daily_calories)

    st.subheader(f"Recommended Daily Calories: {int(daily_calories)} kcal")

# ---------------------------
# STEP 2: FOOD IMAGE DETECTION
# ---------------------------

st.header("Step 2: Upload Food Image")

uploaded_file = st.file_uploader("Upload a food photo", type=["jpg", "jpeg", "png"])

if uploaded_file:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width="stretch")

    # Load pretrained model
    model = tf.keras.applications.MobileNetV2(weights="imagenet")

    # Preprocess image
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    # Predict
    preds = model.predict(img_array)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1)

    label = decoded[0][0][1].lower().replace(" ", "_")
    confidence = decoded[0][0][2]

    st.subheader("Detected Food:")
    st.write(f"{label} ({confidence*100:.2f}% confidence)")

    # Calorie calculation
    if label in calorie_db:
        portion = st.selectbox("Select Portion Size", ["small", "medium", "large"])
        calories = calorie_db[label][portion]

        st.session_state["eaten_calories"] = calories

        st.success(f"Estimated Calories: {calories} kcal")
    else:
        st.warning("Food detected but not in calorie database.")

# ---------------------------
# STEP 3: DAILY SUMMARY
# ---------------------------

st.header("Step 3: Daily Calorie Summary")

if "daily_calories" in st.session_state and "eaten_calories" in st.session_state:

    daily = st.session_state["daily_calories"]
    eaten = st.session_state["eaten_calories"]
    remaining = daily - eaten

    st.write(f"Daily Target: {daily} kcal")
    st.write(f"Calories Eaten: {eaten} kcal")

    progress = eaten / daily
    st.progress(min(progress, 1.0))

    if remaining > 0:
        st.success(f"You can still eat {remaining} kcal today.")
    else:
        st.error(f"You exceeded your limit by {abs(remaining)} kcal.")

else:
    st.info("Calculate your body info and upload food to see summary.")
