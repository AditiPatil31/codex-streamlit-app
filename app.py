import streamlit as st
import pandas as pd
import pickle
import numpy as np

# ── Load model and encoder ──
with open('xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

st.title("🔮 CodeX Price Range Predictor")
st.divider()

# ── Tabs ──
tab1, tab2, tab3 = st.tabs(["👤 Personal", "🥤 Consumption", "🏷️ Brand & Product"])

with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=80, value=25)
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with col3:
        zone = st.selectbox("Zone", ["Urban", "Metro", "Rural", "Semi-Urban"])

    col4, col5 = st.columns(2)
    with col4:
        occupation = st.selectbox("Occupation",
            ["Working Professional", "Student", "Entrepreneur", "Retired"])
    with col5:
        income_levels = st.selectbox("Income Levels",
            ["Not Reported", "<10L", "10L - 15L", "16L - 25L", "26L - 35L", "> 35L"])

with tab2:
    col6, col7, col8 = st.columns(3)
    with col6:
        consume_frequency = st.selectbox("Consume Frequency (Weekly)",
            ["0-2 times", "3-4 times", "5-7 times"])
    with col7:
        preferable_consumption_size = st.selectbox("Consumption Size",
            ["Small (250 ml)", "Medium (500 ml)", "Large (1 L)"])
    with col8:
        typical_consumption_situations = st.selectbox("Consumption Situation",
            ["Active (eg. Sports, gym)", "Social (eg. Parties)", "Casual (eg. At home)"])

    col9, col10 = st.columns(2)
    with col9:
        health_concerns = st.selectbox("Health Concerns",
            ["Low (Not very concerned)", "Medium (Moderately health-conscious)",
             "High (Very health-conscious)"])

with tab3:
    col11, col12, col13 = st.columns(3)
    with col11:
        current_brand = st.selectbox("Current Brand", ["Newcomer", "Established"])
    with col12:
        awareness_of_other_brands = st.selectbox("Awareness of Other Brands",
            ["0 to 1", "2 to 4", "above 4"])
    with col13:
        reasons_for_choosing_brands = st.selectbox("Reasons for Choosing",
            ["Price", "Quality", "Availability", "Brand Reputation"])

    col14, col15, col16 = st.columns(3)
    with col14:
        flavor_preference = st.selectbox("Flavor Preference", ["Traditional", "Exotic"])
    with col15:
        purchase_channel = st.selectbox("Purchase Channel", ["Online", "Retail Store"])
    with col16:
        packaging_preference = st.selectbox("Packaging Preference",
            ["Simple", "Premium", "Eco-Friendly"])

st.divider()

# ── Predict Button ──
predict = st.button("🔮 Predict Price Range", use_container_width=True)

if predict:
    # Step 1: Convert age to age_group
    def categorize_age(age):
        if age <= 25: return '18-25'
        elif age <= 35: return '26-35'
        elif age <= 45: return '36-45'
        elif age <= 55: return '46-55'
        elif age <= 70: return '56-70'
        else: return '70+'

    age_group = categorize_age(age)

    # Step 2: Calculate engineered features
    frequency_map = {"0-2 times": 1, "3-4 times": 2, "5-7 times": 3}
    awareness_map = {"0 to 1": 1, "2 to 4": 2, "above 4": 3}
    zone_map = {"Urban": 3, "Metro": 4, "Rural": 1, "Semi-Urban": 2}
    income_map = {"Not Reported": 0, "<10L": 1, "10L - 15L": 2,
                  "16L - 25L": 3, "26L - 35L": 4, "> 35L": 5}

    freq_score = frequency_map[consume_frequency]
    aware_score = awareness_map[awareness_of_other_brands]
    cf_ab_score = round(freq_score / (aware_score + freq_score), 2)
    zas_score = zone_map[zone] * income_map[income_levels]
    bsi = 1 if (current_brand != "Established" and
                reasons_for_choosing_brands in ["Price", "Quality"]) else 0

    # Step 3: Label encoding mappings
    age_group_map = {'18-25': 1, '26-35': 2, '36-45': 3,
                     '46-55': 4, '56-70': 5, '70+': 6}
    income_label_map = {"Not Reported": 0, "<10L": 1, "10L - 15L": 2,
                        "16L - 25L": 3, "26L - 35L": 4, "> 35L": 5}
    health_map = {"Low (Not very concerned)": 1,
                  "Medium (Moderately health-conscious)": 2,
                  "High (Very health-conscious)": 3}
    consume_map = {"0-2 times": 1, "3-4 times": 2, "5-7 times": 3}
    size_map = {"Small (250 ml)": 1, "Medium (500 ml)": 2, "Large (1 L)": 3}

    # Step 4: Create input DataFrame
    input_data = {
        'gender': gender,
        'zone': zone,
        'occupation': occupation,
        'income_levels': income_label_map[income_levels],
        'consume_frequency(weekly)': consume_map[consume_frequency],
        'current_brand': current_brand,
        'preferable_consumption_size': size_map[preferable_consumption_size],
        'awareness_of_other_brands': awareness_of_other_brands,
        'reasons_for_choosing_brands': reasons_for_choosing_brands,
        'flavor_preference': flavor_preference,
        'purchase_channel': purchase_channel,
        'packaging_preference': packaging_preference,
        'health_concerns': health_map[health_concerns],
        'typical_consumption_situations': typical_consumption_situations,
        'age_group': age_group_map[age_group],
        'cf_ab_score': cf_ab_score,
        'zas_score': zas_score,
        'bsi': bsi
    }

    input_df = pd.DataFrame([input_data])

    # Step 5: One-Hot encode categorical columns
    cat_cols = ['gender', 'zone', 'occupation', 'current_brand',
                'awareness_of_other_brands', 'reasons_for_choosing_brands',
                'flavor_preference', 'purchase_channel',
                'packaging_preference', 'typical_consumption_situations']

    encoded = encoder.transform(input_df[cat_cols])
    encoded_df = pd.DataFrame(encoded.toarray(),
                              columns=encoder.get_feature_names_out(cat_cols))

    input_df = input_df.drop(columns=cat_cols)
    input_final = pd.concat([input_df, encoded_df], axis=1)

    # Step 6: Predict
    prediction = model.predict(input_final)

    price_map = {0: "💰 ₹50 - ₹100", 1: "💰 ₹100 - ₹150",
                 2: "💰 ₹150 - ₹200", 3: "💰 ₹200 - ₹250"}

    st.success(f"Predicted Price Range: {price_map[prediction[0]]}")