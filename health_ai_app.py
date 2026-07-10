import streamlit as st
import numpy as np
import pandas as pd
import time
import random
from datetime import datetime

# page config 
st.set_page_config(
    page_title="AI Health Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# custom css 
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1a73e8;
        text-align: center;
        margin-bottom: 0.3rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #5f6368;
        text-align: center;
        margin-bottom: 2rem;
    }
    .card {
        background: #f8f9fa;
        border-left: 4px solid #1a73e8;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .green-card {
        background: #e8f5e9;
        border-left: 4px solid #2e7d32;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .red-card {
        background: #ffebee;
        border-left: 4px solid #c62828;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .yellow-card {
        background: #fffde7;
        border-left: 4px solid #f9a825;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .metric-box {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .report-section {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.07);
        margin-bottom: 1rem;
    }
    div[data-testid="stSidebar"] {
        background: #e8f0fe;
    }
</style>
""", unsafe_allow_html=True)


#  symptom → disease logic (rule-based, no heavy ML needed) 
DISEASE_MAP = {
    "Flu / Viral Fever": {
        "symptoms": ["fever", "body pain", "fatigue", "headache", "chills", "sore throat"],
        "advice": "Rest well, stay hydrated. Take paracetamol for fever. Consult doctor if fever exceeds 103°F.",
        "severity": "moderate"
    },
    "Common Cold": {
        "symptoms": ["runny nose", "sneezing", "sore throat", "cough", "mild fever"],
        "advice": "Steam inhalation, honey-ginger tea, rest. Should resolve in 5–7 days.",
        "severity": "mild"
    },
    "Migraine": {
        "symptoms": ["headache", "nausea", "sensitivity to light", "dizziness", "vomiting"],
        "advice": "Rest in a dark quiet room. Stay hydrated. Consider prescribed medication.",
        "severity": "moderate"
    },
    "Gastroenteritis": {
        "symptoms": ["nausea", "vomiting", "diarrhea", "stomach pain", "fever", "fatigue"],
        "advice": "ORS (oral rehydration salts), bland diet (BRAT). Avoid dairy for 2 days.",
        "severity": "moderate"
    },
    "Hypertension Risk": {
        "symptoms": ["headache", "dizziness", "chest pain", "shortness of breath", "fatigue"],
        "advice": "Monitor BP regularly. Reduce salt, stress, and caffeine. See a doctor soon.",
        "severity": "high"
    },
    "Anxiety / Stress": {
        "symptoms": ["anxiety", "restlessness", "shortness of breath", "palpitations", "fatigue", "headache"],
        "advice": "Practice deep breathing, mindfulness. Reduce caffeine. Talk to a counselor.",
        "severity": "moderate"
    },
    "Dehydration": {
        "symptoms": ["dizziness", "fatigue", "headache", "dry mouth", "dark urine", "confusion"],
        "advice": "Drink 8–10 glasses of water/day. ORS if severe. Avoid sugary drinks.",
        "severity": "mild"
    },
    "Possible COVID-19 / Respiratory Infection": {
        "symptoms": ["fever", "cough", "shortness of breath", "fatigue", "loss of taste", "body pain"],
        "advice": "Self-isolate. Get tested. Monitor oxygen levels. Seek medical attention if SpO2 < 94%.",
        "severity": "high"
    },
}

ALL_SYMPTOMS = sorted(set(
    s for d in DISEASE_MAP.values() for s in d["symptoms"]
))


def predict_disease(selected_symptoms):
    if not selected_symptoms:
        return []
    
    results = []
    for disease, info in DISEASE_MAP.items():
        match = len(set(selected_symptoms) & set(info["symptoms"]))
        total = len(info["symptoms"])
        confidence = round((match / total) * 100, 1)
        if confidence >= 25:
            results.append({
                "disease": disease,
                "confidence": confidence,
                "advice": info["advice"],
                "severity": info["severity"]
            })
    
    return sorted(results, key=lambda x: x["confidence"], reverse=True)[:3]


# BMI functions 
def calculate_bmi(weight_kg, height_cm):
    h = height_cm / 100
    bmi = weight_kg / (h ** 2)
    
    if bmi < 18.5:
        category = "Underweight"
        color = "yellow"
        tip = "Increase calorie intake with nutritious foods. Consider strength training."
    elif bmi < 25:
        category = "Normal Weight ✅"
        color = "green"
        tip = "Great! Maintain your diet and keep up regular exercise."
    elif bmi < 30:
        category = "Overweight"
        color = "yellow"
        tip = "Reduce refined carbs and sugar. 30 mins cardio daily recommended."
    else:
        category = "Obese"
        color = "red"
        tip = "Consult a doctor. Focus on diet and guided exercise. Avoid crash diets."
    
    return round(bmi, 1), category, color, tip


def fitness_plan(activity_level, goal):
    plans = {
        ("sedentary", "weight loss"): [
            "🚶 Walk 30 min every day to start",
            "🥗 Cut 300–400 calories from daily intake",
            "💧 Drink 2.5L water daily",
            "📵 Reduce screen time before bed"
        ],
        ("sedentary", "muscle gain"): [
            "🏋️ Start with bodyweight: push-ups, squats, lunges",
            "🥚 Increase protein intake (eggs, dal, paneer)",
            "😴 Sleep 7–8 hours for muscle recovery",
            "📈 Progress gradually every 2 weeks"
        ],
        ("moderate", "weight loss"): [
            "🏃 30 min cardio 5x/week (running, cycling, swimming)",
            "🍽️ Follow a calorie-deficit diet",
            "🏋️ Add strength training 2x/week",
            "📊 Track food intake with an app"
        ],
        ("moderate", "muscle gain"): [
            "🏋️ Weight training 4x/week (push/pull/legs split)",
            "🍗 Protein: 1.6–2g per kg of body weight",
            "⏰ Eat every 3–4 hours",
            "🧘 Include rest days for recovery"
        ],
        ("active", "weight loss"): [
            "🔥 HIIT training 3x/week for fat burn",
            "🥦 Focus on whole foods, avoid processed snacks",
            "📉 Track macros (not just calories)",
            "🧊 Cold shower post-workout can help"
        ],
        ("active", "muscle gain"): [
            "🏋️ Progressive overload principle – increase weights weekly",
            "💊 Consider creatine + whey protein supplements",
            "🍌 Carb-load before workouts",
            "📅 Periodize training every 6–8 weeks"
        ],
    }
    
    key = (activity_level.lower(), goal.lower())
    return plans.get(key, ["🌟 Stay consistent with exercise and balanced diet!"])


#  mental health analysis 
def analyze_mental_health(stress, sleep, mood, social, thoughts):
    score = 0
    
    # score calculation (higher = more concern)
    score += (10 - stress) * 0  # stress 1-10
    score += stress * 2          # high stress adds points
    
    if sleep < 5:
        score += 20
    elif sleep < 7:
        score += 10
    
    mood_scores = {"Very Bad": 25, "Bad": 15, "Neutral": 5, "Good": 0, "Very Good": -5}
    score += mood_scores.get(mood, 5)
    
    if social == "Isolated":
        score += 15
    elif social == "Sometimes":
        score += 5
    
    negative_thoughts = ["Always", "Most of the time"]
    if thoughts in negative_thoughts:
        score += 20
    elif thoughts == "Sometimes":
        score += 8
    
    score += stress * 1.5  # stress weight
    
    if score < 15:
        level = "Good Mental Health 😊"
        color = "green"
        suggestions = [
            "Keep up your positive habits!",
            "Continue regular social connections",
            "Maintain your sleep schedule",
            "Practice gratitude daily"
        ]
    elif score < 35:
        level = "Mild Stress / Concern 😐"
        color = "yellow"
        suggestions = [
            "Try 10-min mindfulness or meditation daily",
            "Journal your thoughts before sleeping",
            "Connect with friends or family more often",
            "Reduce caffeine intake after 2 PM",
            "Take short breaks every 90 minutes at work"
        ]
    elif score < 55:
        level = "Moderate Anxiety / Burnout Risk ⚠️"
        color = "orange"
        suggestions = [
            "Consider speaking to a counselor or therapist",
            "Set clear work-life boundaries",
            "Exercise 3–4 times per week (proven to reduce anxiety)",
            "Practice 4-7-8 breathing technique",
            "Limit social media to 30 min/day",
            "Reach out to a trusted friend or mentor"
        ]
    else:
        level = "High Stress / Please Seek Help 🚨"
        color = "red"
        suggestions = [
            " Please consult a mental health professional soon",
            "Talk to someone you trust today",
            "iCall helpline (India): 9152987821",
            "Vandrevala Foundation: 1860-2662-345 (24/7)",
            "Avoid alcohol and excessive screen time",
            "Small steps: go outside for 10 minutes today"
        ]
    
    return level, color, suggestions, min(score, 100)


# ─── SESSION STATE init ────────────────────────────────────────────────────────
if "user_name" not in st.session_state:
    st.session_state.user_name = ""
if "report_data" not in st.session_state:
    st.session_state.report_data = {}


# ════════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 👤 Your Profile")
    name = st.text_input("Your Name", placeholder="Enter your name")
    age = st.number_input("Age", min_value=5, max_value=100, value=25)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    
    st.markdown("---")
    st.markdown("###  Navigation")
    page = st.radio("Go to", [
        "🏠 Home",
        "🤒 Symptom Checker",
        "🧠 Mental Health",
        "⚖️ BMI & Fitness",
        "📋 Health Report"
    ])
    
    st.markdown("---")
    st.caption(f"📅 {datetime.now().strftime('%d %b %Y, %I:%M %p')}")
    st.caption("v1.0 | For educational use only")


# store profile
st.session_state.user_name = name
st.session_state.age = age
st.session_state.gender = gender


# ════════════════════════════════════════════════════════════════════════════════
# HOME PAGE
# ════════════════════════════════════════════════════════════════════════════════
if page == " Home":
    st.markdown('<div class="main-title"> AI Health Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Your personal health companion – symptom checker, mental health & fitness advisor</div>', unsafe_allow_html=True)
    
    greeting = f"Hello, {name}! 👋" if name else "Welcome! 👋"
    st.info(f"{greeting} Use the sidebar to navigate between modules.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="card">
        <h4> Symptom Checker</h4>
        <p>Select your symptoms and get possible diagnoses with care advice.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
        <h4> Mental Health</h4>
        <p>Assess your stress, sleep & mood. Get personalized mental wellness tips.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="card">
        <h4> BMI & Fitness</h4>
        <p>Calculate your BMI and get a custom workout + diet plan.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("###  How to Use")
    steps = [
        "Fill in your profile in the sidebar (name, age, gender)",
        "Go to **Symptom Checker** → select symptoms → get analysis",
        "Go to **Mental Health** → answer quick questions → get wellness score",
        "Go to **BMI & Fitness** → enter height/weight → get fitness plan",
        "Visit **Health Report** for a complete summary of all your results"
    ]
    for i, step in enumerate(steps, 1):
        st.markdown(f"**Step {i}:** {step}")
    

# SYMPTOM CHECKER
elif page == " Symptom Checker":
    st.markdown("##  Symptom Checker")
    st.markdown("Select the symptoms you are currently experiencing:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        symptoms_1 = st.multiselect(
            "Common Symptoms",
            ALL_SYMPTOMS[:len(ALL_SYMPTOMS)//2],
            placeholder="Search or select..."
        )
    
    with col2:
        symptoms_2 = st.multiselect(
            "Additional Symptoms",
            ALL_SYMPTOMS[len(ALL_SYMPTOMS)//2:],
            placeholder="Search or select..."
        )
    
    selected = symptoms_1 + symptoms_2
    
    duration = st.selectbox("How long have you had these symptoms?",
                             ["Just started (today)", "1–2 days", "3–5 days", "More than a week"])
    
    severity_self = st.slider("How severe do you feel? (1 = mild, 10 = very severe)", 1, 10, 4)
    
    if st.button(" Analyze Symptoms", type="primary"):
        if not selected:
            st.warning("Please select at least one symptom to continue.")
        else:
            with st.spinner("Analyzing your symptoms..."):
                time.sleep(1.2)
            
            results = predict_disease(selected)
            
            st.markdown("###  Analysis Results")
            st.success(f"Analyzed {len(selected)} symptoms. Duration: {duration}. Self-severity: {severity_self}/10")
            
            if not results:
                st.info("No strong pattern found. Symptoms may be general fatigue or stress. Monitor and see a doctor if they persist.")
            else:
                for r in results:
                    sev_color = {"mild": "green", "moderate": "yellow", "high": "red"}.get(r["severity"], "yellow")
                    card_class = {"green": "green-card", "yellow": "yellow-card", "red": "red-card"}.get(sev_color, "card")
                    
                    st.markdown(f"""
                    <div class="{card_class}">
                    <h4>🔹 {r['disease']}</h4>
                    <p><b>Match Confidence:</b> {r['confidence']}% &nbsp;|&nbsp; <b>Severity Level:</b> {r['severity'].title()}</p>
                    <p><b> Advice:</b> {r['advice']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # save to report
            st.session_state.report_data["symptoms"] = {
                "selected": selected,
                "duration": duration,
                "severity": severity_self,
                "results": results
            }
            
            st.info(" Your results have been saved. Visit **Health Report** for a full summary.")


# MENTAL HEALTH
elif page == "Mental Health":
    st.markdown("## Mental Health Assessment")
    st.markdown("Answer honestly – this is private and helps give you better suggestions.")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        stress = st.slider(" Stress Level (1 = relaxed, 10 = overwhelmed)", 1, 10, 5)
        sleep_hours = st.number_input(" Average Sleep (hours/night)", min_value=1.0, max_value=12.0, value=6.5, step=0.5)
        mood = st.select_slider(" Overall Mood (past week)",
                                 options=["Very Bad", "Bad", "Neutral", "Good", "Very Good"],
                                 value="Neutral")
    
    with col2:
        social = st.selectbox(" Social Interaction",
                               ["Socially active", "Sometimes", "Isolated"])
        negative_thoughts = st.selectbox(" Frequency of negative/anxious thoughts",
                                          ["Rarely / Never", "Sometimes", "Most of the time", "Always"])
        work_life = st.selectbox(" Work-Life Balance",
                                  ["Very balanced", "Somewhat balanced", "Often unbalanced", "Completely unbalanced"])
    
    if st.button("Assess Mental Wellness", type="primary"):
        with st.spinner("Processing your responses..."):
            time.sleep(1)
        
        level, color, suggestions, score = analyze_mental_health(
            stress, sleep_hours, mood, social, negative_thoughts
        )
        
        st.markdown("###  Your Mental Wellness Report")
        
        col_a, col_b = st.columns([1, 2])
        
        with col_a:
            st.metric("Wellness Score", f"{100 - score}/100")
            st.metric("Status", level)
        
        with col_b:
            card_class = {
                "green": "green-card",
                "yellow": "yellow-card",
                "orange": "yellow-card",
                "red": "red-card"
            }.get(color, "card")
            
            st.markdown(f'<div class="{card_class}"><h4>{level}</h4></div>', unsafe_allow_html=True)
            
            st.markdown("**Personalized Suggestions:**")
            for s in suggestions:
                st.markdown(f"- {s}")
        
        # sleep tip
        st.markdown("---")
        if sleep_hours < 6:
            st.error(f" You're sleeping only {sleep_hours}h/night. Aim for 7–9 hours. Poor sleep is linked to anxiety and depression.")
        elif sleep_hours >= 7:
            st.success(f"✅ Good sleep! {sleep_hours}h per night is within the healthy range.")
        
        st.session_state.report_data["mental"] = {
            "stress": stress,
            "sleep": sleep_hours,
            "mood": mood,
            "score": 100 - score,
            "level": level,
            "suggestions": suggestions
        }
        
        st.info(" Saved to your Health Report!")


# BMI & FITNESS
elif page == " BMI & Fitness":
    st.markdown("##  BMI Calculator & Fitness Advisor")
    
    col1, col2 = st.columns(2)
    
    with col1:
        weight = st.number_input("Weight (kg)", min_value=20.0, max_value=250.0, value=70.0, step=0.5)
        height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.5)
        activity = st.selectbox("Activity Level", ["Sedentary", "Moderate", "Active"])
    
    with col2:
        goal = st.selectbox("Your Goal", ["Weight Loss", "Muscle Gain", "General Fitness"])
        diet_pref = st.selectbox("Diet Preference", ["Vegetarian", "Non-Vegetarian", "Vegan"])
        water_intake = st.number_input("Daily Water Intake (liters)", 0.5, 6.0, 1.5, 0.25)
    
    if st.button(" Calculate BMI & Get Plan", type="primary"):
        with st.spinner("Calculating..."):
            time.sleep(0.8)
        
        bmi, category, color, tip = calculate_bmi(weight, height)
        
        st.markdown("###  Your BMI Result")
        
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("BMI", bmi)
        col_b.metric("Category", category.replace("✅", ""))
        col_c.metric("Ideal BMI", "18.5 – 24.9")
        
        card_class = {"green": "green-card", "yellow": "yellow-card", "red": "red-card"}.get(color, "card")
        st.markdown(f'<div class="{card_class}"><b>Assessment:</b> {category}<br><b>Tip:</b> {tip}</div>', unsafe_allow_html=True)
        
        # water tip
        if water_intake < 2:
            st.warning(f" You're drinking only {water_intake}L/day. Try to increase to at least 2.5L.")
        else:
            st.success(f" Good hydration! {water_intake}L/day is healthy.")
        
        # fitness plan
        st.markdown("### 🏃 Your Personalized Fitness Plan")
        plan = fitness_plan(activity, goal)
        for item in plan:
            st.markdown(f"- {item}")
        
        # diet tips
        st.markdown("###  Diet Tips")
        if diet_pref == "Vegetarian":
            diet_tips = [
                "Protein sources: Paneer, dal, rajma, eggs (if lacto-veg), tofu, quinoa",
                "Include 5 servings of fruits & vegetables daily",
                "Avoid refined flour (maida) and excess sugar",
                "Have a handful of nuts (almonds, walnuts) daily"
            ]
        elif diet_pref == "Vegan":
            diet_tips = [
                "Protein: lentils, chickpeas, tofu, tempeh, seeds",
                "Supplement B12 and Vitamin D (often deficient in vegans)",
                "Include flaxseeds or chia seeds for omega-3",
                "Eat fortified plant milks for calcium"
            ]
        else:
            diet_tips = [
                "Lean proteins: chicken breast, fish, eggs",
                "2 servings of fish per week for omega-3",
                "Avoid processed meats (sausages, salami)",
                "Balance each plate: 50% veggies, 25% protein, 25% complex carbs"
            ]
        
        for tip_item in diet_tips:
            st.markdown(f"- {tip_item}")
        
        st.session_state.report_data["bmi"] = {
            "bmi": bmi,
            "category": category,
            "weight": weight,
            "height": height,
            "goal": goal,
            "activity": activity,
            "diet": diet_pref
        }
        
        st.info(" Saved to your Health Report!")


# HEALTH REPORT
elif page == " Health Report":
    st.markdown("##  Personalized Health Report")
    
    user_display = st.session_state.user_name or "User"
    age_display = st.session_state.get("age", "—")
    gender_display = st.session_state.get("gender", "—")
    
    st.markdown(f"""
    <div class="report-section">
    <h4>👤 Patient Profile</h4>
    <p><b>Name:</b> {user_display} &nbsp;|&nbsp; <b>Age:</b> {age_display} &nbsp;|&nbsp; <b>Gender:</b> {gender_display}</p>
    <p><b>Report Generated:</b> {datetime.now().strftime('%d %B %Y, %I:%M %p')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    rd = st.session_state.report_data
    
    if not rd:
        st.warning("No data yet! Please complete at least one module (Symptom Checker / Mental Health / BMI) first.")
    else:
        # symptom section
        if "symptoms" in rd:
            s = rd["symptoms"]
            st.markdown("### Symptom Analysis")
            st.markdown(f"**Symptoms reported:** {', '.join(s['selected'])}")
            st.markdown(f"**Duration:** {s['duration']} | **Self-severity:** {s['severity']}/10")
            
            if s["results"]:
                st.markdown("**Possible Conditions:**")
                for r in s["results"]:
                    st.markdown(f"- **{r['disease']}** (Confidence: {r['confidence']}%)")
            st.markdown("---")
        
        # mental health section
        if "mental" in rd:
            m = rd["mental"]
            st.markdown("### Mental Health Summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("Stress Level", f"{m['stress']}/10")
            col2.metric("Sleep", f"{m['sleep']}h/night")
            col3.metric("Wellness Score", f"{m['score']}/100")
            st.markdown(f"**Status:** {m['level']}")
            st.markdown(f"**Mood:** {m['mood']}")
            st.markdown("---")
        
        # bmi section
        if "bmi" in rd:
            b = rd["bmi"]
            st.markdown("### BMI & Fitness Summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("BMI", b["bmi"])
            col2.metric("Category", b["category"].replace("✅", "").strip())
            col3.metric("Goal", b["goal"])
            st.markdown(f"**Height:** {b['height']} cm | **Weight:** {b['weight']} kg")
            st.markdown(f"**Activity Level:** {b['activity']} | **Diet:** {b['diet']}")
            st.markdown("---")
        
        # overall advice
        st.markdown("### 🌟 Overall Health Suggestions")
        suggestions = [
            "Stay hydrated – aim for 2.5–3L of water daily",
            "Sleep 7–9 hours every night consistently",
            "Exercise at least 150 minutes per week (any form)",
            "Eat whole foods – reduce processed and packaged foods",
            "Take a 10-minute mental break every 2 hours",
            "Schedule a doctor checkup every 6 months",
            "Stay connected with family and friends"
        ]
        for sug in suggestions:
            st.markdown(f" {sug}")
        
        st.markdown("---")
        st.error("⚠️ **Disclaimer:** This report is generated by an AI assistant for educational purposes only. It does NOT replace professional medical advice. Please consult a qualified doctor for any health concerns.")
        
        if st.button("Clear Report & Start Fresh"):
            st.session_state.report_data = {}
            st.success("Report cleared! Go complete the modules again.")
            st.rerun()
