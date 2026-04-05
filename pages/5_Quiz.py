st.title("📝 Quiz")

questions = [
    {
        "q": "What does ε represent?",
        "options": ["Charge", "Bond length", "Energy well depth", "Mass"],
        "ans": 2
    },
    {
        "q": "What does σ represent?",
        "options": ["Charge", "Distance", "Angle", "Mass"],
        "ans": 1
    }
]

score = 0

for i, q in enumerate(questions):
    choice = st.radio(q["q"], q["options"], key=i)
    if q["options"].index(choice) == q["ans"]:
        score += 1

if st.button("Submit Quiz"):
    st.success(f"Score: {score}/{len(questions)}")
