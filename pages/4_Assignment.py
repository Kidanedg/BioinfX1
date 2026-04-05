import random

st.title("🎓 Assignment Generator")

problem_type = st.selectbox("Select Problem", ["Bond", "Angle", "LJ"])

if problem_type == "Bond":
    r = random.uniform(1.0, 2.0)
    st.write(f"Given: r = {r:.2f}, r0 = 1.5, kb = 300")
    
    answer = st.number_input("Your Answer")
    true = 300*(r-1.5)**2

elif problem_type == "Angle":
    theta = random.uniform(90, 130)
    st.write(f"θ = {theta:.2f}, θ0 = 109.5, k = 40")
    
    answer = st.number_input("Your Answer")
    true = 40*(theta-109.5)**2

elif problem_type == "LJ":
    r = random.uniform(2.5, 6.0)
    st.write(f"r = {r:.2f}, ε=0.2, σ=3.5")
    
    answer = st.number_input("Your Answer")
    true = 4*0.2*((3.5/r)**12 - (3.5/r)**6)

if st.button("Check Answer"):
    error = abs(answer - true)
    st.write(f"Correct: {true:.2f}")
    st.write(f"Error: {error:.2f}")
