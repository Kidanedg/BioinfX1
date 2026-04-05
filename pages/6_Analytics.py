st.title("📊 Student Performance")

scores = [60, 75, 80, 90, 50]

st.line_chart(scores)

st.write("Average:", sum(scores)/len(scores))
