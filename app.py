from PIL import Image
from PyPDF2 import PdfReader
from docx import Document
from openai import OpenAI
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# General Setup
st.set_page_config(layout='wide')
client = OpenAI(api_key='sk-proj-qokY7pcHHnewUtuaJQMCT3BlbkFJBP0p5YufG4cHv8IAFpFg')

# Format text for consistent LaTeX display in Streamlit
def format_with_latex(text):
    formatted_text = text.replace('\\', '\\\\')  # Escape backslashes for LaTeX
    return formatted_text

# Read text from uploaded files (PDF, Word, and plain text)
def read_text_from_file(file):
    try:
        if file.type == "application/pdf":
            pdf_reader = PdfReader(file)
            text = "".join([pdf_reader.pages[page_num].extract_text() for page_num in range(len(pdf_reader.pages))])
            return text
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = Document(file)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        else:
            return file.read().decode("utf-8")
    except Exception as e:
        st.error(f"Error reading file {file.name}: {e}")
        return ""

############### Lesson Curator ###############

# Curate a detailed lesson using the OpenAI API
def curate_detailed_lesson(background_info, misconceptions, remedial_handouts, topic):
    try:
        # Construct the prompt for the OpenAI API
        prompt = f"""
        Create a detailed lesson for the topic '{topic}' using the following resources along with your pre-existing knowledge.
        
        Math Background Information:
        {background_info}
        
        Misconceptions to Address:
        {misconceptions}
        
        Remedial Handouts for Class Exercise:
        {remedial_handouts}
        
        Lesson Format:
        - Lesson Overview: Summarize the information from {background_info}.
        - Introduction: Generate a story to introduce {topic} and engage students. Include a real-world example.
        - Key Terms: List key terms with definitions about {topic}.
        - Practice Problems: Create exactly 5 practice problems with the question and answer listed. Use {remedial_handouts} to guide you.
        - Hands-On Group Activity: Generate an activity with clear instructions for the teachers to follow.
        - Common Misconceptions: Select three key misconceptions from {misconceptions} to highlight.
        - Class Discussion: Summarize key ideas for the teacher to go over with the class (3 key bullet points).
        - Assessment: Assign the Remedial Question Handout.
        """
        # Make API request to OpenAI
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an elementary school teacher curating detailed math lessons for students."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4096,
            n=1,
            temperature=0.5
        )
        lesson = response.choices[0].message.content.strip()
        return lesson
    except Exception as e:
        st.error(f"Error generating lesson: {e}")
        return ""
    
# Lesson Curator UI
def lesson():
    # Title and logo side by side
    col1, col2 = st.columns([1, 10])

    # Column for logo
    with col1:
        image = Image.open("assets/KH logo.webp")
        st.image(image, width=200)  # Adjust width to fit layout

    # Column for title
    with col2:
        st.markdown("""
            <h1 style='text-align: left; margin-bottom: 0; padding-bottom: 0; font-size: 7em; line-height: 1.5;'>
                <span style='color: #1B9CE4;'>KnowledgeHook</span>
                <span style='color: black;'> | AI Exploration</span>
            </h1>
            """, unsafe_allow_html=True)
        
    st.markdown("""---""")
    st.markdown("""
        <h2 style='font-family: "Apple Chancery", cursive; color: #183172; text-align: left; font-size: 4em; line-height:0.5;'>
            Lesson Curator Tool
        </h2>
        """, unsafe_allow_html=True)
    st.markdown("\n\n")

    # File upload widgets
    background_files = st.file_uploader("Upload Math Background Sheets", accept_multiple_files=True, key="background")
    misconception_files = st.file_uploader("Upload Misconception Charts", accept_multiple_files=True, key="misconceptions")
    remedial_files = st.file_uploader("Upload Remedial Question Handouts", accept_multiple_files=True, key="remedial")
    topic = st.text_input("Enter the lesson topic (e.g., Irrational Fractions):")

    # Check if all required inputs are provided
    if background_files and misconception_files and remedial_files and topic:
        background_info = "".join([read_text_from_file(file) + "\n\n" for file in background_files])
        misconceptions = "".join([read_text_from_file(file) + "\n\n" for file in misconception_files])
        remedial_handouts = "".join([read_text_from_file(file) + "\n\n" for file in remedial_files])

        # Button to trigger lesson curation
        if st.button("Curate Detailed Lesson"):
            with st.spinner("Curating your detailed lesson..."):
                lesson = curate_detailed_lesson(background_info, misconceptions, remedial_handouts, topic)
                if lesson:
                    st.subheader("Curated Detailed Lesson")
                    st.markdown(lesson, unsafe_allow_html=True)
                else:
                    st.error("Failed to generate lesson. Please try again.")

############### Lesson Feedback Tool ###############

# Generate feedback for a lesson using the OpenAI API
def generate_feedback(lesson_text, background_info, misconceptions, remedial_handouts):
    try:
        prompt = f"""
        Provide detailed feedback for the following lesson:
        
        Teacher's Lesson:
        {lesson_text}
        
        Use the following resources along with your pre-existing knowledge to provide feedback:
        
        Math Background Information:
        {background_info}
        
        Misconceptions to Address:
        {misconceptions}
        
        Remedial Handouts:
        {remedial_handouts}
        
        Feedback Format:
        - Overall Feedback: Summarize the overall quality and effectiveness of the lesson.
        - Lesson Effectiveness: Assess how well the lesson meets its objectives and student learning outcomes.
        - Engagement Strategies: Suggest ways to make the lesson more engaging for students.
        - Content Accuracy: Identify any mistakes or inaccuracies in the lesson content.
        - Differentiation: Recommend strategies to differentiate instruction for diverse learners.
        - Assessment and Evaluation: Provide recommendations for assessing student understanding during and after the lesson.
        - Common Misconceptions: List 3 key misconceptions from the Knowledgehook chart relevant to the topic.
        - Teaching Strategies: Suggest effective ways of delivering the content.
        """
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an educational consultant providing detailed feedback on lesson plans."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4096,
            n=1,
            temperature=0.5
        )
        feedback = response.choices[0].message.content.strip()
        return feedback
    except Exception as e:
        st.error(f"Error generating feedback: {e}")

# Feedback Tool UI
def feedback_tool():
    # Title and logo side by side
    col1, col2 = st.columns([1, 10])

    # Column for logo
    with col1:
        image = Image.open("assets/KH logo.webp")
        st.image(image, width=200)

    # Column for title
    with col2:
        st.markdown("""
            <h1 style='text-align: left; margin-bottom: 0; padding-bottom: 0; font-size: 7em; line-height: 1.5;'>
                <span style='color: #1B9CE4;'>KnowledgeHook</span>
                <span style='color: black;'> | AI Exploration</span>
            </h1>
            """, unsafe_allow_html=True)
        
    st.markdown("""---""")
    st.markdown("""
        <h2 style='font-family: "Apple Chancery", cursive; color: #183172; text-align: left; font-size: 4em; line-height:0.5;'>
            Lesson Feedback Editor Tool
        </h2>
        """, unsafe_allow_html=True)
    st.markdown("\n\n")

    # Displaying each file uploader in its respective column
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        lesson_file = st.file_uploader("Upload Teacher's Lesson", accept_multiple_files=False, key="lesson")

    with col2:
        background_files = st.file_uploader("Upload Math Background Sheets", accept_multiple_files=True, key="background")

    with col3:
        misconception_files = st.file_uploader("Upload Misconception Charts", accept_multiple_files=True, key="misconceptions")

    with col4:
        remedial_files = st.file_uploader("Upload Remedial Question Handouts", accept_multiple_files=True, key="remedial")

    # Ensure the button is always at the bottom
    generate_feedback_button = st.button("Generate Feedback")

    # Only process files and generate feedback if the button is pressed
    if generate_feedback_button:
        if lesson_file and background_files and misconception_files and remedial_files:
            lesson_text = read_text_from_file(lesson_file)
            background_info = "".join([read_text_from_file(file) + "\n\n" for file in background_files])
            misconceptions = "".join([read_text_from_file(file) + "\n\n" for file in misconception_files])
            remedial_handouts = "".join([read_text_from_file(file) + "\n\n" for file in remedial_files])

            with st.spinner("Generating your detailed feedback..."):
                feedback = generate_feedback(lesson_text, background_info, misconceptions, remedial_handouts)
                if feedback:
                    st.subheader("Lesson Feedback")
                    st.markdown(feedback, unsafe_allow_html=True)
                else:
                    st.error("Failed to generate feedback. Please try again.")
        else:
            st.error("Please upload all required files to generate feedback.")

############### Student Support ###############

# Generate instructional aids based on curriculum expectations
def generate_instructional_aid(curriculum_expectations, grade_level):
    prompt = f"Create a quick instructional aid based on the following curriculum expectations: {curriculum_expectations} for grade {grade_level}."
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
                {"role": "system", "content": "You are a helpful assistant providing detailed instructional aids for students."},
                {"role": "user", "content": prompt}
            ],
        max_tokens=400,
        n=1,
        temperature=0.7
    )
    
    instructional_aid = response.choices[0].message.content.strip()
    return instructional_aid

# Student Support UI
def student_support():
    # Title and logo side by side
    col1, col2 = st.columns([1, 10])

    # Column for logo
    with col1:
        image = Image.open("assets/KH logo.webp")
        st.image(image, width=200)

    # Column for title
    with col2:
        st.markdown("""
            <h1 style='text-align: left; margin-bottom: 0; padding-bottom: 0; font-size: 7em; line-height: 1.5;'>
                <span style='color: #1B9CE4;'>KnowledgeHook</span>
                <span style='color: black;'> | AI Exploration</span>
            </h1>
            """, unsafe_allow_html=True)
        
    st.markdown("""---""")
    st.markdown("""
        <h2 style='font-family: "Apple Chancery", cursive; color: #183172; text-align: left; font-size: 4em; line-height:0.5;'>
            Student Support
        </h2>
        """, unsafe_allow_html=True)
    st.markdown("\n\n")

    st.header("Enter the curriculum expectations and grade level")
    curriculum_expectations = st.text_area("Curriculum Expectations", "Example: Explain and illustrate strategies to solve single variable linear inequalities with rational coefficients within a problem solving context.")
    grade_level = st.selectbox("Grade Level", ["Kindergarten", "Grade 1", "Grade 2", "Grade 3", "Grade 4", "Grade 5", "Grade 6", "Grade 7", "Grade 8", "Grade 9", "Grade 10", "Grade 11", "Grade 12"])

    if st.button("Generate Instructional Aid"):
        if curriculum_expectations and grade_level:
            instructional_aid = generate_instructional_aid(curriculum_expectations, grade_level)
            formatted_aid = format_with_latex(instructional_aid)
            st.subheader("Generated Instructional Aid")
            st.markdown(formatted_aid, unsafe_allow_html=True)
        else:
            st.error("Please enter all required fields.")

############### Content Editor ###############

# Generate question variations using the OpenAI API
def clone_question(question, answer):
    prompt = f"""
    You are an educational content editor. Generate 3 different but conceptually similar questions to the following question and answer pair. The new questions should test the same concept but use different numbers or scenarios.

    Original Question: {question}
    Original Answer: {answer}

    Format the output as:
    1. Question: [new similar question]
    Answer: [corresponding answer]

    2. Question: [new similar question]
    Answer: [corresponding answer]

    3. Question: [new similar question]
    Answer: [corresponding answer]
    """

    # Make the API call to generate variations
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant responsible for creating variations of educational content. The new questions should be different but test the same concept."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=700,
        n=1,
        temperature=0.7
    )

    # Extract the variations from the response and split them into manageable parts
    variations_text = response.choices[0].message.content.strip()
    variations = variations_text.split("\n\n")  # Assuming variations are separated by double newlines

    return variations

# Flag and correct errors in questions using the OpenAI API
def flag_errors(question, answer):
    prompt = (
        f"Review the following question and answer for correctness and clarity:\n\n"
        f"Question: {question}\n\n"
        f"Answer: {answer}\n\n"
        f"Tasks:\n"
        f"1. Identify any errors in the question.\n"
        f"2. Verify if the answer is correct and consistent with the question.\n"
        f"3. Provide corrected versions with explanations.\n"
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant identifying issues in educational content."},
            {"role": "user", "content": prompt}
            ],
        max_tokens=700,
        n=1,
        temperature=0.5
    )
    
    flagged_text = response.choices[0].message.content.strip()
    return flagged_text

# Content Editor UI
def content_editor():
    # Title and logo side by side
    col1, col2 = st.columns([1, 10])

    # Column for logo
    with col1:
        image = Image.open("assets/KH logo.webp")
        st.image(image, width=200)

    # Column for title
    with col2:
        st.markdown("""
            <h1 style='text-align: left; margin-bottom: 0; padding-bottom: 0; font-size: 7em; line-height: 1.5;'>
                <span style='color: #1B9CE4;'>KnowledgeHook</span>
                <span style='color: black;'> | AI Exploration</span>
            </h1>
            """, unsafe_allow_html=True)
        
    st.markdown("""---""")
    st.markdown("""
        <h2 style='font-family: "Apple Chancery", cursive; color: #183172; text-align: left; font-size: 4em; line-height:0.5;'>
            Content Editor
        </h2>
        """, unsafe_allow_html=True)
    st.markdown("\n\n")

    # Section for Cloning Questions with Answers
    st.header("Clone Questions with Answers")
    question_to_clone = st.text_area("Enter the question you want to clone:", key="clone_question")
    answer_to_clone = st.text_area("Enter the corresponding answer:", key="clone_answer")

    if st.button("Clone Question with Answer"):
        if question_to_clone and answer_to_clone:
            cloned_variations = clone_question(question_to_clone, answer_to_clone)
            st.subheader("Cloned Questions and Answers")
            for i, variation in enumerate(cloned_variations, 1):
                st.markdown(f"**Variation {i}**\n{variation}")
        else:
            st.error("Please enter both a question and an answer to clone.")

    st.markdown("""---""")  # Separator line between sections

    # Section for Flagging Errors
    st.header("Flag Errors in Question")
    question_to_flag = st.text_area("Enter the question to check for errors:", key="flag_question")
    answer_to_flag = st.text_area("Enter the answer to check for errors:", key="flag_answer")

    if st.button("Flag Errors"):
        if question_to_flag and answer_to_flag:
            flagged_feedback = flag_errors(question_to_flag, answer_to_flag)
            st.subheader("Flagged Errors and Corrections")
            st.markdown(flagged_feedback, unsafe_allow_html=True)
        else:
            st.error("Please enter both a question and an answer to check for errors.")

############### Search Engine Optimization ###############

# Load a dataset of course names for search engine functionality
def load_preprocessed_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Perform a search using TF-IDF and return the top N results
def search_resources_tfidf(query, dataframe, top_n=10):
    corpus = dataframe['CurriculumExpectation'].values
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    query_vec = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = cosine_similarities.argsort()[-top_n:][::-1]
    return dataframe.iloc[top_indices]

# Refine search results using OpenAI API
def refine_results_openai(query, resources):
    prompt = f"""
    Given the following query: "{query}", and the initial list of resources:
    {resources.to_string(index=False)}
    Provide the most relevant resources and summarize their relevance to the query. Also list the learning goals and curriculum expectations for each relevant resource.
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
                {"role": "system", "content": "You are an educational consultant assisting teachers in finding and utilizing relevant resources within the Knowledgehook platform."},
                {"role": "user", "content": prompt}
            ],
        max_tokens=1500,
        n=1,
        temperature=0.5,
    )
    refined_results = response.choices[0].message.content.strip()
    return refined_results

# SEO UI
def SEO(file_path):
    # Title and logo side by side
    col1, col2 = st.columns([1, 10])

    # Column for logo
    with col1:
        image = Image.open("assets/KH logo.webp")
        st.image(image, width=200)

    # Column for title
    with col2:
        st.markdown("""
            <h1 style='text-align: left; margin-bottom: 0; padding-bottom: 0; font-size: 7em; line-height: 1.5;'>
                <span style='color: #1B9CE4;'>KnowledgeHook</span>
                <span style='color: black;'> | AI Exploration</span>
            </h1>
            """, unsafe_allow_html=True)
        
    st.markdown("""---""")
    st.markdown("""
        <h2 style='font-family: "Apple Chancery", cursive; color: #183172; text-align: left; font-size: 4em; line-height:0.5;'>
            Search Engine Optimization
        </h2>
        """, unsafe_allow_html=True)
    st.markdown("\n\n")

    query = st.text_input("Enter your lesson topic or search query:", "")
    if file_path:
        df = load_preprocessed_data(file_path)
        if query:
            with st.spinner("Searching for relevant resources..."):
                initial_results = search_resources_tfidf(query, df)
                if not initial_results.empty:
                    refined_results = refine_results_openai(query, initial_results)
                    st.write("### Refined Search Results")
                    st.markdown(refined_results, unsafe_allow_html=True)
                else:
                    st.write("No relevant resources found. Please try a different query.")

############### Student Report Tool ###############

# Load the dataset and preprocess it
def load_data(file_path):
    data = pd.read_csv(file_path)
    data.dropna(inplace=True)
    return data

# Analyze student performance data and generate reports
def analyze_student_performance(data):
    student_reports = []
    curriculum_performance = {}

    # Group data by StudentId to analyze each student's performance
    grouped_data = data.groupby('StudentId')

    for student_id, group in grouped_data:
        strengths = []
        weaknesses = []
        misconceptions = set()
        correct_answers = group[group['BestAttemptResult'] == 1]
        incorrect_answers = group[group['BestAttemptResult'] == 0]

        # Identify strengths and weaknesses based on Learning Goals
        strength_goals = correct_answers['LearningGoal'].unique()
        weakness_goals = incorrect_answers['LearningGoal'].unique()

        # Analyze correct answers for strengths
        for goal in strength_goals:
            strengths.append(goal)
            for expectation in correct_answers[correct_answers['LearningGoal'] == goal]['CurriculumExpectation'].unique():
                if expectation not in curriculum_performance:
                    curriculum_performance[expectation] = {'correct': 0, 'incorrect': 0}
                curriculum_performance[expectation]['correct'] += 1

        # Analyze incorrect answers for weaknesses
        for goal in weakness_goals:
            weaknesses.append(goal)
            for expectation in incorrect_answers[incorrect_answers['LearningGoal'] == goal]['CurriculumExpectation'].unique():
                if expectation not in curriculum_performance:
                    curriculum_performance[expectation] = {'correct': 0, 'incorrect': 0}
                curriculum_performance[expectation]['incorrect'] += 1
            misconceptions.update(group[group['LearningGoal'] == goal]['CurriculumExpectation'].unique())

        # Generate 3 challenge questions based on strengths and weaknesses
        challenge_questions = generate_challenge_questions(strengths, weaknesses)

        # Create a report for the student
        student_report = {
            'StudentId': student_id,
            'Strengths': strengths,
            'Weaknesses': weaknesses,
            'Misconceptions': list(misconceptions),
            'ChallengeQuestions': challenge_questions
        }
        student_reports.append(student_report)

    return student_reports, curriculum_performance

# Generate custom challenge questions for a student
def generate_challenge_questions(strengths, weaknesses):
    # Placeholder function to generate 3 challenge questions
    challenge_questions = [
        {"question": "Question 1 based on strengths and weaknesses", "answer": "Answer 1"},
        {"question": "Question 2 based on strengths and weaknesses", "answer": "Answer 2"},
        {"question": "Question 3 based on strengths and weaknesses", "answer": "Answer 3"}
    ]
    return challenge_questions

# Compile and generate a student report
def generate_student_report(student_report):
    try:
        prompt = f"""
        Generate a detailed student report card based on the following information:
        
        StudentId: {student_report['StudentId']}
        Strengths: {', '.join(student_report['Strengths'])}
        Weaknesses: {', '.join(student_report['Weaknesses'])}
        Misconceptions: {', '.join(student_report['Misconceptions'])}
        
        Include the following sections:
        - What is the student doing well in (strengths)?
        - What is the student struggling with (weaknesses)?
        - What are some potential misconceptions that the student may have regarding the topic?
        - Design 3 questions that will challenge the student but that they will be able to solve based on their strengths and weaknesses:
        {student_report['ChallengeQuestions']}
        """
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an educational consultant providing detailed student report cards."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4096,
            n=1,
            temperature=0.5
        )
        report = response.choices[0].message.content.strip()
        return report
    except Exception as e:
        print(f"Error generating report for student {student_report['StudentId']}: {e}")
        return None
    
# Generate instructional aid for content editors based on curriculum performance data
def generate_editor_aid(curriculum_performance):
    try:
        prompt = f"""
        Based on the following curriculum performance data, generate a comprehensive and actionable instructional aid for content editors. This should include specific recommendations on changes to be made to the curriculum expectations, such as adding more practice questions for poorly performing areas or modifying the expectations themselves.

        Curriculum Performance Data:
        {curriculum_performance}
        """
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an educational consultant providing instructional aids for curriculum improvement."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4096,
            n=1,
            temperature=0.5
        )
        instructional_aid = response.choices[0].message.content.strip()
        return instructional_aid
    except Exception as e:
        print(f"Error generating instructional aid: {e}")
        return None

# Student Report Tool UI
def report_tool():
    # Title and logo side by side
    col1, col2 = st.columns([1, 10])

    # Column for logo
    with col1:
        image = Image.open("assets/KH logo.webp")
        st.image(image, width=200)

    # Column for title
    with col2:
        st.markdown("""
            <h1 style='text-align: left; margin-bottom: 0; padding-bottom: 0; font-size: 7em; line-height: 1.5;'>
                <span style='color: #1B9CE4;'>KnowledgeHook</span>
                <span style='color: black;'> | AI Exploration</span>
            </h1>
            """, unsafe_allow_html=True)
        
    st.markdown("""---""")
    st.markdown("""
        <h2 style='font-family: "Apple Chancery", cursive; color: #183172; text-align: left; font-size: 4em; line-height:0.5;'>
            Student Report Tool
        </h2>
        """, unsafe_allow_html=True)
    st.markdown("\n\n")

    uploaded_file = st.file_uploader("Upload Class Attempt Data CSV", type="csv")

    if uploaded_file:
        # Load and preprocess the dataset
        data = load_data(uploaded_file)
        
        # Display the first few rows of the dataset
        st.write("Dataset Preview:")
        st.write(data.head())

        # Analyze the dataset
        student_reports, curriculum_performance = analyze_student_performance(data)
        
        # Generate and display detailed student report cards
        for student_report in student_reports[:5]:  # Limit to the first 5 students for this example
            report = generate_student_report(student_report)
            if report:
                st.subheader(f"Report for Student {student_report['StudentId']}")
                st.markdown(report)
        
        # Generate and display instructional aid for content editors
        instructional_aid = generate_editor_aid(curriculum_performance)
        if instructional_aid:
            st.subheader("Instructional Aid for Content Editors")
            st.markdown(instructional_aid)
    else:
        st.write("Please upload the dataset to proceed.")

# Main application routing
def main():
    st.sidebar.title("Navigation")
    app_page = st.sidebar.radio("Go to", ["Lesson Curator", "Feedback Editor", "Student Support", "Content Editor", "SEO", "Student Report Tool"])
    
    if app_page == "Lesson Curator":
        lesson()
    elif app_page == "Feedback Editor":
        feedback_tool()
    elif app_page == "Student Support":
        student_support()
    elif app_page == "Content Editor":
        content_editor()
    elif app_page == "SEO":
        file_path = 'C:\KH-StreamlitApp\student_data.csv'
        SEO(file_path)
    elif app_page == "Student Report Tool":
        report_tool()

if __name__ == "__main__":
    main()