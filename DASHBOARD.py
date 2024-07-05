import streamlit as st
import util, json
import time,os
from dotenv import load_dotenv
from prompts import IntentPrompt
from dotenv import load_dotenv



load_dotenv()  # take environment variables from .env.
SPEECH_KEY = os.getenv('SPEECH_KEY')
SPEECH_REGION = os.getenv('SPEECH_REGION')
O_API_KEY = os.getenv('O_API_KEY')
endpoint = os.getenv('endpoint')
key = os.getenv('key')

# Example usage:
prompt_instance = IntentPrompt()
# Title of the web app
st.title("Autobots - Kaizen Claims")
#st.write(prompt_instance.USER_INTENT_PROMPT)

@st.cache_data
def extractTextFromImage(uploaded_file_bytes):
    # uploaded_file_bytes = st.file_uploader("Choose a file")
    if uploaded_file_bytes is not None:
        with st.spinner("Image being scanned for details..."):
            extractedJson = util.extractFromForm(uploaded_file_bytes)
            st.write(extractedJson)
            st.success("Done!")
        uploaded_file_bytes = None


# Input text box

with open("prompts.json", "r") as file:
    # Parse the JSON data
    promptLibrary = json.load(file)

# promptLibrary['USER_INTENT_RECOGNITION']

# Slider
# age = st.slider("Select your age", 0, 100, 25)
with st.container(border=True):
    options = ["en-US", "ja-JP"]
    selected_option = st.selectbox("Choose user language.", options)

    # Display the selected option
    st.write(f"User Language: {selected_option}")
    # Button
    step = 0
    if st.button("Context Recognition"):
        st.write(SPEECH_REGION)
        util.respondtoUser("Hello! Welcome to Kaizen Cliams - How may I assist you.")
        st.write("Please speak.")
        userCall = util.recognize_from_microphone(selected_option)
        # st.write("STEP "+str(step) + " : "+userCall)
        # st.write("Strat Intent Extraction.")
        #userIntent = util.prompt_Creation(
        #    userCall, promptLibrary["USER_INTENT_RECOGNITION"], 0.2
        #)
        userIntent = util.prompt_Creation(
            userCall, prompt_instance.USER_INTENT_PROMPT, 0.1
        )
        st.write(userIntent)

        ## RESPOND BASED ON CONTEXT
        contextualresponse = util.prompt_Creation(
            userCall,prompt_instance.EMPATHY, 0.1
        )
        st.write(contextualresponse)
        util.respondtoUser(contextualresponse)


        st.write()
        st.write()

        ## RETRY TO GATHER USER INTENT
        # if userIntent == "Not Known":

        if "not known" in userIntent.lower():
            gatheruserIntent = util.prompt_Creation(
                "Not Known", promptLibrary["EXTRACT_INTENT"], 0.1
            )
            st.write(gatheruserIntent)
            
            #util.respondtoUser(gatheruserIntent)
            #userCall_Retry = util.recognize_from_microphone(selected_option)
            #userIntentRetry = util.prompt_Creation(
            #    userCall_Retry, promptLibrary["USER_INTENT_RECOGNITION"], 0.2
            #)
            #st.write(userIntentRetry)
        else:
            with st.spinner("Extracting Intent Relevant Questions"):
                # st.write(util.generate_response())
                st.write(util.generate_response(userIntent))

with st.container(border=True):
    if st.button("Natural Conversation with user"):
        ## START NATURAL DIALOGUE
        messages = st.container()

        with open("questions.json", "r") as file:
            # Parse the JSON data
            listOfQues = json.load(file)
        # print(promptLibrary)
        listOfKeyPhrases = {}
        for ques in listOfQues:
            listOfKeyPhrases[ques] = util.prompt_to_question(ques)

        extractKYC = "You are a expert call centre agent and have the folloing text :"
        whatWeNeed = " extract name, date , age, gender,number from the text"
        receivedKYC = {}
        for kyc, question in listOfKeyPhrases.items():
            # st.write('Respond to queries')
            messages.chat_message("assistant").write(question)
            util.respondtoUser(question)
            st.write("Please speak.")
            userResponse = util.recognize_from_microphone()
            messages.chat_message("user").write(userResponse)
            kyc_FromResponse = util.prompt_Creation(
                "Extract " + kyc + " from the text",
                extractKYC
                + userResponse
                + """ just return extracted value from the text. 
                
                ###
                Example 
                kyc:'My name is Jack Sparrow', 
                jack sparrow

                kyc:'It happened on 21st of Feb 2022'
                21/02/2022
                  
                  """,
                0.1,
            )
            receivedKYC[kyc] = kyc_FromResponse
        st.write(receivedKYC)


with st.container(border=True):
    uploaded_file_bytes = st.file_uploader("Upload bills or invoices.")
    extractTextFromImage(uploaded_file_bytes)


with st.container(border=True):
    options = ["ja-JP", "en-US"]
    selected_option = st.selectbox("Choose user language.", options)

    # Display the selected option
    st.write(f"User Language: {selected_option}")
    # Button
    step = 0
    if st.button("RAG For Japanese Language:"):
        messages = st.container()
        question = "Please ask your query!"
        messages.chat_message("assistant").write(question)
        util.respondtoUser(question)
        st.write("Please speak.")
        userQuery = userCall_Retry = util.recognize_from_microphone(selected_option)

        st.write("User Query: " + userQuery)

        with st.spinner("Querying RAG"):
            start_time = time.time()
            returnedResponse = util.japaneseRag(userQuery)
            end_time = time.time()
            with st.container(border=True):
                st.markdown(
                    "TOTAL TIME QUERYING RAG "
                    + str(round(end_time - start_time, 2))
                    + " sec."
                )
            st.write(returnedResponse)
