from openai import OpenAI
import azure.cognitiveservices.speech as speechsdk
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
import os
from dotenv import load_dotenv



load_dotenv()  # take environment variables from .env.
SPEECH_KEY = os.getenv('SPEECH_KEY')
SPEECH_REGION = os.getenv('SPEECH_REGION')
O_API_KEY = os.getenv('O_API_KEY')
endpoint = os.getenv('endpoint')
key = os.getenv('key')



import json

with open("prompts.json", "r") as file:
    # Parse the JSON data
    promptLibrary = json.load(file)

#promptLibrary["USER_INTENT_RECOGNITION"]

client = OpenAI(api_key=O_API_KEY)
global speech_config
speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)


USER_INTENT_PROMPT = """
    Refer to the context and the inferred intent in the examples 
    below: 
    
    Context: I was in a car accident yesterday and need to get my vehicle repaired. 
    Intent: File Accident Claim 
    
    Context: Can you help me with that? 
    Intent: Not known
    
    Context:Hello, I'd like to start the process of filing a claim. What information do you need from me? 
    Intent: Not known
    
    Context:I had a medical emergency last night and was hospitalized. 
    Intent: File Medical Claim

    Context:A close relative relative passed away unexpectedly. 
    Intent: File Life Claim 

    If not sure - Say 'Not Known'. 
    
    You must make sure that if the context has accident or medical then the intent must 
    include that. Even for slightest doubt Say 'Not Known'. You must infer the 
    intent from the context
    Think step by step.

"""
promptLibrary["USER_INTENT_RECOGNITION"] = USER_INTENT_PROMPT

def recognize_from_microphone(locale="en-US"):
    # This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
    # speech_config = speechsdk.SpeechConfig(subscription=os.environ.get('SPEECH_KEY'), region=os.environ.get('SPEECH_REGION'))

    speech_config.speech_recognition_language = locale  # ja-JP | en-US

    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    speech_recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config, audio_config=audio_config
    )

    print("Speak into your microphone.")
    speech_recognition_result = speech_recognizer.recognize_once_async().get()

    if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Recognized: {}".format(speech_recognition_result.text))
        return speech_recognition_result.text
    elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
        print(
            "No speech could be recognized: {}".format(
                speech_recognition_result.no_match_details
            )
        )
    elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_recognition_result.cancellation_details
        print("Speech Recognition canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))
            print("Did you set the speech resource key and region values?")
    return "NO CONTENT RECOGNIZED"


def respondtoUser(text):


    
    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)

    speech_config.speech_synthesis_voice_name = "en-US-AvaMultilingualNeural"

    # Creates a speech synthesizer using the default speaker as audio output.
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)

    # Synthesizes the received text to speech.
    # The synthesized speech is expected to be heard on the speaker with this line executed.
    result = speech_synthesizer.speak_text_async(text).get()

    # Checks result.
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        # print("Speech synthesized to speaker for text [{}]".format(text))
        return True
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech synthesis canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            if cancellation_details.error_details:
                print("Error details: {}".format(cancellation_details.error_details))
        print("Did you update the subscription info?")
    return False


def prompt_to_question(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a professional chatbot, given a input convert it to a relevant question.",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=50,  # Limit the maximum length of the generated question
        temperature=0.1,  # Controls the randomness of the generated text
        n=1,  # Generate only one response
        stop=None,  # Stop generation at a specific token
    )
    return response.choices[0].message.content


def prompt_Creation(query, prmpt, temp=0.1):

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prmpt},
            {"role": "user", "content": query},
        ],
        max_tokens=75,  # Limit the maximum length of the generated question
        temperature=temp,  # Controls the randomness of the generated text
        n=1,  # Generate only one response
        stop=None,  # Stop generation at a specific token
    )
    return response.choices[0].message.content


def extractFromForm(imageFile):
    document_analysis_client = DocumentAnalysisClient(
        endpoint=endpoint, credential=AzureKeyCredential(key)
    )
    fileBytes = imageFile.read()
    # result = document_client.begin_analyze_document(document_data)
    # poller = document_analysis_client.begin_analyze_document_from_url("prebuilt-document", formUrl)
    poller = document_analysis_client.begin_analyze_document(
        "prebuilt-document", fileBytes
    )
    result = poller.result()

    # valuesToExtract = ['ADDRESSEE','STATEMENT DATE','PATIENT NAME','HOSPITAL NAME','FROM','Laboratory General','TOTAL CHARGES','']

    computerVision_FormOCR = {}

    # print("----Key-value pairs found in document----")
    for kv_pair in result.key_value_pairs:
        if kv_pair.key and kv_pair.value:
            # print("Key '{}': Value: '{}'".format(kv_pair.key.content, kv_pair.value.content))
            # if kv_pair.key.content in valuesToExtract:
            computerVision_FormOCR[kv_pair.key.content] = kv_pair.value.content

    return computerVision_FormOCR


def semantic_search(query, chroma_store, top_k=5):

    # Perform the search on the Chroma vector store
    results = chroma_store.similarity_search(query, top_k=top_k)
    return results


def generate_response(userIntent):
    # def generate_response():
    # query = ("What are the mandatory documenst needed for Life Insuarnce Claims submission.")
    query = f"What are the mandatory documenst needed for {userIntent}."

    listOfVectorDB = {
        "disability": "chromaDB/Disability_Claim",
        "life": "chromaDB/Life_Claims",
        "death": "chromaDB/Death_Claim",
        "cancer": "chromaDB/Cancer_Claim",
        "critical": "chromaDB/Critical_Illness_Claim",
    }
    segment = ""
    userIntent = userIntent.lower()
    for k in listOfVectorDB.keys():
        if k in userIntent.lower():
            segment = listOfVectorDB[k]

            break
    print("Segment Found " + segment)
    embModel = OpenAIEmbeddings(api_key=O_API_KEY)

    intentDB = Chroma(
        persist_directory=segment,
        embedding_function=embModel,
    )

    # Perform semantic search to get relevant documents
    search_results = semantic_search(query, intentDB)

    # Extract the content from the search results
    responses = [result.page_content for result in search_results]
    sources = [result.metadata["source"].split("/")[-1] for result in search_results]
    sources = set(sources)
    docSource = " ".join(sources)

    # Combine the responses into a single context
    context = " ".join(responses)

    promptAndContext = [
        ("human", "{query}."),
        (
            "system",
            "You are an expert insurance agent for claim processing. Only refer the context provided in to answer the queryin a concise, bulleted points if necessary and professional manner,  #### Context:{context}.",
        ),
    ]
    # llm = OpenAI(api_key=O_API_KEY)
    chat = ChatOpenAI(
        model="gpt-3.5-turbo-0125", api_key=O_API_KEY, temperature=0.1
    )
    # Define the prompt template
    chatTemplate = ChatPromptTemplate.from_messages(promptAndContext)

    message = chatTemplate.format_messages(context=context, query=query)

    # Generate the prompt

    aiResp = chat.invoke(message)
    # Generate a response using the LLM

    return aiResp.content + "For further details refer: \n\n" + docSource + "\n\n"


def japaneseRag(query):

    ## LLMs and Embeddings
    chat = ChatOpenAI(
        model="gpt-3.5-turbo-0125", api_key=O_API_KEY, temperature=0.1
    )
    embModel = OpenAIEmbeddings(api_key=O_API_KEY)

    ## Invoke the Vector Database

    intentDB = Chroma(
        persist_directory="chromaDB/JapanRAG", embedding_function=embModel
    )
    # Perform semantic search to get relevant documents
    search_results = intentDB.similarity_search(query, top_k=5)

    # Extract the content from the search results
    responses = [result.page_content for result in search_results]

    # Combine the responses into a single context
    context = " ".join(responses)
    promptAndContext = [
        ("human", "Given the context:{context}, {query}."),
        (
            "system",
            """You are an expert in Japanees and English 
                         langugae with extensive knowledge of Japanees culture and economy. Your response must be in JAPANEES. Think STEP BY STEP""",
        ),
    ]
    # Define the prompt template
    chatTemplate = ChatPromptTemplate.from_messages(promptAndContext)
    message = chatTemplate.format_messages(query=query, context=context)

    # chatTemplate = ChatPromptTemplate.from_messages(promptAndContext)
    # message = chatTemplate.format_messages(context=context, query=query)

    # Generate the prompt
    aiResp = chat.invoke(message)
    # Generate a response using the LLM
    return aiResp.content
