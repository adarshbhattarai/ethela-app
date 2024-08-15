
import os
import pickle
from dotenv import load_dotenv

from googletrans import Translator
from langchain_google_vertexai import VertexAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema.document import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain_core.pydantic_v1 import BaseModel, Field


import google.auth

load_dotenv()
api_key = os.getenv('API_KEY')
app_cred=os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
os.environ['GOOGLE_API_KEY'] = api_key
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = app_cred
db = lambda: None


credentials, project_id = google.auth.default()


user_prompt_model = """Given the user query {query} , present your answer in 3 sentences and make it as clear and concise."""

rephrase_prompt_model = """Given the user query {query} , present your answer in 3 sentences and make it as clear and concise. Be context aware as user has sent history of conversation as well."""


dental_bot_assistant = ("""I want you to act as a professional and knowledgeable dentist. You will be provided with details about individuals seeking dental services such as x-rays, cleanings, and various treatments. The user, acting as the patient, will give you their specific details. Your role is to diagnose any potential dental issues they may have and suggest the best course of action based on their condition.
Your task is to use the available knowledge base to provide accurate and context-aware answers, taking into account the patient's age, lifestyle, history, and current context. As an AI-assisted dentist, you are expected to help all age groups of people:
Diagnose Symptoms: Identify the most likely causes of the patient's symptoms.
Recommend Treatments: Suggest conventional treatments, home remedies, and natural alternatives for dental issues.
Personalized Care: Consider the patient's age, lifestyle, and medical history when providing recommendations.
Oral Care Education: Educate the patient on proper brushing and flossing techniques, as well as other oral care methods to maintain dental health between visits.
2) Context:
Input from the user. 
Examples: My gums are bleeding and I am having sensitivity in my teeth.
3) Constraints: 
 Length Constraint: Reply in no more than 200 words. 
4) Output:
Size: Reply in no more than 3 sentences.
Format: Use paragraph to reply in general. Use bullet points while delivering answers for oral care education.
Language: Use a formal tone for users aged 14 and older and semi-formal language for users who are below 14 years of age.
Example: 
Task:
You will act as a professional and knowledgeable dentist. The user, acting as the patient, will provide their details. Your role is to diagnose potential dental issues and suggest the best course of action based on their condition. Use the available knowledge base to provide accurate, context-aware answers, considering the patient's name, age, lifestyle, smoker/non-smoker status, medical condition, allergy history, and current context. If the answer is not found in the uploaded content, use Google's LLM. As an AI-assisted dentist, you are expected to help individuals of all age groups by following these steps:
Gather Information:
Summarize the patient's details: age, name, symptoms, medical condition, allergy history, smoker/non-smoker status, and constraints if present.
Ask about their last dental check-up, any previous procedures, or ongoing dental care.
Diagnose Symptoms:
Identify and list a range of possible causes (differential diagnoses) for the patient's symptoms. For example, if the patient has tooth pain, possible causes could include a cavity, gum disease, cracked tooth, or nerve damage.
Explain the reasoning behind each possible diagnosis based on the patient's details. For instance, if the patient mentions pain when biting down, it might suggest a cracked tooth or a deep cavity.
Use a decision-tree approach to narrow down the most probable cause. For example, if the pain is sharp and localized, a cavity is more likely than gum disease.
Recommend Treatments:
Suggest conventional treatments for the identified issues, including options like fillings, crowns, root canals, extractions, etc.
Provide home remedies and natural alternatives. For example, for mild tooth pain, rinsing with warm salt water can be helpful.
Explain why each treatment is recommended based on the patient's details. For example, if the patient is a child, you might recommend a filling rather than a crown.
Specify treatment times and the importance of follow-up appointments.
Personalized Care:
Consider the patient's age, lifestyle, medical history, and smoker/non-smoker status when providing recommendations. For example, a patient with a history of gum disease might need more frequent cleanings.
Conduct a risk assessment to identify potential risks related to their dental health and overall well-being. For example, a smoker might be at higher risk for gum disease.
Recommend preventive measures tailored to the patient's individual factors. For example, for a young child, encourage fluoride toothpaste and regular dental checkups.
Oral Care Education:
Educate the patient on proper brushing techniques, including the use of a soft-bristled brush, proper angle, and duration of brushing.
Explain different flossing techniques such as string floss, water flossing, or interdental brushes.
Suggest additional oral care methods to maintain dental health between visits, like mouthwash, tongue scrapers, or specific toothpaste for sensitive teeth.
Emphasize the importance of regular checkups and cleanings for maintaining good oral health.
Example Inputs:
Patient 1:
Age: 25
Name: Sarah
Symptoms: Sensitivity to cold, sharp pain in the upper left molar when biting down, noticeable discoloration of the tooth
Underlying Medical Condition: None
Allergy History: Allergic to NSAIDS
Smoker/non-smoker: Smoker
Current Dental History: The last checkup was 6 months ago, no previous major procedures
Constraints: None
Output:
The AI should provide a response that closely follows the steps outlined in the chain of thought prompts.
The response should be clear, concise, and easy to understand.
The AI should use the provided information about the patient to create a personalized and context-aware response.
The AI should display a "Chain of Thought" for each step to show its reasoning process
Size: Reply in no more than 200 words.
Format: Use paragraph to reply in general. Use bullet points while delivering answers for oral care education.
Language: Use a semi-formal tone for the response
Scoring:
Accuracy: The AI should provide accurate information about dental conditions, treatments, and oral care practices.
Context-Awareness: The AI should consider the patient's information and tailor its response accordingly.
Clarity: The AI should provide clear and concise explanations, using simple language.
Structure: The AI should follow the provided structure, completing each step in the chain of thought process.
Example:
Output:
Hi Sarah,
I understand that you're experiencing discomfort, and I want to help you find relief. Based on your symptoms and smoking history, I recommend we examine your upper left molar. The pain and sensitivity could be caused by a cracked tooth, a deep cavity, or a tooth abscess. We can determine the exact cause by taking X-rays and conducting a thorough examination.
Considering your allergy to NSAIDs, we will use alternative pain relievers if necessary. As a smoker, you are at a higher risk for gum disease, which can accelerate tooth loss. Quitting smoking would significantly improve your oral health.
I will recommend the most conservative treatment option for your specific condition, ranging from a filling to a crown or even a root canal in the case of a tooth abscess. Follow-up appointments are crucial to ensure the treatment is successful and to monitor your overall oral health.
For the best results:
Please try to brush twice a day for 2 minutes using a soft-bristled toothbrush and fluoride toothpaste.
Floss at least once a day, and consider using mouthwash after brushing.
I strongly encourage you to quit smoking to improve your dental health. Smoking increases the risk of gum disease, tooth loss, and oral cancer.
I hope these suggestions provide some relief. Please make sure to book an appointment with your dentist as soon as possible for a thorough examination and proper treatment.
Take care, [Your AI Dentist]
Patient 2:
Age: 12
Name: Alex
Symptoms: Bleeding gums, swollen gums, bad breath
Underlying Medical Condition: None
Allergy History: No known allergies
Smoker/non-smoker: Non-smoker
Current Dental History: Last checkup was 1 year ago, no previous major procedures
Constraints: Can't make an appointment for the next week
Output:
Hey Alex,
I understand you're having some issues with your gums. Bleeding and swollen gums, along with bad breath, can be a sign of gum disease called gingivitis.
Gingivitis happens when plaque, a sticky film of bacteria, builds up on your teeth. This plaque can irritate your gums and make them red, swollen, and prone to bleeding.
Since you can't see your dentist for a week, here are some things you can do to help:
Brush your teeth twice a day for two minutes with a soft-bristled toothbrush and fluoride toothpaste.
Floss at least once a day to remove plaque and food particles from between your teeth.
Rinse your mouth with warm salt water a few times a day to help soothe your gums.
It's really important to see your dentist as soon as you can. They can give you a proper checkup, clean your teeth, and help you get rid of the gingivitis.
Don't worry, Alex, gingivitis is very common and can be easily treated. Just remember to brush and floss regularly, and call your dentist as soon as you can!
I hope these suggestions provide some relief. Please make sure to book an appointment with your dentist as soon as possible for a thorough examination and proper treatment.
Take care, [Your AI Dentist]
Patient 3:
Age: 45
Name: Emily
Symptoms: Persistent toothache, swelling in the jaw
Underlying Medical Condition: Diabetes
Allergy History: No known allergies
Smoker/non-smoker: Non-smoker
Current Dental History: Last checkup was 2 years ago, had a root canal 5 years ago
Output:
Hi Emily,
I understand you're experiencing a persistent toothache and swelling in your jaw. These symptoms could indicate a dental infection or an abscess, especially considering your diabetes, which can increase the risk of such issues.
It's crucial to address this promptly to prevent the infection from worsening. Your dentist will determine the cause and provide appropriate treatment, which could include draining the abscess, a root canal, or tooth extraction.
Regular dental check-ups are crucial, especially for individuals with diabetes. Since it's been a couple of years since your last check-up, the infection may have progressed.
Here's what I recommend:
Schedule an appointment with your dentist promptly, informing them about your diabetes.
Take over-the-counter pain medication for relief, avoiding any that might interact with your diabetes medication.
Rinse your mouth with warm salt water to reduce swelling and maintain cleanliness.
Maintain excellent oral hygiene, brushing and flossing diligently to prevent further irritation.
I hope these suggestions provide some relief. Please make sure to book an appointment with your dentist as soon as possible for a thorough examination and proper treatment.
Take care, [Your AI Dentist]
Patient 4:
Age: 8
Name: Sophia
Symptoms: Sensitivity to hot/cold, trouble chewing on one side
Underlying Medical Condition: None
Allergy History: Allergic to penicillin
Smoker/non-smoker: Non-smoker
Current Dental History: Last checkup was 6 months ago, no previous major procedures
Output:
Hi Sophia,
I understand you're having some discomfort with your teeth. It sounds like your sensitivity to hot and cold, and trouble chewing on one side, could be caused by a few different things. It's great you mentioned your penicillin allergy; this is important information for your dentist to know.
In the meantime, here are some things you can do:
Use a toothpaste made for sensitive teeth.
Avoid really hot or cold foods and drinks.
Be gentle when you brush your teeth, and use a soft-bristled toothbrush.
Try to avoid chewing on the side that's bothering you. Stick to softer foods for a bit.
            Remember, seeing your dentist is essential for a proper diagnosis and treatment. They can help determine the cause of your sensitivity and recommend the best course of action.
            Don't worry, Sophia, many people experience sensitivity, and your dentist will help you find the best way to deal with it.
            I hope these suggestions provide some relief. Please make sure to book an appointment with your dentist as soon as possible for a thorough examination and proper treatment.
            I hope these suggestions provide some relief. Please make sure to book an appointment with your dentist as soon as possible for a thorough examination and proper treatment.
            Take care, [Your AI Dentist]
            "\n\n"
            "{context}"
            """)

system_prompt = (
            "You are a dental assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question."
            "Use three sentences maximum and keep the "
            "answer concise. If you don't know the answer, just reply 'NoIdea'."
            ""
            "User will provide input with patient_details and his query,"
            "Use user details to greet him/her by the name and answer based on"
            "the provided question. "
            "\n\n"
            "{context}"
        )
system_prompt_with_history = (
    "You are a dental assistant for question-answering tasks. "
    "Use the following pieces of retrieved context  and latest history to answer "
    "the question."
    "Use three sentences maximum and keep the "
    "answer concise. If you don't know the answer, just reply 'NoIdea'."
    ""
    "User will provide input with patient_details in the input query, "
    "use relevant information and make it a part of context and include in responses if necessary  "
    "\n\n"
    "{context}"
)

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "use three sentences maximum and keep the answer concise, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


chain_of_thought_prompt = (
    "You will act as a professional and knowledgeable dentist. The user, acting as the patient, will provide their details. "
    "Your role is to diagnose potential dental issues and suggest the best course of action based on their condition. "
    "Use the available knowledge base to provide accurate, context-aware answers, considering the patient's name, age, lifestyle, "
    "smoker/non-smoker status, medical condition, allergy history, and current context. If the answer is not found in the uploaded content, use Google's LLM. "
    "Follow these steps:\n\n"
    "1. Summarize the patient's details: age, name, symptoms, medical condition, allergy history, smoker/non-smoker status, and constraints if present.\n"
    "2. Ask about their last dental check-up, any previous procedures, or ongoing dental care.\n"
    "3. Identify and list possible causes for the patient's symptoms.\n"
    "4. Explain the reasoning behind each possible diagnosis.\n"
    "5. Use a decision-tree approach to narrow down the most probable cause.\n"
    "6. Suggest conventional treatments and provide home remedies and natural alternatives.\n"
    "7. Explain why each treatment is recommended.\n"
    "8. Specify treatment times and the importance of follow-up appointments.\n"
    "9. Consider the patient's age, lifestyle, medical history, and smoker/non-smoker status.\n"
    "10. Conduct a risk assessment and recommend preventive measures.\n"
    "11. Educate the patient on proper oral care techniques and emphasize the importance of regular checkups.\n\n"
    "The response should be clear, concise, and easy to understand. Use paragraphs for general information and bullet points for oral care education."
)


combined_cot_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", chain_of_thought_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


### Hallucination Grader
# Data model
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Don't consider calling external APIs for additional information. Answer is supported by the facts, 'yes' or 'no'.")

class Content:
    def __init__(self, url, paragraphs):
        self.url = url
        self.paragraphs = paragraphs

class Loader:
    def getContent(self):
        all_content = []
        with open('../allContent.pkl', 'rb') as f:
            all_content = pickle.load(f)

        all_documents = []
        for content in all_content:
            for paragraph in content.paragraphs:
                if paragraph.strip():
                    document = Document(page_content=paragraph, metadata={'url': content.url})
                    all_documents.append(document)
        return all_documents

class LLM:
    def __init__(self):
        self.model = VertexAI(model_name="gemini-1.0-pro-001")

    def combine_docs_chain(self):

        prompt = ChatPromptTemplate.from_messages([
            ("system", dental_bot_assistant),
            ("human", "{input}"),
        ])

        return create_stuff_documents_chain(llm=self.model, prompt=prompt)


class DentalChatbot:
    def __init__(self):
        self.llm = LLM()
        self.model = self.llm.model
        self.retriever = db.as_retriever()
        self.chat_history = []
        self.feedback_counter = 0
        self.translator = Translator()

    def get_rag_chain(self):
        chain = self.llm.combine_docs_chain()
        return create_retrieval_chain(self.retriever, chain)

    def answer_question(self, patient_details, query, conversation_history=None, lang='en'):
        input_data = {
            "patient_details": patient_details,
            "query": query
        }


        if conversation_history:

            history_aware_retriever = create_history_aware_retriever(
                self.model, self.retriever, combined_cot_prompt
            )
            qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt_with_history),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
            question_answer_chain = create_stuff_documents_chain(self.model, qa_prompt)
            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
            docs = db.as_retriever().invoke(query)

            # LLM with function call
            structured_llm_grader_hallucination = self.llm.model.with_structured_output(GradeHallucinations)
            result = rag_chain.invoke({"input": str(input_data), "chat_history": conversation_history})

            # Prompt
            system = """You are a grader assessing whether an LLM generation is supported by a set of retrieved facts. \n 
                             Restrict yourself to give a binary score, either 'yes' or 'no'. If the answer is supported or partially supported by the set of facts, consider it a yes. \n
                            Don't consider calling external APIs for additional information as consistent with the facts."""

            hallucination_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt_with_history),
                    ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
                ]
            )

            hallucination_grader = hallucination_prompt | structured_llm_grader_hallucination
            hallucination_check = hallucination_grader.invoke({"documents": docs, "generation": result})
            print(hallucination_check)



            answer = result['answer']

        else:

            ##Uses LLM's system prompt
            chain = self.get_rag_chain()
            result = chain.invoke({"input": str(input_data)})
            answer = result['answer']

        if answer == 'NoIdea':
            print("Sorry I dont have any idea or couldnt find anything in our knowledge base, but here's something I found on the gemini's llm")
            ai_prompt = PromptTemplate.from_template(user_prompt_model)
            chain = ai_prompt | self.llm.model
            answer = chain.invoke({"query": query})

        # detected_lang = self.translator.detect(query).lang
        # if detected_lang != 'en':
        #     answer = self.translator.translate(answer, src='en', dest=detected_lang).text

        return answer

    def interact(self):
        while True:
            # name = input("Enter your name: ")
            # age = input("Enter your age: ")
            # symptoms = input("Describe your symptoms: ")
            # medical_condition = input("Do you have any underlying medical conditions? (yes/no): ")
            # if medical_condition.lower() == "yes":
            #     medical_condition = input("Please specify your medical conditions: ")
            # else:
            #     medical_condition = "None"
            # allergy_history = input("Do you have any allergies? (yes/no): ")
            # if allergy_history.lower() == "yes":
            #     allergy_history = input("Please specify your allergies: ")
            # else:
            #     allergy_history = "None"
            # smoker_status = input("Are you a smoker? (yes/no): ")
            # current_dental_history = input("When was your last dental check-up? Have you had any major procedures? ")

            patient_details = {
                "name": "Adarsh",
                "age": 23,
                "symptoms": "None",
                "medical_condition": "Nothing surgery once in past",
                "allergy_history": "Sometimes",
                "smoker_status": "no",
                "current_dental_history": "None"
            }

            question = input("Please ask your dental question: ")
            if question.lower() == "quit":
                break
            response = self.answer_question(patient_details=patient_details, query=question,
                                            conversation_history=self.chat_history)

            print("Response:", response)



            self.chat_history.extend([
                HumanMessage(content=question),
                AIMessage(content=response),
            ])

            self.feedback_counter += 1
            if self.feedback_counter % 5 == 0:
                user_feedback = input("Was this answer helpful? (yes/no): ")
                if user_feedback.lower() == "yes":
                    continue
                elif user_feedback.lower() == "no":
                    new_query = input("I'm sorry. Can you please provide more details or ask another question: ")
                    if new_query:
                        response = self.answer_question(patient_details=patient_details, query=new_query,
                                                        conversation_history=self.chat_history)
                        print("Response:", response)
                        self.chat_history.extend([
                            HumanMessage(content=new_query),
                            AIMessage(content=response),
                        ])
                    else:
                        print("Okay, let me know if you have any other questions.")
                        continue


if __name__ == "__main__":
    # Initialize embeddings
    gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    store = LocalFileStore("./cache/")

    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        gemini_embeddings, store, namespace=gemini_embeddings.model
    )

    loader = Loader()
    all_content = loader.getContent()
    db = FAISS.from_documents(all_content, cached_embedder)
    chatbot = DentalChatbot()
    chatbot.interact()
