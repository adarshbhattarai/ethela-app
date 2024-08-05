
import os
import pickle
from dotenv import load_dotenv
import vertexai

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
from nltk import WordNetLemmatizer, word_tokenize
from nltk.corpus import stopwords

from app.prompts.prompts import contextualize_q_system_prompt, dental_bot_assistant, system_prompt_with_history, \
    user_prompt_model
from app.models.content import Content
### Download NLTK and SpaCy Resources and configure vertex ai




load_dotenv()
api_key = os.getenv('API_KEY')
os.environ['GOOGLE_API_KEY'] = api_key
#project_id = "mimetic-fulcrum-407320"
project_id = "southern-idea-407320"
vertexai.init(project=project_id, location="us-east1")



class MyCustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            module = "app.models.content"
        return super().find_class(module, name)

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Don't consider calling external APIs for additional information. Answer is supported by the facts, 'yes' or 'no'.")

class Loader:

    # lemmatizer = WordNetLemmatizer()
    # def preprocess_text(text):
    #     tokens = word_tokenize(text)
    #     lemmatized_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens]
    #     return " ".join(lemmatized_tokens)
    #
    # stop_words = set(stopwords.words('english'))
    # def remove_stopwords(text):
    #     tokens = word_tokenize(text)
    #     filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    #     return " ".join(filtered_tokens)

    def getContent(self):
        all_content = []
        with open('allContent.pkl', 'rb') as f:
            unpickler = MyCustomUnpickler(f)
            all_content = unpickler.load()

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

        detected_lang = self.translator.detect(query).lang
        if detected_lang != 'en':
            answer = self.translator.translate(answer, src='en', dest=detected_lang).text

        return answer

    def interact(self):
        while True:
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