import openai
import os
import faiss
import numpy as np
from docx import Document
from langchain.prompts import PromptTemplate

# Set your Azure OpenAI API key and endpoint
openai.api_key = 'your-azure-openai-api-key'
endpoint = "https://your-endpoint.openai.azure.com/"

# Function to load Word documents
def load_documents_from_folder(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Process only Word documents
        if file_path.endswith('.docx'):
            doc = Document(file_path)
            doc_text = ''
            
            # Extract text from each paragraph in the Word document
            for para in doc.paragraphs:
                doc_text += para.text + '\n'
            
            documents.append(doc_text)
    
    return documents

# Function to generate embeddings from Azure OpenAI
def generate_embeddings(text):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",  # You can choose other models here
        input=text
    )
    return response['data'][0]['embedding']

# Function to set up FAISS index
def create_faiss_index(embeddings):
    embedding_matrix = np.array(embeddings).astype('float32')
    dimension = len(embedding_matrix[0])  # Dimension of the embeddings
    index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity
    index.add(embedding_matrix)
    return index

# Set up prompt template
template = "Based on the following document, answer the user's query: {document}"
prompt_template = PromptTemplate(input_variables=["document"], template=template)

# Function to get a response from GPT-4o-mini
def get_response_from_openai(prompt):
    response = openai.Completion.create(
        model="gpt-4o-mini",  # Use the GPT-4o-mini model
        prompt=prompt,
        max_tokens=150,  # Adjust based on the expected response length
        temperature=0.7  # You can fine-tune this for creativity vs accuracy
    )
    return response.choices[0].text.strip()

# Full pipeline: Load docs, create embeddings, store in FAISS, generate response
def query_respond(query, folder_path):
    # Step 1: Load documents
    documents = load_documents_from_folder(folder_path)

    # Step 2: Generate embeddings for each document
    embeddings = [generate_embeddings(doc) for doc in documents]

    # Step 3: Create FAISS index
    faiss_index = create_faiss_index(embeddings)

    # Step 4: Generate embedding for the query
    query_embedding = generate_embeddings(query)

    # Step 5: Perform similarity search in FAISS
    query_embedding = np.array([query_embedding]).astype('float32')
    D, I = faiss_index.search(query_embedding, k=1)  # Find the closest document

    # Step 6: Get the most similar document
    similar_doc = documents[I[0][0]]

    # Step 7: Create a prompt using the template
    prompt = prompt_template.format(document=similar_doc)

    # Step 8: Get response from GPT-4o-mini
    response = get_response_from_openai(prompt)
    
    return response

# Example usage:
folder_path = './docs'  # Your folder containing Word documents
user_query = "What is the main theme of the document?"
response = query_respond(user_query, folder_path)
print(response)
