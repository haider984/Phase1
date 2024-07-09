from .serializers import *
from django.shortcuts import render
from rest_framework_simplejwt.tokens import AccessToken
from rest_framework.views import APIView
from rest_framework.response import Response
from django.utils import timezone
from django.contrib.auth import logout
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework import status
import os
from django.shortcuts import redirect
from django.http import HttpResponseRedirect
import PyPDF2
from rest_framework_simplejwt.exceptions import InvalidToken, TokenError
from rest_framework_simplejwt.tokens import RefreshToken, UntypedToken
from django.conf import settings
from rest_framework.permissions import IsAuthenticated
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from django.contrib.auth.tokens import default_token_generator
from django.utils.encoding import force_bytes, force_str
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.contrib.auth import authenticate
from django.core.mail import send_mail
from django.conf import settings
from rest_framework import generics, status
from rest_framework.response import Response
from rest_framework.views import APIView
from .models import User
from .serializers import (
    SignupSerializer, LoginSerializer, PasswordResetRequestSerializer,
    PasswordResetConfirmSerializer, DocumentQuerySerializer, ChatQuerySerializer,
    PDFUploadSerializer, ConversationHistorySerializer, ClearHistoryResponseSerializer
)
from django.template.loader import render_to_string
from django.http import HttpResponse
import logging
import jwt
import datetime
from pinecone.grpc import pinecone
from pinecone import ServerlessSpec
import pinecone
from pinecone import Pinecone, ServerlessSpec

# Initialize Pinecone and index globally
pc = Pinecone(
    api_key=settings.PINECONE_API_KEY,
    environment=settings.PINECONE_ENVIRONMENT,
   # proxy_url='http://proxy.server:3128'
)

index_name = 'finbot'

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

# Use the Pinecone index object correctly
# At the top of your file, after initializing the Pinecone client
pinecone_index = pc.Index(index_name)

# Function to generate prediction with the OpenAI API key
def generate_prediction():
    embeddings = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)
    prompt = ChatPromptTemplate.from_template("""
        context = {context}
        question = {question}
        history = {history}

        Given the provided context and conversation history, please follow these instructions:
        1. If the question can be answered using the provided context, answer it based on the context.
        2. If the question references information from the conversation history, answer it based on the history.
        3. If the question is irrelevant to both the context and the history, respond with "Sorry, I cannot answer this question. Please ask something relevant to the context or the conversation history."

        Ensure your response is clear and concise.
    """)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, api_key=settings.OPENAI_API_KEY)

    chain = {
        "context": RunnablePassthrough(),
        "question": RunnablePassthrough(),
        "history": RunnablePassthrough(),
        "prompt": prompt,
        "llm": llm,
        "parser": StrOutputParser()
    }

    memory = ConversationBufferMemory()
    return chain, memory

user_data = {}

def initialize_user_data(user_ids,document_name):
    global user_data
    print(user_ids)
    print(document_name)
    user_id=f'{user_ids}-{document_name}'
    print()
    if user_id not in user_data:
        print("Now initlize")
        chain, memory = generate_prediction()
        user_data[user_id] = {
            'chain': chain,
            'history': []  # Initialize history or modify as needed
        }

def test_email(request):
    send_mail(
        'Test Email',
        'This is a test email sent from Django.',
        settings.EMAIL_HOST_USER,
        ['recipient-email@gmail.com'],
        fail_silently=False,
    )
    return HttpResponse("Test email sent.")

def get_user_data(user_ids,document_name):
    global user_data
    user_id=f'{user_ids}-{document_name}'
    if user_id not in user_data:
        print("In initlize")
        initialize_user_data(user_ids,document_name)
        print("end nitlize")
        print(user_data)
    return user_data[user_id]['chain'], user_data[user_id]['history']

def remove_user_data(user_ids,document_name):
    global user_data
    user_id=f'{user_ids}-{document_name}'
    if user_id in user_data:
        del user_data[user_id]

def index(request):
    return render(request, 'myapp/index.html')


class SignupView(generics.CreateAPIView):
    serializer_class = UserSerializer

    def create(self, request, *args, **kwargs):
        data = request.data.copy()
        essential_fields = {'username', 'email', 'password'}
        provided_fields = set(data.keys())
        is_free = provided_fields == essential_fields

        data['is_free'] = is_free

        serializer = self.get_serializer(data=data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()

        token = RefreshToken.for_user(user).access_token

        uid = urlsafe_base64_encode(force_bytes(user.pk))
        confirm_url = f"{settings.FRONTEND_URL}/confirm-email/{uid}/{token}/"

        try:
            send_mail(
                'Confirm your email address',
                f'Please click the following link to confirm your email address: {confirm_url}',
                settings.EMAIL_HOST_USER,
                [user.email],
                fail_silently=False,
            )
        except Exception as e:
            logger.error(f"Failed to send confirmation email: {e}")
            return Response({'detail': f'Account created, but failed to send confirmation email: {e}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        account_type = 'free' if is_free else 'premium'
        return Response({
            'detail': f'{account_type.capitalize()} account created. Please check your email for verification.',
            'account_type': account_type,
            'unique_id': user.unique_id
        }, status=status.HTTP_201_CREATED)

class ConfirmEmailView(generics.GenericAPIView):
    def get(self, request, uidb64, token):
        try:
            uid = force_str(urlsafe_base64_decode(uidb64))
            user = User.objects.get(pk=uid)
        except (TypeError, ValueError, OverflowError, User.DoesNotExist):
            return Response({'error': 'Invalid activation link'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            UntypedToken(token)
        except (InvalidToken, TokenError):
            return Response({'error': 'Invalid or expired token'}, status=status.HTTP_400_BAD_REQUEST)

        if user is not None:
            user.is_active = True
            user.email_confirmed = True
            user.save()
            logger.info("Email confirmed, redirecting to login page.")
            # Redirect to the frontend login page
            return HttpResponseRedirect('http://localhost:5173/Login')
        else:
            return Response({'error': 'Invalid activation link'}, status=status.HTTP_400_BAD_REQUEST)

class LoginView(APIView):
    def post(self, request):
        email = request.data.get('email')
        password = request.data.get('password')

        try:
            user = User.objects.get(email=email)
            if user.check_password(password):
                if user.is_active:
                    refresh = RefreshToken.for_user(user)
                    access_token = str(refresh.access_token)
                    refresh_token = str(refresh)

                    # Update user model with tokens and login status
                    user.access_token = access_token
                    user.refresh_token = refresh_token
                    user.is_logged_in = True
                    user.save()

                    # Include is_free variable in the response
                    return Response({
                        'refresh': refresh_token,
                        'access': access_token,
                        'is_free': user.is_free,
                        'unique_id':user.unique_id
                    }, status=status.HTTP_200_OK)
                else:
                    return Response({'error': 'Please verify your email address before logging in'}, status=status.HTTP_401_UNAUTHORIZED)
            else:
                return Response({'error': 'Invalid credentials'}, status=status.HTTP_401_UNAUTHORIZED)
        except User.DoesNotExist:
            return Response({'error': 'Invalid credentials'}, status=status.HTTP_401_UNAUTHORIZED)

class PasswordResetRequestView(generics.CreateAPIView):
    serializer_class = PasswordResetRequestSerializer

    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        email = serializer.validated_data['email']

        try:
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            user = None

        if user:
            # Generate reset token and URL (similar to your existing logic)
            uid = urlsafe_base64_encode(force_bytes(user.pk))
            token = default_token_generator.make_token(user)
            reset_url = f"{settings.FRONTEND_URL}/reset-password/{uid}/{token}/"

            # Send password reset email (similar to your existing logic)
            mail_subject = 'Password Reset Request'
            message = render_to_string('password_reset_email.html', {
                'user': user,
                'reset_url': reset_url,
            })
            send_mail(mail_subject, message, settings.EMAIL_HOST_USER, [user.email])

            return Response({'detail': 'Password reset email has been sent'}, status=status.HTTP_200_OK)
        else:
            return Response({'detail': 'If the email address is in our database, you will receive a password reset email shortly.'}, status=status.HTTP_200_OK)

class PasswordResetConfirmView(generics.GenericAPIView):
    serializer_class = PasswordResetConfirmSerializer

    def post(self, request, uidb64, token, *args, **kwargs):
        try:
            uid = force_str(urlsafe_base64_decode(uidb64))
            user = User.objects.get(pk=uid)
        except (TypeError, ValueError, OverflowError, User.DoesNotExist):
            user = None

        if user is not None and default_token_generator.check_token(user, token):
            serializer = self.get_serializer(data=request.data)
            serializer.is_valid(raise_exception=True)
            password = serializer.validated_data['password']
            user.set_password(password)
            user.save()
            return Response({'detail': 'Password reset successful'}, status=status.HTTP_200_OK)
        else:
            return Response({'error': 'Invalid or expired token'}, status=status.HTTP_400_BAD_REQUEST)

class LogoutView(APIView):
    def post(self, request):
        refresh_token = request.data.get('refresh_token')
        if refresh_token:
            try:
                user = User.objects.get(refresh_token=refresh_token)
                user.access_token = None
                user.refresh_token = None
                user.is_logged_in = False
                user.save()
                return Response({'message': 'Logged out successfully'}, status=status.HTTP_200_OK)
            except User.DoesNotExist:
                return Response({'error': 'Invalid refresh token'}, status=status.HTTP_401_UNAUTHORIZED)
        else:
            return Response({'error': 'Refresh token is required'}, status=status.HTTP_400_BAD_REQUEST)

# class DocumentQueryView(APIView):
#     parser_classes = [MultiPartParser, FormParser]

#     def post(self, request, *args, **kwargs):
#         file_obj = request.FILES.get('file')
#         user_id = request.data.get('user_id')
#         document_name = request.data.get('document_name')

#         if not file_obj or not user_id or not document_name:
#             raise ValueError('Both file,document_name and user_id are required')

#         pc = Pinecone(
#             api_key=settings.PINECONE_API_KEY,
#             environment=settings.PINECONE_ENVIRONMENT
#         )

#         index_name = f'{user_id}-{document_name}'

#         if index_name not in pc.list_indexes().names():
#             pc.create_index(
#                 name=index_name,
#                 dimension=1536,
#                 metric='cosine',
#                 spec=ServerlessSpec(
#                     cloud='aws',
#                     region='us-east-1'
#                 )
#             )
#         pinecone_index = pc.Index(index_name)

#         def extract_text(self, file_path):
#             file_extension = os.path.splitext(file_path)[1].lower()
#             if file_extension == '.pdf':
#                 return self.extract_text_from_pdf(file_path)
#             elif file_extension == '.txt':
#                 with open(file_path, 'r', encoding='utf-8') as file_obj:
#                     return file_obj.read()
#             elif file_extension == '.docx':
#                 doc = docx.Document(file_path)
#                 return "\n".join([paragraph.text for paragraph in doc.paragraphs])
#             elif file_extension == '.doc':
#                 return textract.process(file_path).decode('utf-8')
#             else:
#                 raise ValueError(f"Unsupported file format: {file_extension}")


#         try:
#             # # Read PDF
#             # pdf_reader = PyPDF2.PdfReader(file_obj)
#             # text = ""
#             # for page in pdf_reader.pages:
#             #     text += page.extract_text()

#             text = self.extract_text(temp_file.name)

#             # Split text
#             text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
#             texts = text_splitter.split_text(text)

#             # Generate embeddings
#             embeddings = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)
#             vectors = [embeddings.embed_query(chunk) for chunk in texts]

#             # Prepare data for Pinecone
#             vector_data = [
#                 (f"{user_id}_{i}", vec, {"text": chunk})
#                 for i, (vec, chunk) in enumerate(zip(vectors, texts))
#             ]

#             # Upsert to Pinecone
#             pinecone_index.upsert(vectors=vector_data)

#             return Response({'message': f'Processed and stored {len(vectors)} embeddings for user {user_id}'}, status=status.HTTP_200_OK)

#         except Exception as e:
#             print(f"Error: {str(e)}")
#             raise


from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
import os
import tempfile
import PyPDF2
# import docx
# import textract
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
import logging

logger = logging.getLogger(__name__)

class DocumentQueryView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, *args, **kwargs):
        file_obj = request.FILES.get('file')
        user_id = request.data.get('user_id')
        document_name = request.data.get('document_name')

        if not file_obj or not user_id or not document_name:
            raise ValueError('Both file, document_name, and user_id are required')

        pc = Pinecone(
            api_key=settings.PINECONE_API_KEY,
            environment=settings.PINECONE_ENVIRONMENT
        )

        index_name = f'{user_id}-{document_name}'

        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=1536,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
        pinecone_index = pc.Index(index_name)

        try:
            # Save the uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_obj.name)[1]) as temp_file:
                for chunk in file_obj.chunks():
                    temp_file.write(chunk)
                temp_file_path = temp_file.name
                logger.debug(f'Temporary file created at: {temp_file_path}')

            # Extract text from the file
            text = self.extract_text(temp_file_path)
            logger.debug(f'Extracted text: {text[:100]}')  # Log first 100 characters of extracted text

            # Split text
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            texts = text_splitter.split_text(text)

            # Generate embeddings
            embeddings = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)
            vectors = [embeddings.embed_query(chunk) for chunk in texts]

            # Prepare data for Pinecone
            vector_data = [
                (f"{user_id}_{i}", vec, {"text": chunk})
                for i, (vec, chunk) in enumerate(zip(vectors, texts))
            ]

            # Upsert to Pinecone
            pinecone_index.upsert(vectors=vector_data)

            return Response({'message': f'Processed and stored {len(vectors)} embeddings for user {user_id}'}, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error: {str(e)}")
            raise

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def extract_text(self, file_path):
        file_extension = os.path.splitext(file_path)[1].lower()
        logger.debug(f'File extension: {file_extension}')
        if file_extension == '.pdf':
            return self.extract_text_from_pdf(file_path)
        # elif file_extension == '.txt':
        #     with open(file_path, 'r', encoding='utf-8') as file_obj:
        #         return file_obj.read()
        # elif file_extension == '.docx':
        #     doc = docx.Document(file_path)
        #     return "\n".join([paragraph.text for paragraph in doc.paragraphs])
        # elif file_extension == '.doc':
        #     return textract.process(file_path).decode('utf-8')
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

    def extract_text_from_pdf(self, file_path):
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)  # Use PdfReader instead of PdfFileReader
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text


from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate

class PineconeRetriever:
    def __init__(self, embeddings, index):
        self.embeddings = embeddings
        self.index = index

    def __call__(self, query):
        query_embedding = self.embeddings.embed_query(query)
        similar_embeddings = self.index.query(vector=query_embedding, top_k=5, include_metadata=True)
        print(similar_embeddings)
        similar_docs = "\n\n".join(embedding['metadata']['text'] for embedding in similar_embeddings['matches'])
        print(similar_docs)
        return similar_docs

class ChatQueryView(APIView):
    def post(self, request, *args, **kwargs):
        serializer = ChatQuerySerializer(data=request.data)
        if not serializer.is_valid():
            print(f"Serializer Error: {serializer.errors}")
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        try:
            query = serializer.validated_data['query']
            document_name = serializer.validated_data['document_name']
        except KeyError:
            print("Error: 'query' field is missing from the validated data")
            return Response({'error': 'Query is required'}, status=status.HTTP_400_BAD_REQUEST)

        user_id = kwargs.get('user_id')
        if not user_id:
            return Response({'error': 'User ID is required'}, status=status.HTTP_400_BAD_REQUEST)

        print(user_id)
        user_chain, user_history = get_user_data(user_id,document_name)
        index_name = f'{user_id}-{document_name}'
        pinecone_index = pc.Index(index_name)
        try:
            # Use PineconeRetriever to get similar documents
            embeddings = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)
            retriever = PineconeRetriever(embeddings, pinecone_index)
            similar_docs = retriever(query)

            # Prepare the prompt template
            prompt = ChatPromptTemplate.from_template("""
                Use the following pieces of context to answer the question at the end. 
                If you don't know the answer, just say that you don't know, don't try to make up an answer.

                Context: {context}
                History: {history}
                Question: {question}

                Answer:
            """)

            # Set up the RAG chain
            rag_chain = (
                {"context": lambda x: similar_docs, "question": RunnablePassthrough(), "history": lambda x: user_history}
                | prompt
                | user_chain["llm"]
                | StrOutputParser()
            )

            # Invoke the chain
            response = rag_chain.invoke(query)

            # Update user history
            user_history.append({"role": "user", "content": query})
            user_history.append({"role": "assistant", "content": response})

            return Response({'response': response}, status=status.HTTP_200_OK)
        except Exception as e:
            print(f"Error in chat processing: {str(e)}")
            return Response({'error': 'Error processing chat query'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class CustomReportView(APIView):
    def post(self, request, *args, **kwargs):
        serializer = CustomReportSerializer(data=request.data)
        if not serializer.is_valid():
            print(f"Serializer Error: {serializer.errors}")
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        try:
            query = serializer.validated_data['query']
            document_name = serializer.validated_data['document_name']
        except KeyError:
            print("Error: 'query' field is missing from the validated data")
            return Response({'error': 'Query is required'}, status=status.HTTP_400_BAD_REQUEST)

        user_id = kwargs.get('user_id')
        if not user_id:
            return Response({'error': 'User ID is required'}, status=status.HTTP_400_BAD_REQUEST)

        print(user_id)
        user_chain, user_history = get_user_data(user_id,document_name)
        index_name = f'{user_id}-{document_name}'
        pinecone_index = pc.Index(index_name)
        try:
            # Use PineconeRetriever to get similar documents
            embeddings = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)
            retriever = PineconeRetriever(embeddings, pinecone_index)
            similar_docs = retriever(query)

            # Prepare the prompt template
            prompt = ChatPromptTemplate.from_template(""" You need to create report
                Use the following pieces of context and instruction mentioned use information provied by user replace it with content. 
                If you don't know the answer, just say that you don't know, don't try to make up an answer.

                Context: {context}
                History: {history}
                Question: {question}

                Answer:
            """)

            # Set up the RAG chain
            rag_chain = (
                {"context": lambda x: similar_docs, "question": RunnablePassthrough(), "history": lambda x: user_history}
                | prompt
                | user_chain["llm"]
                | StrOutputParser()
            )

            # Invoke the chain
            response = rag_chain.invoke(query)

            # Update user history
            user_history.append({"role": "user", "content": query})
            user_history.append({"role": "assistant", "content": response})

            return Response({'response': response}, status=status.HTTP_200_OK)
        except Exception as e:
            print(f"Error in chat processing: {str(e)}")
            return Response({'error': 'Error processing chat query'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import PDFUploadSerializer
import os
import tempfile
import PyPDF2
# import docx
# import textract
import fitz  # PyMuPDF
from openai import OpenAI
from django.conf import settings
import tiktoken

class PDFSummaryView(APIView):
    def post(self, request, *args, **kwargs):
        serializer = PDFUploadSerializer(data=request.data)
        if serializer.is_valid():
            uploaded_file = serializer.validated_data['file']
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            
            # Create a temporary file with the appropriate extension
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                for chunk in uploaded_file.chunks():
                    temp_file.write(chunk)
            
            try:
                # Extract text from the file
                extracted_text = self.extract_text(temp_file.name)
                
                # Generate summary
                summary = self.extract_summary(extracted_text)
                
                # Clean up the temporary file
                os.unlink(temp_file.name)
                
                return Response({'summary': summary}, status=status.HTTP_200_OK)
            except Exception as e:
                return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    def extract_text(self, file_path):
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == '.pdf':
            return self.extract_text_from_pdf(file_path)
        # elif file_extension == '.txt':
        #     with open(file_path, 'r', encoding='utf-8') as file_obj:
        #         return file_obj.read()
        # elif file_extension == '.docx':
        #     doc = docx.Document(file_path)
        #     return "\n".join([paragraph.text for paragraph in doc.paragraphs])
        # elif file_extension == '.doc':
        #     return textract.process(file_path).decode('utf-8')
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    def extract_text_from_pdf(self, pdf_path):
        pdf_document = fitz.open(pdf_path)
        text = ""
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            text += page.get_text()
        return text
    
    def split_text_into_chunks(self, text, max_tokens):
        tokenizer = tiktoken.get_encoding("gpt2")
        tokens = tokenizer.encode(text)
        chunks = []
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
        return chunks
    
    def generate_summary_for_chunks(self, chunks):
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        summaries = []
        for chunk in chunks:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": f"Generate a summary from the following text:\n\n{chunk}",
                    }
                ],
                model="gpt-3.5-turbo",
            )
            summaries.append(chat_completion.choices[0].message.content.strip())
        return summaries
    
    def extract_summary(self, text):
        max_tokens = 16385  # Set max tokens limit for each chunk
        chunks = self.split_text_into_chunks(text, max_tokens)
        chunk_summaries = self.generate_summary_for_chunks(chunks)
        combined_summary = " ".join(chunk_summaries)

        # Generate a final summary if the combined summary is still too long
        tokenizer = tiktoken.get_encoding("gpt2")
        if len(tokenizer.encode(combined_summary)) > max_tokens:
            final_summary_chunks = self.split_text_into_chunks(combined_summary, max_tokens)
            final_summary = self.generate_summary_for_chunks(final_summary_chunks)
            return " ".join(final_summary)

        return combined_summary

class ConversationHistoryView(APIView):
    def get(self, request, user_id, format=None):
        conversation_history = request.session.get(f'full_history_{user_id}', [])
        serializer = ConversationHistorySerializer({'conversation_history': conversation_history})
        return Response(serializer.data)

class ClearConversationHistoryView(APIView):
    def get(self, request, user_id, format=None):
        request.session[f'conversation_history_{user_id}'] = []
        request.session[f'full_history_{user_id}'] = []
        remove_user_data(user_id)
        serializer = ClearHistoryResponseSerializer({'status': 'Conversation history cleared successfully'})
        return Response(serializer.data)

