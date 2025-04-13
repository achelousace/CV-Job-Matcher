import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.tools import Tool, StructuredTool
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_community.document_loaders import PlaywrightURLLoader
from pydantic import BaseModel, Field
from typing import Optional, Union

# Set your OpenAI API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key

class CVJobMatcher:
    def __init__(self, file_path, job_url, api_key=None):
        # Set up OpenAI API key
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables or provided as argument")
        
        os.environ["OPENAI_API_KEY"] = self.api_key
        
        # Set paths
        self.file_path = file_path
        self.job_url = job_url
        
        # Initialize vectorstores and retrievers
        self.url_vectorstore = self.process_url_to_vectordb(self.job_url)
        self.url_retriever = self.url_vectorstore.as_retriever()
        
        self.vectorstore = self.process_file_to_vectordb(self.file_path)
        self.retriever = self.vectorstore.as_retriever()
        
        # Create LLM
        self.llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-4",
            openai_api_key=self.api_key,
        )
        
        # Global variable to store job analyzer output
        self.job_analyzer_agent_output = ""
        
        # Set up tools and agents
        self.setup_tools_and_agents()
    
    def process_file_to_vectordb(self, file_path, chunk_size=1000, chunk_overlap=200):
        # Load the document
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create vector store using OpenAI embeddings
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)
        
        return vectorstore
    
    def read_raw_file(self, file_path_input=None):
        """Read the entire document without chunking."""
        path_to_use = file_path_input or self.file_path
        try:
            loader = PyPDFLoader(path_to_use)
            documents = loader.load()
            # Combine all pages into one text
            full_text = "\n".join([doc.page_content for doc in documents])
            return full_text
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    def process_url_to_vectordb(self, url, chunk_size=1000, chunk_overlap=200):
        # Load the document
        url_loader = PlaywrightURLLoader(urls=[url])
        url_documents = url_loader.load()
        
        # Split the document into chunks
        url_text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        url_chunks = url_text_splitter.split_documents(url_documents)
        
        # Create vector store using OpenAI embeddings
        url_embeddings = OpenAIEmbeddings()
        url_vectorstore = FAISS.from_documents(url_chunks, url_embeddings)
        
        return url_vectorstore
    
    def extract_job_requirements(self, query=""):
        """Extract the key job requirements from the URL."""
        job_docs = self.url_retriever.get_relevant_documents("job requirements skills experience qualifications")
        job_text = "\n".join([doc.page_content for doc in job_docs])
        
        prompt = f"""
        Analyze the following job posting content and extract all important requirements:
        
        {job_text}
        
        Extract and list:
        1. Technical skills required
        2. Experience requirements
        3. Education requirements
        4. Any other important requirements
        
        Be thorough and specific. This will be used to match a CV to the job.
        """
        
        response = self.llm.invoke(prompt)
        return response.content
    
    def rewrite_cv_with_requirements(self, requirements, cv_text=None):
        """
        Rewrite the CV to better match the job requirements using the vector database
        to find actual matching skills in the CV.
        """
        # Get the full CV text as a reference
        if cv_text is None:
            cv_text = self.read_raw_file()
        
        # Extract key terms from job requirements
        prompt_extract = f"""
        Extract a list of specific skills, technologies, qualifications, and experience 
        requirements from the following job requirements. Return ONLY a comma-separated 
        list of these terms (no numbering, no categories, just the terms):
        
        {requirements}
        """
        
        skills_response = self.llm.invoke(prompt_extract)
        key_terms = [term.strip() for term in skills_response.content.split(',') if term.strip()]
        
        print(f"Extracted {len(key_terms)} key terms from job requirements")
        
        # Query the vector database for each term to find relevant matches in the CV
        matching_content = []
        for term in key_terms:
            chunks = self.retriever.get_relevant_documents(term)
            if chunks:
                # Add the most relevant chunk for each term
                matching_content.append({
                    "term": term,
                    "content": chunks[0].page_content,
                    "score": "high match" if len(chunks) > 0 else "partial match"
                })
        
        # Format matching content for the prompt
        matching_content_text = "\n\n".join([
            f"TERM: {match['term']}\nFOUND IN CV: {match['content']}\nMATCH LEVEL: {match['score']}"
            for match in matching_content
        ])
        
        # Get exact matches vs missing skills
        matched_terms = [match["term"] for match in matching_content]
        missing_terms = [term for term in key_terms if term not in matched_terms]
        
        # Create a prompt that emphasizes using only existing skills
        prompt = f"""
        You are a professional CV writer specializing in matching CVs to job descriptions.
        Your task is to rewrite a CV to better match job requirements, but ONLY using skills
        and experiences the candidate actually has.
        
        JOB REQUIREMENTS:
        {requirements}
        
        ORIGINAL CV:
        {cv_text}
        
        SKILLS FOUND IN CV THAT MATCH JOB REQUIREMENTS:
        {matching_content_text}
        
        SKILLS REQUIRED BUT NOT EXPLICITLY FOUND IN CV:
        {', '.join(missing_terms)}
        
        INSTRUCTIONS:
        1. Rewrite the CV to highlight the matching skills and experiences found above
        2. Reorganize content to emphasize the most relevant qualifications first
        3. Tailor the summary/objective statement to the job
        4. DO NOT add any skills or experiences that were not in the original CV
        5. For skills listed as "missing" above, only include them if they are actually in the original CV but were not detected by the matching system
        6. Maintain the same overall structure and sections as the original CV
        7. Use exact keywords from the job requirements when they genuinely match the candidate's experience
        
        Return the complete rewritten CV that only contains truthful information from the original.
        """
        
        response = self.llm.invoke(prompt)
        
        # Save to file
        with open("tailored_cv.txt", "w", encoding="utf-8") as f:
            f.write(response.content)
        
        
        # Generate a report of what was matched and what wasn't
        match_report = f"""
        === CV Tailoring Report ===
        
        Skills/requirements found in your CV: {len(matched_terms)} out of {len(key_terms)}
        
        Matched: {', '.join(matched_terms[:10])}{'...' if len(matched_terms) > 10 else ''}
        
        Not explicitly found: {', '.join(missing_terms[:10])}{'...' if len(missing_terms) > 10 else ''}
        
        The CV has been tailored to emphasize your existing skills that match the job requirements.
        No new skills were added - only your actual experience was highlighted.
        """
        
        with open("match_report.txt", "w", encoding="utf-8") as f:
            f.write(match_report)
        
        return "CV successfully tailored to job requirements and saved to 'tailored_cv.txt'\n\n" + match_report
    
    def setup_tools_and_agents(self):
        # Define schemas for each tool's arguments
        class QuerySchema(BaseModel):
            query: str = Field("", description="Search query")

        class FilePathSchema(BaseModel):
            file_path_input: Optional[str] = Field(None, description="Optional path to file")
            
        class RewriteSchema(BaseModel):
            job_requirements: Union[str,dict] = Field(..., description="Job requirements to match in CV")
        
        # Job analyzer tools
        self.job_analyzer_tools = [
            StructuredTool(
                name="Job_URL_Search",
                description="Search for information in the job posting URL.",
                func=self.url_retriever.get_relevant_documents,
                args_schema=QuerySchema
            ),
            StructuredTool(
                name="Extract_Job_Requirements",
                description="Extract the key requirements from the job posting URL.",
                func=self.extract_job_requirements,
                args_schema=QuerySchema
            )
        ]
        
        # CV analyzer tools
        self.cv_analyzer_tools = [
            StructuredTool(
                name="CV_Search",
                description="Search for information in the CV document.",
                func=self.retriever.get_relevant_documents,
                args_schema=QuerySchema
            ),
            StructuredTool(
                name="Read_Full_CV",
                description="Read the entire CV document without chunking.",
                func=self.read_raw_file,
                args_schema=FilePathSchema
            )
        ]
        
        # CV rewriter tools
        self.cv_rewriter_tools = [
            StructuredTool(
                name="CV_Search",
                description="Search for specific information in the CV document.",
                func=self.retriever.get_relevant_documents,
                args_schema=QuerySchema
            ),
            StructuredTool(
                name="Read_Full_CV",
                description="Read the entire CV document without chunking.",
                func=self.read_raw_file,
                args_schema=FilePathSchema
            ),
            StructuredTool(
                name="Rewrite_CV",
                description="Rewrite the CV to better match the job requirements using vector search to find actual matching skills.",
                func=lambda job_requirements: self.rewrite_cv_with_requirements(requirements=job_requirements),
                args_schema=RewriteSchema
            )
        ]
        
        # Initialize the agents
        self.job_analyzer_agent = initialize_agent(
            tools=self.job_analyzer_tools,
            llm=self.llm,
            agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )
        
        self.cv_analyzer_agent = initialize_agent(
            tools=self.cv_analyzer_tools,
            llm=self.llm,
            agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )
        
    def match_cv_to_job(self):
        """Main method that orchestrates the CV-job matching process."""
        print("CV-Job Matching Process Started")
        print(f"CV Path: {self.file_path}")
        print(f"Job URL: {self.job_url}")
        print("\n1. Analyzing job requirements...")
        
        # Step 1: Analyze job requirements using the job analyzer agent
        job_requirements_result = self.job_analyzer_agent.invoke(
            "Analyze the job posting URL and extract a comprehensive list of skills, experience, education, and other requirements."
        )
        
        self.job_analyzer_agent_output = job_requirements_result.get('output', job_requirements_result)
        print(f"\nJob Requirements Analysis Complete:\n{self.job_analyzer_agent_output}\n")
        
        # Step 2: Analyze the CV using the CV analyzer agent
        print("\n2. Analyzing CV content...")
        cv_analysis_result = self.cv_analyzer_agent.invoke(
            "Analyze the CV document and identify the key skills, experience, and qualifications of the candidate."
        )
        
        cv_analyzer_agent_output = cv_analysis_result.get('output', cv_analysis_result)
        print(f"\nCV Analysis Complete:\n{cv_analyzer_agent_output}\n")
        
        # Step 3: Rewrite the CV using the CV rewriter agent and the job requirements
        print("\n3. Rewriting CV to match job requirements...")
        
        # Initialize the CV rewriter agent here so it can access the job requirements
        cv_rewriter_agent = initialize_agent(
            tools=self.cv_rewriter_tools,
            llm=self.llm,
            agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )
        
        rewrite_result = cv_rewriter_agent.invoke(
            f"Rewrite the CV to better match this job description. Use the Rewrite_CV tool with these job requirements: {self.job_analyzer_agent_output}"
        )
        
        rewrite_output = rewrite_result.get('output', rewrite_result)
        print(f"\nCV Rewriting Complete:\n{rewrite_output[:500]}...\n")
        print("Matching report saved to 'match_report.txt'")
        return "CV-Job matching process completed successfully."

# Usage example
if __name__ == "__main__":
    cv_path = "your_cv.pdf"
    job_posting_url = "https:/job_link"
    
    matcher = CVJobMatcher(cv_path, job_posting_url, api_key=api_key)
    # Start the matching process
    matcher.match_cv_to_job()
