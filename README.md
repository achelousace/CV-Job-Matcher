# CV-Job-Matcher Agent
An intelligent Agentic tool that matches your CV to job descriptions using LangChain and OpenAI's GPT-4.

![Screenshot 2025-04-12 193042](https://github.com/user-attachments/assets/c3496cd2-be88-4ad8-9b9d-47423b3ff919)


## Overview

CV-Job Matcher is a Python-based tool that helps tailor your CV/resume to specific job postings. It uses vector embeddings to analyze both your CV and job descriptions, then provides tailored recommendations to improve your job application.

## Features

- **Automatic Job Requirement Extraction**: Parses job descriptions from URLs/Local to identify key skills, qualifications, and requirements
- **CV Analysis**: Processes your CV to identify your existing skills and experiences
- **Smart CV Tailoring**: Rewrites your CV to highlight relevant experiences that match the job requirements
- **Truthful Matching**: Only highlights skills and experiences that are actually in your CV (no fabrication)
- **Match Reports**: Generates detailed reports showing which job requirements match your CV and which don't

## Requirements

- Python 3.7+
- OpenAI API key
- Dependencies as listed in requirements.txt

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/cv-job-matcher.git
   cd cv-job-matcher
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key by creating a `.env` file:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

### Basic Usage

```python
from cv_job_matcher import CVJobMatcher

# Initialize with your CV file path and a job posting URL
matcher = CVJobMatcher(
    file_path="path/to/your/cv.pdf",
    job_url="https://www.example.com/job-posting"
)

# Run the matching process
matcher.match_cv_to_job()
```

### Output Files

The tool generates two main output files:

1. `tailored_cv.txt` - Your CV rewritten to highlight relevant skills for the job
2. `match_report.txt` - A report showing which job requirements were found in your CV

## How It Works

1. **Job Analysis**: The tool extracts requirements from the job posting using vector search
2. **CV Processing**: Your CV is processed and converted to a searchable vector database
3. **Matching**: Each job requirement is matched against your CV to find relevant experience
4. **Tailoring**: A tailored version of your CV is created that emphasizes matching qualifications
5. **Reporting**: A summary report identifies matches and potential gaps in your qualifications

## Advanced Usage

### Custom API Key

You can provide your API key directly:

```python
matcher = CVJobMatcher(
    file_path="path/to/your/cv.pdf",
    job_url="https://www.example.com/job-posting",
    api_key="your_openai_api_key_here"
)
```

### Manual Job Requirements

You can extract job requirements separately:

```python
# Get job requirements
requirements = matcher.extract_job_requirements()

# Then rewrite your CV based on these requirements
matcher.rewrite_cv_with_requirements(requirements)
```

## Architecture

The tool is built using several key components:

- **Document Loaders**: Reads PDFs and web content
- **Vector Databases**: Converts text into searchable embeddings
- **LLM Agents**: Uses GPT-4 to analyze content and make intelligent decisions
- **Structured Tools**: Organizes the workflow into specific analysis tasks

## Limitations

- Requires an OpenAI API key with GPT-4 access
- Works best with PDFs that have extractable text
- Job URL parsing may vary depending on website structure
- Processing time depends on CV length and complexity

## License

[MIT License](LICENSE)

## Acknowledgements

This project uses the following technologies:
- LangChain
- OpenAI API
- FAISS vector database
- PyPDF and Playwright for document loading
