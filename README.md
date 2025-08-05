# AI-Powered Financial Research Assistant

This project is an AI-powered research assistant designed to help users with financial research. It utilizes a Retrieval-Augmented Generation (RAG) pipeline, web crawling, and natural language processing to provide accurate and context-aware answers to user queries. The application features a user-friendly chat interface for interacting with the AI assistant.
 
## Features 

This project offers a range of features to assist users with financial research:

*   **Retrieval-Augmented Generation (RAG) Pipeline:** The core of the application is a RAG pipeline that combines the power of large language models with information retrieval. This allows the AI assistant to generate accurate and context-aware answers based on a knowledge base of financial documents and real-time web data.
*   **Semantic Caching:** To improve performance and reduce redundant computations, the application implements a semantic cache. This cache stores the embeddings of previous questions and their corresponding answers, allowing the system to quickly retrieve answers to similar queries.
*   **Web Crawling:** The application includes a web crawling module that can gather up-to-date financial information from various online sources. This ensures that the AI assistant has access to the latest market trends, news, and analysis.
*   **Financial Data Analysis:** The AI assistant is designed to help users analyze and understand complex financial data. It can answer questions about stock markets, cryptocurrencies, bonds, personal finance, investment strategies, and economic policies.
*   **User-Friendly Interface:** The project includes a web-based chat interface that allows users to easily interact with the AI assistant.

## Project Structure

The repository is organized into two main projects: `Experiement_code` and `PROJECT`.

*   **`Experiement_code/`**: This directory contains experimental code, data files (primarily PDFs), and a Dockerfile. It appears to be used for data processing or machine learning experiments.
    *   `expt.py`: Python script for running experiments.
    *   `exptdata/`: Directory containing data files used in experiments.
    *   `Dockerfile`: Dockerfile for building a container for the experiments.
    *   `requirements.txt`: Python dependencies for the experimental code.

*   **`PROJECT/`**: This directory contains the main web application, which includes a frontend and a backend.
    *   **`backend/`**: This directory contains the Python/Flask backend application.
        *   `app.py`: The main Flask application file, containing the RAG pipeline, semantic cache, and API endpoints.
        *   `crawler.py`: Python script for web crawling.
        *   `deepsearch.py`: Python script for performing deep searches.
        *   `tools.py`: Python script containing utility functions.
        *   `crawled_data/`: Directory containing data crawled from the web.
        *   `data/`: Directory containing PDF documents used in the RAG pipeline.
        *   `requirements.txt`: Python dependencies for the backend application.
        *   `Dockerfile`: Dockerfile for building a container for the backend application.
    *   **`frontend/`**: This directory contains the Next.js frontend application.
        *   `app/`: Directory containing the main application code, including pages and components.
        *   `components/`: Directory containing reusable React components.
        *   `lib/`: Directory containing utility functions and type definitions.
        *   `public/`: Directory containing static assets.
        *   `package.json`: Node.js dependencies for the frontend application.
        *   `Dockerfile`: Dockerfile for building a container for the frontend application.
    *   `docker-compose.yaml`: Docker Compose file for running the frontend and backend applications together.


## Setup and Installation

To set up and run this project, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Set up environment variables:**
    *   The backend application requires a Google API key for the Gemini model and a Hugging Face token. Create a `.env` file in the `PROJECT/backend/` directory with the following content:
        ```
        GOOGLE_API_MODEL=<your_google_api_key>
        HF_TOKEN=<your_hugging_face_token>
        ```
    *   Replace `<your_google_api_key>` and `<your_hugging_face_token>` with your actual API key and token.

3.  **Install dependencies:**
    *   **Backend:**
        ```bash
        cd PROJECT/backend
        pip install -r requirements.txt
        ```
    *   **Frontend:**
        ```bash
        cd PROJECT/frontend
        npm install
        ```

4.  **Run the application:**
    *   You can run the frontend and backend applications separately or use Docker Compose.
    *   **Separately:**
        *   **Backend:**
            ```bash
            cd PROJECT/backend
            flask run
            ```
        *   **Frontend:**
            ```bash
            cd PROJECT/frontend
            npm run dev
            ```
    *   **Using Docker Compose:**
        ```bash
        cd PROJECT
        docker-compose up --build
        ```
    *   This will build and run both the frontend and backend containers. The frontend will be accessible at `http://localhost:3000` and the backend at `http://localhost:4200`.

**Note:** The `Experiement_code/` directory has its own `requirements.txt` and `Dockerfile`. If you want to run the experimental code, navigate to that directory and follow similar steps for installing dependencies and building the Docker container.

## How to Use

Once the application is running, you can access the frontend by opening your web browser and navigating to `http://localhost:3000`.

The main interface is a chat window where you can type your questions related to financial research. The AI assistant will process your query and provide an answer based on its knowledge base and real-time web data.

### Frontend

*   **Chat Interface:** The primary way to interact with the AI assistant. Type your questions in the input field and press Enter or click the send button.
*   **Theme Toggle:** A button (usually represented by a sun or moon icon) to switch between light and dark mode for the user interface.
*   **Deep Search (Coming Soon):** The interface may include elements related to a "Deep Search" functionality, which is likely a planned feature for more in-depth information retrieval.

### Backend API

The backend exposes API endpoints that the frontend uses to communicate with the RAG pipeline and other services. If you are developing or testing, you might interact with these directly (e.g., using tools like Postman or curl).

*   **`/query` (POST):** This is likely the main endpoint for sending user queries to the RAG pipeline.
    *   **Request body:** Typically a JSON object containing the user's question and possibly chat history.
    *   **Response:** A JSON object containing the AI's answer and any relevant source documents.
*   **`/report` (POST):** This endpoint might be used for users to report issues or provide feedback on the AI's responses.
*   **`/deepsearch` (POST):** This endpoint could be used to trigger the deepsearch functionality of the project.

**Example Query (Conceptual):**

To get information on a specific stock, you might type:

```
What is the current price of AAPL and recent news?
```

If you want to enable the data-driven (web search) mode, you might append `|||TRUE|||` to your query:
```
What are the latest trends in cryptocurrency? |||TRUE|||
```
The AI will then attempt to answer using its document knowledge base and, if the flag is present, also by searching the web via its tools.

The backend `app.py` script details how queries are processed, including the use of different prompts (`PROMPT_DATA_DRIVEN` vs. `PROMPT_STRICT`) based on the presence of the `|||TRUE|||` flag.

## Contributing

Contributions to this project are welcome! If you want to contribute, please follow these guidelines:

1.  **Fork the repository.**
2.  **Create a new branch** for your feature or bug fix:
    ```bash
    git checkout -b feature/your-feature-name
    ```
    or
    ```bash
    git checkout -b bugfix/your-bug-fix
    ```
3.  **Make your changes** and ensure that the code lints and tests pass (if applicable).
4.  **Commit your changes** with a clear and descriptive commit message:
    ```bash
    git commit -m "feat: Add new feature X" 
    ```
    or
    ```bash
    git commit -m "fix: Resolve issue Y"
    ```
    (Consider using [Conventional Commits](https://www.conventionalcommits.org/) for commit messages.)
5.  **Push your changes** to your forked repository:
    ```bash
    git push origin feature/your-feature-name
    ```
6.  **Create a pull request** to the `main` branch of the original repository.
7.  **Clearly describe your changes** in the pull request description.

If you are planning to make significant changes, please open an issue first to discuss your ideas.

## License

This project is currently not licensed. 

It is recommended to add a `LICENSE` file to the repository to specify the terms under which the software can be used, modified, and distributed. Popular open-source licenses include MIT, Apache 2.0, and GPLv3. You can choose a license that best suits your project's needs.

Once a license is chosen, you can update this section to reflect it, for example:

```
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

### Future Plans (from original README)

The initial project notes mentioned several areas for future development:

*   Refining the data structure from the web crawler.
*   Implementing stop word removal more effectively.
*   Exploring data visualization (e.g., graphs).
*   Investigating external caching solutions like Redis.
*   Thorough application testing, including load testing if the application is intended for public use.
*   Potential development of a research paper based on the project.

### Acknowledgements

*   This project utilizes various open-source libraries and tools. Refer to the `requirements.txt` files in the `PROJECT/backend` and `Experiement_code` directories, and the `package.json` in `PROJECT/frontend` for a list of dependencies.
*   The RAG pipeline leverages models and embeddings from Google (Gemini) and Sentence Transformers.
