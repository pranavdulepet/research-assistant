Steps to run:
1. python3.11 install.py
2. create a .env file containing:
    OPENAI_API_KEY=your_openai_key_here
    GEMINI_API_KEY=your_gemini_key_here
3. python run.py
4. open [text](http://localhost:5000) in browser





-----------------------------------
Notes:
- Research paper reader helper
    - Cursor for research papers
- Upload research paper, extract text, figures, etc.
- Extract citations and find the cited papers
- Allow user to annotate and write notes
- Add “Ask AI to summarize”
    - this can be done in the "chat with paper" section
- Add “Ask AI to verify” (notes, annotation, etc.)
    - verify if notes are valid in the paper's context and general knowledge
- Add “Ask AI to answer” (any questions in annotations, this can be a specific type of “question” annotation)
    - answer the question based on the paper's context and general knowledge


Additional features:
- explain math specifically and how it relates to the paper content