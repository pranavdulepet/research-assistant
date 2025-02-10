Steps to run:
1. create a .env file containing (need at least one, ignore other if not needed):
    OPENAI_API_KEY=your_openai_key_here
   
    GEMINI_API_KEY=your_gemini_key_here
3. python3.11 install.py
4. source venv/bin/activate
3. python3 run.py
4. open [http://127.0.0.1:5000](http://127.0.0.1:5000) in browser





-----------------------------------
Features:
- Cursor for literature review
- Upload research paper, extract text, figures, etc.
- Allow user to annotate and write notes
- Add “Chat with paper”
    - ask anything about the paper and get a response based on the paper's context and general knowledge
- Add “Ask AI to verify” (notes, annotation, etc.)
    - verify if notes are valid in the paper's context and general knowledge
    - searches the web to help verify notes
- Add “Ask AI to answer” (any questions in annotations, this can be a specific type of “question” annotation)
    - answer the question based on the paper's context and general knowledge
    - searches the web to help answer questions


Upcoming features:
- add all cited papers to the a data store and use RAG to use that context as to verify notes and answer questions
- explain math specifically and how it relates to the paper content
