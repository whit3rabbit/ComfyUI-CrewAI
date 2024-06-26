# ComfyUI-CrewAI

### Using ComfyUI to develop crews without any code.

Inspired by [ComfyUI](https://github.com/comfyanonymous/ComfyUI) and [CrewAI](https://www.crewai.com/), this project aims to develop nodes within ComfyUI that enable the execution of multi-agent tasks without requiring any coding.

![Screenshot of sample](./sample_screenshot/Example_1.png)

### Nodes implemented

1. Crew, refer to [Crews](https://docs.crewai.com/core-concepts/Crews/)
2. LLM (OpenAI, Claude, Groq)
3. Agent, refer to [Agents](https://docs.crewai.com/core-concepts/Agents/)
4. Agent List
5. Task, refer to [Tasks](https://docs.crewai.com/core-concepts/Tasks/)
6. Task List
7. Tool List
8. Various Search and Scraping Tools:

   - Web Scraper (ScrapeWebsiteTool)
   - Web Searcher (SerperDevTool)
   - File Reader (FileReadTool)
   - MDX Searcher (MDXSearchTool)
   - PDF Searcher (PDFSearchTool)
   - CSV Searcher (CSVSearchTool)
   - Directory Reader (DirectoryReadTool)
   - Browserbase Loader (BrowserbaseLoadTool)
   - Code Docs Searcher (CodeDocsSearchTool)
   - Directory Searcher (DirectorySearchTool)
   - DOCX Searcher (DOCXSearchTool)
   - EXA Searcher (EXASearchTool)
   - Github Searcher (GithubSearchTool)
   - JSON Searcher (JSONSearchTool)
   - PostgreSQL Searcher (PGSearchTool)
   - RAG Tool (RagTool)
   - Element Scraper (ScrapeElementFromWebsiteTool)
   - Selenium Scraper (SeleniumScrapingTool)
   - Website Searcher (WebsiteSearchTool)
   - XML Searcher (XMLSearchTool)
   - Youtube Channel Searcher (YoutubeChannelSearchTool)
   - Youtube Video Searcher (YoutubeVideoSearchTool)
   - TXT Searcher (TXTSearchTool)
   - LlamaIndex Tool (LlamaIndexTool)

### Installation

1. Install [ComfyUI](https://github.com/comfyanonymous/ComfyUI) by following the instructions on their GitHub page
2. Clone this repository to custom_nodes folder
3. Install requirements by running

`pip install -r requirements.txt`

or if you use portable (run this in ComfyUI_windows_portable -folder):

`python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-CrewAI\requirements.txt`

For showing text you will need this addon as well:

https://github.com/pythongosssss/ComfyUI-Custom-Scripts

### How to Connect Nodes

1. **LLM Nodes**: Connect these to Agent nodes to set the language model for each agent.
2. **Agent Nodes**: Create multiple agents and connect them to an Agent List node.
3. **Task Nodes**: Create tasks and connect them to a Task List node.
4. **Tool Nodes**: Select the tools you need and connect them to a Tool List node.
5. **Crew Node**: Connect the Agent List, Task List, and Tool List nodes to the Crew node to set up your multi-agent system.

Example workflow:

1. Create LLM nodes (e.g., OpenAI LLM) and set API keys.
2. Create Agent nodes and connect LLM nodes to them.
3. Connect Agent nodes to an Agent List node.
4. Create Task nodes with descriptions of what needs to be done.
5. Connect Task nodes to a Task List node.
6. Select necessary Tool nodes (e.g., Web Scraper, PDF Searcher) and connect them to a Tool List node.
7. Connect Agent List, Task List, and Tool List to the Crew node.
8. Run the workflow to execute your multi-agent task.

### API Keys and Environment Variables

Some tools require API keys. Set these in your environment variables or provide them directly in the tool nodes:

- OPENAI_API_KEY: For OpenAI LLM
- ANTHROPIC_API_KEY: For Claude LLM
- SERPER_API_KEY: For Web Searcher (SerperDevTool)
- BROWSERBASE_API_KEY: For Browserbase Loader
- EXA_API_KEY: For EXA Searcher

### Future work

1. Add more examples demonstrating various multi-agent scenarios
2. Improve documentation for each tool and its specific use cases
3. Develop custom UI elements for better parameter input and visualization
4. Implement error handling and validation for API keys and inputs

### License

[MIT](./LICENSE)
