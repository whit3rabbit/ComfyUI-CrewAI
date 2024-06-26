import os
from typing import Any, List, Mapping, Optional
from crewai import Crew, Agent, Task, Process
from crewai_tools import (
    ScrapeWebsiteTool, 
    SerperDevTool, 
    FileReadTool,
    PDFSearchTool,
    MDXSearchTool,
    CSVSearchTool,
    DirectoryReadTool,
    BrowserbaseLoadTool,
    CodeDocsSearchTool,
    DirectorySearchTool,
    DOCXSearchTool,
    EXASearchTool,
    GithubSearchTool,
    JSONSearchTool,
    PGSearchTool,
    RagTool,
    ScrapeElementFromWebsiteTool,
    SeleniumScrapingTool,
    WebsiteSearchTool,
    XMLSearchTool,
    YoutubeChannelSearchTool,
    YoutubeVideoSearchTool,
    TXTSearchTool,
    LlamaIndexTool
)
from langchain_openai import ChatOpenAI
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
class CrewNode:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "AGENTLIST": ("AGENTLIST",),
                "TASKLIST": ("TASKLIST",),
            },
            "optional": {
                "verbose": ("BOOLEAN", {"default": False}),
                "max_rpm": ("INT", {"default": 25, "min": 0, "max": 100, "step": 1}),
                "language": ("STRING", {"default": "English"}),
                "memory": ("BOOLEAN", {"default": False}),
                "process": (["sequential", "hierarchical"], {"default": "sequential"}),
                "TOOLLIST": ("TOOLLIST",),
            }
        }
 
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("RESULT",)
 
    FUNCTION = "execute"
 
    CATEGORY = "Crewai"
  
    def execute(self, AGENTLIST, TASKLIST, verbose, TOOLLIST=None, **kwargs):
        crew = Crew(
            agents=AGENTLIST,
            tasks=TASKLIST, 
            verbose=verbose, 
            tools=TOOLLIST,
            **kwargs
        )

        result = crew.kickoff()

        formatted_result = f"""Crew Execution Result:

{result}

Agents used: {len(AGENTLIST)}
Tasks completed: {len(TASKLIST)}
Tools available: {len(TOOLLIST) if TOOLLIST else 0}
Execution process: {kwargs.get('process', 'sequential')}
Language: {kwargs.get('language', 'English')}
Memory enabled: {kwargs.get('memory', False)}
Max RPM: {kwargs.get('max_rpm', 25)}
"""

        return (formatted_result,)
 
class ClaudeLLM(LLM):
    client: Anthropic
    model: str
    max_tokens: int
    temperature: float

    def __init__(self, api_key: str, model: str, max_tokens: int = 1000, temperature: float = 0.7):
        super().__init__()
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _call(
        self, 
        prompt: str, 
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        response = self.client.completions.create(
            model=self.model,
            prompt=f"{HUMAN_PROMPT} {prompt} {AI_PROMPT}",
            max_tokens_to_sample=self.max_tokens,
            temperature=self.temperature,
            stop_sequences=stop or []
        )
        return response.completion

    @property
    def _llm_type(self) -> str:
        return "claude"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model": self.model, "max_tokens": self.max_tokens, "temperature": self.temperature}
class AgentNode:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "role": ("STRING", {"multiline": True, "default": ""}),
                "goal": ("STRING", {"multiline": True, "default": ""}),
                "backstory": ("STRING", {"multiline": True, "default": ""}),                                                                       
            },
            "optional": {
                "LLM": ("LLM",),
                "function_LLM": ("LLM",),
                "TOOLLIST": ("TOOLLIST",),
                "max_iter": ("INT", {"default": 25, "min": 0, "max": 100, "step": 1}),
                "max_rpm": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "max_execution_time": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "verbose": ("BOOLEAN", {"default": False}),
                "allow_delegation": ("BOOLEAN", {"default": True}),
                "callback": ("FUNCTION", {"default": None}),
                "cache": ("BOOLEAN", {"default": True}),
            }
        }
 
    RETURN_TYPES = ("AGENT",)
    RETURN_NAMES = ()
 
    FUNCTION = "set_agent"
 
    OUTPUT_NODE = True
 
    CATEGORY = "Crewai"
 
    def set_agent(self, role, goal, backstory, LLM=None, function_LLM=None, TOOLLIST=None, **kwargs):
        agent = Agent(
            role=role,
            goal=goal,
            backstory=backstory,
            llm=LLM,
            function_calling_llm=function_LLM,
            tools=TOOLLIST if TOOLLIST else [],
            **{k: v for k, v in kwargs.items() if v is not None and v != 0}
        )

        return (agent,)
    
class TaskNode:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "description": ("STRING", {"multiline": True, "default": ""}),
                "AGENT": ("AGENT",),
                "expected_output": ("STRING", {"multiline": True, "default": ""}),                                                                                      
            },
            "optional": {
                "TOOLLIST": ("TOOLLIST",),
                "async_execution": ("BOOLEAN", {"default": False}),
                "context": ("TASKLIST",),
                "output_file": ("STRING", {"default": "output.md"}),
                "callback": ("FUNCTION", {"default": None}),
                "human_feedback": ("BOOLEAN", {"default": False}),
            }
        }
 
    RETURN_TYPES = ("TASK",)
    RETURN_NAMES = ()
 
    FUNCTION = "set_task"
 
    OUTPUT_NODE = True
 
    CATEGORY = "Crewai"
 
    def set_task(self, description, AGENT, expected_output, TOOLLIST=None, context=None, **kwargs):
        task = Task(
            description=description,
            agent=AGENT,
            expected_output=expected_output,
            tools=TOOLLIST if TOOLLIST else [],
            context=context if context else [],
            **kwargs
        )

        return (task,)
class OpenAILLMNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (["gpt-4", "gpt-3.5-turbo"], {"default": "gpt-3.5-turbo"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": os.getenv("OPENAI_API_KEY", "")}),
                "base_url": ("STRING", {"default": "https://api.openai.com/v1"}),
                "max_tokens": ("INT", {"default": 1000, "min": 1, "max": 100000}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1}),
            }
        }
 
    RETURN_TYPES = ("LLM",)
    RETURN_NAMES = ()
    FUNCTION = "set_openai_llm"
    OUTPUT_NODE = True
    CATEGORY = "Crewai/LLMs"
 
    def set_openai_llm(self, model, api_key="", base_url="https://api.openai.com/v1", max_tokens=1000, temperature=0.7):
        if not api_key:
            raise ValueError("OpenAI API key is required. Please provide it or set the OPENAI_API_KEY environment variable.")
        return (ChatOpenAI(base_url=base_url, api_key=api_key, model=model, max_tokens=max_tokens, temperature=temperature),)

class ClaudeLLMNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"], {"default": "claude-3-opus-20240229"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": os.getenv("ANTHROPIC_API_KEY", "")}),
                "max_tokens": ("INT", {"default": 1000, "min": 1, "max": 100000}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1}),
            }
        }
 
    RETURN_TYPES = ("LLM",)
    RETURN_NAMES = ()
    FUNCTION = "set_claude_llm"
    OUTPUT_NODE = True
    CATEGORY = "Crewai/LLMs"
 
    def set_claude_llm(self, model, api_key="", max_tokens=1000, temperature=0.7):
        if not api_key:
            raise ValueError("Anthropic API key is required. Please provide it or set the ANTHROPIC_API_KEY environment variable.")
        return (ClaudeLLM(api_key=api_key, model=model, max_tokens=max_tokens, temperature=temperature),)

class GroqLLMNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (["mixtral-8x7b-32768", "llama2-70b-4096"], {"default": "mixtral-8x7b-32768"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": os.getenv("GROQ_API_KEY", "")}),
                "max_tokens": ("INT", {"default": 1000, "min": 1, "max": 100000}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1}),
            }
        }
 
    RETURN_TYPES = ("LLM",)
    RETURN_NAMES = ()
    FUNCTION = "set_groq_llm"
    OUTPUT_NODE = True
    CATEGORY = "Crewai/LLMs"
 
    def set_groq_llm(self, model, api_key="", max_tokens=1000, temperature=0.7):
        if not api_key:
            raise ValueError("Groq API key is required. Please provide it or set the GROQ_API_KEY environment variable.")
        return (ChatOpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key, model=model, max_tokens=max_tokens, temperature=temperature),)

class AgentListNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "agent_01": ("AGENT",),
            },
            "optional": {
                "agent_02": ("AGENT",),
                "agent_03": ("AGENT",),
                "agent_04": ("AGENT",),
            }
        }
    RETURN_TYPES = ("AGENTLIST",)
    RETURN_NAMES = ()
 
    FUNCTION = "set_agents"
 
    OUTPUT_NODE = True
 
    CATEGORY = "Crewai"
    
    def set_agents(self, agent_01, agent_02=None, agent_03=None, agent_04=None):
        return ([agent for agent in [agent_01, agent_02, agent_03, agent_04] if agent is not None],)

class TaskListNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "task_01": ("TASK",),
            },
            "optional": {
                "task_02": ("TASK",),
                "task_03": ("TASK",),
                "task_04": ("TASK",),
            }
        }
    RETURN_TYPES = ("TASKLIST",)
    RETURN_NAMES = ()
 
    FUNCTION = "set_tasks"
 
    OUTPUT_NODE = True
 
    CATEGORY = "Crewai"
    
    def set_tasks(self, task_01, task_02=None, task_03=None, task_04=None):
        return ([task for task in [task_01, task_02, task_03, task_04] if task is not None],)

class ToolsListNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tool_01": ("TOOL",),
            },
            "optional": {
                "tool_02": ("TOOL",),
                "tool_03": ("TOOL",),
                "tool_04": ("TOOL",),
            }
        }
    RETURN_TYPES = ("TOOLLIST",)
    RETURN_NAMES = ()
    FUNCTION = "set_tools"
    OUTPUT_NODE = True
    CATEGORY = "Crewai/tools"
    
    def set_tools(self, tool_01, tool_02=None, tool_03=None, tool_04=None):
        return ([tool for tool in [tool_01, tool_02, tool_03, tool_04] if tool is not None],)

class SWTNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {}
        }
    RETURN_TYPES = ("TOOL",)
    RETURN_NAMES = ()
    FUNCTION = "set_swt"
    OUTPUT_NODE = True
    CATEGORY = "Crewai/tools"
 
    def set_swt(self):
        return (ScrapeWebsiteTool(),)

class SDTNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "api_key": ("STRING", {"default": os.getenv("SERPER_API_KEY", "")}),
            }
        }
    RETURN_TYPES = ("TOOL",)
    RETURN_NAMES = ()
    FUNCTION = "set_sdt"
    OUTPUT_NODE = True
    CATEGORY = "Crewai/tools"
 
    def set_sdt(self, api_key=""):
        if not api_key:
            raise ValueError("Serper API key is required. Please provide it or set the SERPER_API_KEY environment variable.")
        return (SerperDevTool(api_key=api_key),)

class MDXSTNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "file_path": ("STRING", {"default": "/file_path"}),                
            },
        }
    RETURN_TYPES = ("TOOL",)
    RETURN_NAMES = ()
    FUNCTION = "set_mdxst"
    OUTPUT_NODE = True
    CATEGORY = "Crewai/tools"

    def set_mdxst(self, file_path):
        return (MDXSearchTool(mdx=file_path),)
    
class FRTNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "file_path": ("STRING", {"default": "/file_path"}),                
            },
        }
    RETURN_TYPES = ("TOOL",)
    RETURN_NAMES = ()
    FUNCTION = "set_frt"
    OUTPUT_NODE = True
    CATEGORY = "Crewai/tools"

    def set_frt(self, file_path):
        return (FileReadTool(file_path=file_path),)

# New tool nodes

class PDFSearchToolNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "pdf": ("STRING", {"default": ""}),
            }
        }
    RETURN_TYPES = ("TOOL",)
    RETURN_NAMES = ()
    FUNCTION = "set_pdf_search_tool"
    OUTPUT_NODE = True
    CATEGORY = "Crewai/tools"

    def set_pdf_search_tool(self, pdf=""):
        return (PDFSearchTool(pdf=pdf),)

class CSVSearchToolNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "csv": ("STRING", {"default": ""}),
            }
        }
    RETURN_TYPES = ("TOOL",)
    RETURN_NAMES = ()
    FUNCTION = "set_csv_search_tool"
    OUTPUT_NODE = True
    CATEGORY = "Crewai/tools"

    def set_csv_search_tool(self, csv=""):
        return (CSVSearchTool(csv=csv),)

class DirectoryReadToolNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": ("STRING", {"default": "/directory/path"}),
            },
        }
    RETURN_TYPES = ("TOOL",)
    RETURN_NAMES = ()
    FUNCTION = "set_directory_read_tool"
    OUTPUT_NODE = True
    CATEGORY = "Crewai/tools"

    def set_directory_read_tool(self, directory):
        return (DirectoryReadTool(directory=directory),)

class BrowserbaseLoadToolNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "api_key": ("STRING", {"default": os.getenv("BROWSERBASE_API_KEY", "")}),
                "text_content": ("BOOLEAN", {"default": False}),
            }
        }
    RETURN_TYPES = ("TOOL",)
    RETURN_NAMES = ()
    FUNCTION = "set_browserbase_load_tool"
    OUTPUT_NODE = True
    CATEGORY = "Crewai/tools"

    def set_browserbase_load_tool(self, api_key="", text_content=False):
        return (BrowserbaseLoadTool(api_key=api_key, text_content=text_content),)

class CodeDocsSearchToolNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "docs_url": ("STRING", {"default": ""}),
            }
        }
    RETURN_TYPES = ("TOOL",)
    RETURN_NAMES = ()
    FUNCTION = "set_code_docs_search_tool"
    OUTPUT_NODE = True
    CATEGORY = "Crewai/tools"

    def set_code_docs_search_tool(self, docs_url=""):
        return (CodeDocsSearchTool(docs_url=docs_url),)

class DirectorySearchToolNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "directory": ("STRING", {"default": ""}),
            }
        }
    RETURN_TYPES = ("TOOL",)
    RETURN_NAMES = ()
    FUNCTION = "set_directory_search_tool"
    OUTPUT_NODE = True
    CATEGORY = "Crewai/tools"

    def set_directory_search_tool(self, directory=""):
        return (DirectorySearchTool(directory=directory),)

class DOCXSearchToolNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "docx": ("STRING", {"default": ""}),
            }
        }
    RETURN_TYPES = ("TOOL",)
    RETURN_NAMES = ()
    FUNCTION = "set_docx_search_tool"
    OUTPUT_NODE = True
    CATEGORY = "Crewai/tools"

    def set_docx_search_tool(self, docx=""):
        return (DOCXSearchTool(docx=docx),)

class EXASearchToolNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {}
        }
    RETURN_TYPES = ("TOOL",)
    RETURN_NAMES = ()
    FUNCTION = "set_exa_search_tool"
    OUTPUT_NODE = True
    CATEGORY = "Crewai/tools"

    def set_exa_search_tool(self):
        return (EXASearchTool(),)

class GithubSearchToolNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "gh_token": ("STRING", {"default": ""}),
            },
            "optional": {
                "github_repo": ("STRING", {"default": ""}),
                "content_types": ("STRING", {"default": "code,repo,pr,issue"}),
            }
        }
    RETURN_TYPES = ("TOOL",)
    RETURN_NAMES = ()
    FUNCTION = "set_github_search_tool"
    OUTPUT_NODE = True
    CATEGORY = "Crewai/tools"

    def set_github_search_tool(self, gh_token, github_repo="", content_types="code,repo,pr,issue"):
        content_types_list = content_types.split(',')
        return (GithubSearchTool(gh_token=gh_token, github_repo=github_repo, content_types=content_types_list),)

class JSONSearchToolNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "json_path": ("STRING", {"default": ""}),
            }
        }
    RETURN_TYPES = ("TOOL",)
    RETURN_NAMES = ()
    FUNCTION = "set_json_search_tool"
    OUTPUT_NODE = True
    CATEGORY = "Crewai/tools"

    def set_json_search_tool(self, json_path=""):
        return (JSONSearchTool(json_path=json_path),)

class PGSearchToolNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "db_uri": ("STRING", {"default": ""}),
                "table_name": ("STRING", {"default": ""}),
            },
        }
    RETURN_TYPES = ("TOOL",)
    RETURN_NAMES = ()
    FUNCTION = "set_pg_search_tool"
    OUTPUT_NODE = True
    CATEGORY = "Crewai/tools"

    def set_pg_search_tool(self, db_uri, table_name):
        return (PGSearchTool(db_uri=db_uri, table_name=table_name),)

class RagToolNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "summarize": ("BOOLEAN", {"default": False}),
            }
        }
    RETURN_TYPES = ("TOOL",)
    RETURN_NAMES = ()
    FUNCTION = "set_rag_tool"
    OUTPUT_NODE = True
    CATEGORY = "Crewai/tools"

    def set_rag_tool(self, summarize=False):
        return (RagTool(summarize=summarize),)

class ScrapeElementFromWebsiteToolNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "website_url": ("STRING", {"default": ""}),
                "css_element": ("STRING", {"default": ""}),
            }
        }
    RETURN_TYPES = ("TOOL",)
    RETURN_NAMES = ()
    FUNCTION = "set_scrape_element_from_website_tool"
    OUTPUT_NODE = True
    CATEGORY = "Crewai/tools"

    def set_scrape_element_from_website_tool(self, website_url="", css_element=""):
        return (ScrapeElementFromWebsiteTool(website_url=website_url, css_element=css_element),)

class SeleniumScrapingToolNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "website_url": ("STRING", {"default": ""}),
                "css_element": ("STRING", {"default": ""}),
            }
        }
    RETURN_TYPES = ("TOOL",)
    RETURN_NAMES = ()
    FUNCTION = "set_selenium_scraping_tool"
    OUTPUT_NODE = True
    CATEGORY = "Crewai/tools"

    def set_selenium_scraping_tool(self, website_url="", css_element=""):
        return (SeleniumScrapingTool(website_url=website_url, css_element=css_element),)

class WebsiteSearchToolNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "website": ("STRING", {"default": ""}),
            }
        }
    RETURN_TYPES = ("TOOL",)
    RETURN_NAMES = ()
    FUNCTION = "set_website_search_tool"
    OUTPUT_NODE = True
    CATEGORY = "Crewai/tools"

    def set_website_search_tool(self, website=""):
        return (WebsiteSearchTool(website=website),)

class XMLSearchToolNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "xml": ("STRING", {"default": ""}),
            }
        }
    RETURN_TYPES = ("TOOL",)
    RETURN_NAMES = ()
    FUNCTION = "set_xml_search_tool"
    OUTPUT_NODE = True
    CATEGORY = "Crewai/tools"

    def set_xml_search_tool(self, xml=""):
        return (XMLSearchTool(xml=xml),)

class YoutubeChannelSearchToolNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "youtube_channel_handle": ("STRING", {"default": ""}),
            }
        }
    RETURN_TYPES = ("TOOL",)
    RETURN_NAMES = ()
    FUNCTION = "set_youtube_channel_search_tool"
    OUTPUT_NODE = True
    CATEGORY = "Crewai/tools"

    def set_youtube_channel_search_tool(self, youtube_channel_handle=""):
        return (YoutubeChannelSearchTool(youtube_channel_handle=youtube_channel_handle),)

class YoutubeVideoSearchToolNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "youtube_video_url": ("STRING", {"default": ""}),
            }
        }
    RETURN_TYPES = ("TOOL",)
    RETURN_NAMES = ()
    FUNCTION = "set_youtube_video_search_tool"
    OUTPUT_NODE = True
    CATEGORY = "Crewai/tools"

    def set_youtube_video_search_tool(self, youtube_video_url=""):
        return (YoutubeVideoSearchTool(youtube_video_url=youtube_video_url),)

class TXTSearchToolNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "txt": ("STRING", {"default": ""}),
            }
        }
    RETURN_TYPES = ("TOOL",)
    RETURN_NAMES = ()
    FUNCTION = "set_txt_search_tool"
    OUTPUT_NODE = True
    CATEGORY = "Crewai/tools"

    def set_txt_search_tool(self, txt=""):
        return (TXTSearchTool(txt=txt),)

class LlamaIndexToolNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tool_type": (["from_tool", "from_query_engine"], {"default": "from_tool"}),
            },
            "optional": {
                "name": ("STRING", {"default": ""}),
                "description": ("STRING", {"default": ""}),
            }
        }
    RETURN_TYPES = ("TOOL",)
    RETURN_NAMES = ()
    FUNCTION = "set_llama_index_tool"
    OUTPUT_NODE = True
    CATEGORY = "Crewai/tools"

    def set_llama_index_tool(self, tool_type, name="", description=""):
        # Note: This is a simplified implementation. You may need to adjust it based on how you want to use LlamaIndexTool in your ComfyUI setup.
        if tool_type == "from_tool":
            return (LlamaIndexTool.from_tool(name=name, description=description),)
        elif tool_type == "from_query_engine":
            return (LlamaIndexTool.from_query_engine(name=name, description=description),)

# Update NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS

NODE_CLASS_MAPPINGS = {
    "Crew": CrewNode,
    "Agent": AgentNode,
    "Task": TaskNode,
    "OpenAI LLM": OpenAILLMNode,
    "Claude LLM": ClaudeLLMNode,
    "Groq LLM": GroqLLMNode,
    "AgentList": AgentListNode,
    "TaskList": TaskListNode,
    "ToolsList": ToolsListNode,
    "WebScraper": SWTNode,
    "WebSearcher": SDTNode,
    "FileReader": FRTNode,
    "MDXSearcher": MDXSTNode,
    "PDFSearcher": PDFSearchToolNode,
    "CSVSearcher": CSVSearchToolNode,
    "DirectoryReader": DirectoryReadToolNode,
    "BrowserbaseLoader": BrowserbaseLoadToolNode,
    "CodeDocsSearcher": CodeDocsSearchToolNode,
    "DirectorySearcher": DirectorySearchToolNode,
    "DOCXSearcher": DOCXSearchToolNode,
    "EXASearcher": EXASearchToolNode,
    "GithubSearcher": GithubSearchToolNode,
    "JSONSearcher": JSONSearchToolNode,
    "PGSearcher": PGSearchToolNode,
    "RAGTool": RagToolNode,
    "ElementScraper": ScrapeElementFromWebsiteToolNode,
    "SeleniumScraper": SeleniumScrapingToolNode,
    "WebsiteSearcher": WebsiteSearchToolNode,
    "XMLSearcher": XMLSearchToolNode,
    "YoutubeChannelSearcher": YoutubeChannelSearchToolNode,
    "YoutubeVideoSearcher": YoutubeVideoSearchToolNode,
    "TXTSearcher": TXTSearchToolNode,
    "LlamaIndexTool": LlamaIndexToolNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CrewNode": "Crew",
    "AgentNode": "Agent",
    "TaskNode": "Task",
    "OpenAILLMNode": "OpenAI LLM",
    "ClaudeLLMNode": "Claude LLM",
    "GroqLLMNode": "Groq LLM",
    "AgentListNode": "Agent List",
    "TaskListNode": "Task List",
    "ToolsListNode": "Tools List",
    "SWTNode": "Web Scraping Tool", 
    "SDTNode": "Web Search Tool",
    "FRTNode": "File Reading Tool",
    "MDXSTNode": "MDX Document Search Tool",
    "PDFSearchToolNode": "PDF Search Tool",
    "CSVSearchToolNode": "CSV Search Tool",
    "DirectoryReadToolNode": "Directory Read Tool",
    "BrowserbaseLoadToolNode": "Browserbase Load Tool",
    "CodeDocsSearchToolNode": "Code Docs Search Tool",
    "DirectorySearchToolNode": "Directory Search Tool",
    "DOCXSearchToolNode": "DOCX Search Tool",
    "EXASearchToolNode": "EXA Search Tool",
    "GithubSearchToolNode": "Github Search Tool",
    "JSONSearchToolNode": "JSON Search Tool",
    "PGSearchToolNode": "PostgreSQL Search Tool",
    "RagToolNode": "RAG Tool",
    "ScrapeElementFromWebsiteToolNode": "Element Scraper Tool",
    "SeleniumScrapingToolNode": "Selenium Scraping Tool",
    "WebsiteSearchToolNode": "Website Search Tool",
    "XMLSearchToolNode": "XML Search Tool",
    "YoutubeChannelSearchToolNode": "Youtube Channel Search Tool",
    "YoutubeVideoSearchToolNode": "Youtube Video Search Tool",
    "TXTSearchToolNode": "TXT Search Tool",
    "LlamaIndexToolNode": "LlamaIndex Tool",
}