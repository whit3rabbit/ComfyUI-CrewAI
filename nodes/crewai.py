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
    DirectoryReadTool
)
from langchain_openai import ChatOpenAI
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun

os.environ["SERPER_API_KEY"] = "your key here"
os.environ["OPENAI_BASE_URL"] = "https://api.groq.com/openai/v1"
os.environ["OPENAI_API_KEY"] = "your key here"

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
                "manager_LLM": ("LLM",),
                "function_LLM": ("LLM",),
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
  
    def execute(self, AGENTLIST, TASKLIST, verbose, manager_LLM=None, function_LLM=None, TOOLLIST=None, **kwargs):
        crew = Crew(
            agents=AGENTLIST,
            tasks=TASKLIST, 
            verbose=verbose, 
            manager_llm=manager_LLM,
            function_calling_llm=function_LLM,
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
"""

        return (formatted_result,)
 
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

class OpenAILLMNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
                "model": (["gpt-4", "gpt-3.5-turbo"], {"default": "gpt-3.5-turbo"}),
            },
            "optional": {
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
 
    def set_openai_llm(self, api_key, model, base_url="https://api.openai.com/v1", max_tokens=1000, temperature=0.7):
        if not api_key:
            raise ValueError("API key is required")
        return (ChatOpenAI(base_url=base_url, api_key=api_key, model=model, max_tokens=max_tokens, temperature=temperature),)

class ClaudeLLMNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
                "model": (["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"], {"default": "claude-3-opus-20240229"}),
            },
            "optional": {
                "max_tokens": ("INT", {"default": 1000, "min": 1, "max": 100000}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1}),
            }
        }
 
    RETURN_TYPES = ("LLM",)
    RETURN_NAMES = ()
    FUNCTION = "set_claude_llm"
    OUTPUT_NODE = True
    CATEGORY = "Crewai/LLMs"
 
    def set_claude_llm(self, api_key, model, max_tokens=1000, temperature=0.7):
        if not api_key:
            raise ValueError("API key is required")
        return (ClaudeLLM(api_key=api_key, model=model, max_tokens=max_tokens, temperature=temperature),)

class GroqLLMNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
                "model": (["mixtral-8x7b-32768", "llama2-70b-4096"], {"default": "mixtral-8x7b-32768"}),
            },
            "optional": {
                "max_tokens": ("INT", {"default": 1000, "min": 1, "max": 100000}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1}),
            }
        }
 
    RETURN_TYPES = ("LLM",)
    RETURN_NAMES = ()
    FUNCTION = "set_groq_llm"
    OUTPUT_NODE = True
    CATEGORY = "Crewai/LLMs"
 
    def set_groq_llm(self, api_key, model, max_tokens=1000, temperature=0.7):
        if not api_key:
            raise ValueError("API key is required")
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
    RETURN_TYPES = ("TOOL",)
    RETURN_NAMES = ()
    FUNCTION = "set_swt"
    OUTPUT_NODE = True
    CATEGORY = "Crewai/tools"
 
    def set_swt(self):
        return (ScrapeWebsiteTool(),)

class SDTNode:
    RETURN_TYPES = ("TOOL",)
    RETURN_NAMES = ()
    FUNCTION = "set_sdt"
    OUTPUT_NODE = True
    CATEGORY = "Crewai/tools"
 
    def set_sdt(self):
        return (SerperDevTool(),)

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

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
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
}
 
# A dictionary that contains the friendly/humanly readable titles for the nodes
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
}