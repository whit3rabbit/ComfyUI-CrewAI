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
os.environ["OPENAI_BASE_URL"]="https://api.groq.com/openai/v1"
os.environ["OPENAI_API_KEY"] = "your key here"


class CrewNode:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "agents": ("AGENTLIST",),
                "tasks":("TASKLIST",),
            },
            "optional": {
                "verbose":("BOOLEAN", {"default": False}),
                "manager_llm": ("LLM",),
                "function_calling_llm": ("LLM",),
                "max_rpm": ("INT", { "default": 25,"min": 0, "max": 100, "step": 1}),
                "language": ("STRING", {"default": "English"}),
                "memory": ("BOOLEAN", {"default": False}),
                "process":(["sequential","hierarchyical"],{"default": "sequential"}),
                # "inputs":("CREW_INPUTS"),
            }
        }
 
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("RESULT",)
 
    FUNCTION = "execute"
 
    #OUTPUT_NODE = False
 
    CATEGORY = "Crewai"
  
    def execute(self,agents,tasks, verbose,manager_llm=None,function_calling_llm=None):
        crew = Crew(agents=agents,tasks=tasks, 
                    verbose=verbose, 
                    manager_llm=manager_llm if manager_llm is not None else None,
                    function_calling_llm=function_calling_llm if function_calling_llm is not None else None
                    )

        result = crew.kickoff()

        return (result,)
 
class AgentNode:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "role": ("STRING", {"multiline": True, "dynamicPrompts": False, "default": ""}),
                "goal": ("STRING", {"multiline": True, "dynamicPrompts": False, "default": ""}),
                "backstory": ("STRING", {"multiline": True, "dynamicPrompts": False, "default": ""}),                                                                       
            },
            "optional":{
                "llm": ("LLM",),
                "function_calling_llm": ("LLM",),
                "tools":("TOOLLIST",),
                "max_iter": ("INT", {"default": 25,"min": 0, "max": 100, "step": 1}),
                "max_rpm": ("INT", {"default": None,"min": 0, "max": 100, "step": 1}),
                "max_execution_time": ("INT", {"default": None,"min": 0, "max": 1000, "step": 1}),
                "verbose": ("BOOLEAN", {"default": False}),
                "allow_delegation": ("BOOLEAN", {"default": True}),
                "step_callback": ("FUNCTION", {"default": None}),
                "cache": ("BOOLEAN", {"default": True}),
            }
        }
 
    RETURN_TYPES = ("AGENT",)
    RETURN_NAMES = ()
 
    FUNCTION = "set_one_agent"
 
    OUTPUT_NODE = True
 
    CATEGORY = "Crewai"
 
    def set_one_agent(self,
                      role,
                      goal, 
                      backstory,
                      llm=None,
                      function_calling_llm=None,
                      tools=[],
                      max_iter=25,
                      max_rpm=None,
                      max_execution_time=None,
                      verbose=False,
                      allow_delegation=True,
                      step_callback=None,
                      cache=True
                      ):
        llm = llm if llm is not None else None
        function_calling_llm = function_calling_llm if function_calling_llm is not None else None
        tools = tools if len(tools)>0 else []
        max_rpm = max_rpm if max_rpm >0 else None
        max_execution_time = max_execution_time if max_execution_time>0 else None

        agent = Agent(
            role=role,
            goal=goal,
            backstory=backstory,
            llm=llm,
            function_calling_llm=function_calling_llm,
            tools=tools,
            max_iter=max_iter,
            max_rpm=max_rpm,
            max_execution_time=max_execution_time,
            verbose=verbose,
            allow_delegation=allow_delegation,
            step_callback=step_callback,
            cache=cache
                      )

        return (agent,)
    
class TaskNode:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "description": ("STRING", {"multiline": True, "dynamicPrompts": False, "default": ""}),
                "agent":("AGENT",),
                "expected_output": ("STRING", {"multiline": True, "dynamicPrompts": False, "default": ""}),                                                                                      
            },
            "optional": {
                "tools": ("TOOLLIST",),
                "async_execution":("BOOLEAN", {"default": False}),
                "context":("TASKLIST",),
                "output_file":("STRING", {"default": "output.md"}),
                "callback":("FUNCTION", {"default": None}),
                "human_feedback":("BOOLEAN", {"default": False}),
                
            }
        }
 
    RETURN_TYPES = ("TASK",)
    RETURN_NAMES = ()
 
    FUNCTION = "set_task"
 
    OUTPUT_NODE = True
 
    CATEGORY = "Crewai"
 
    def set_task(self,
                 description, 
                 agent,
                 expected_output,
                 async_execution,
                 tools=[],
                 context=[],
                 output_file="output.md",
                 callback=None,
                 human_feedback=False
                 ):
        
        tools = tools if len(tools)>0 else []
        context = context if len(context)>0 else []
        callback = callback if callback is not None else None

        task = Task(
            description=description,
            agent=agent,
            expected_output=expected_output,
            async_execution=async_execution,
            tools=tools,
            context=context,
            output_file=output_file,
            callback=callback,
            human_feedback=human_feedback
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

class LLMNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "provider": (["OpenAI", "Claude"],),
                "api_key": ("STRING", {"default": ""}),
             },
            "optional": {
                "base_url": ("STRING", {"default": "https://api.openai.com/v1"}),
                "openai_model": (["gpt-4", "gpt-3.5-turbo"], {"default": "gpt-3.5-turbo"}),
                "claude_model": (["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"], {"default": "claude-3-opus-20240229"}),
                "max_tokens": ("INT", {"default": 1000, "min": 1, "max": 100000}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1}),
            }
        }
 
    RETURN_TYPES = ("LLM",)
    RETURN_NAMES = ()
    FUNCTION = "set_llm"
    OUTPUT_NODE = True
    CATEGORY = "Crewai"
 
    def set_llm(self, provider, api_key, base_url=None, openai_model=None, claude_model=None, max_tokens=1000, temperature=0.7):
        if not api_key:
            raise ValueError("API key is required")

        if provider == "OpenAI":
            return (ChatOpenAI(base_url=base_url, api_key=api_key, model=openai_model, max_tokens=max_tokens, temperature=temperature),)
        elif provider == "Claude":
            return (ClaudeLLM(api_key=api_key, model=claude_model, max_tokens=max_tokens, temperature=temperature),)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

class AgentListNode:
    def __init__(self):
        pass
    
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

        agentList =[]

        agentList.append(agent_01)
        if agent_02 is not None:
            agentList.append(agent_02)
        if agent_03 is not None:
            agentList.append(agent_03)
        if agent_04 is not None:
            agentList.append(agent_04)

        return (agentList,) 

class TaskListNode:
    def __init__(self):
        pass
    
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

        taskList =[]
        taskList.append(task_01)
        if task_02 is not None:
            taskList.append(task_02)
        if task_03 is not None:
            taskList.append(task_03)  
        if task_04 is not None:
            taskList.append(task_04)

        return (taskList,) 

class ToolsListNode:
    def __init__(self):
        pass
    
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
    
    def set_tools(self, tool_01, tool_02=None, tool_03=None,tool_04=None):
        print("within tool list function...")
        toolList =[]
        toolList.append(tool_01)
        if tool_02 is not None:
            toolList.append(tool_02)
        if tool_03 is not None:
            toolList.append(tool_03)  
        if tool_04 is not None:
            toolList.append(tool_04)

        return (toolList,)

class ContextListNode:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context_01": ("TASK",),
            },
            "optional": {
                "context_02": ("TASK",),
                "context_03": ("TASK",),
                "context_04": ("TASK",),
            }
        }
    RETURN_TYPES = ("TASKLIST",)
    RETURN_NAMES = ()
 
    FUNCTION = "set_contexts"
 
    OUTPUT_NODE = True
 
    CATEGORY = "Crewai"
    
    def set_contexts(self, context_01, context_02=None, context_03=None,context_04=None):
        print("within context list function...")
        contextList =[]
        contextList.append(context_01)
        if context_02 is not None:
            contextList.append(context_02)
        if context_03 is not None:
            contextList.append(context_03)  
        if context_04 is not None:
            contextList.append(context_04)

        return (contextList,)
    
class SWTNode:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {

        }
 
    RETURN_TYPES = ("TOOL",)
    RETURN_NAMES = ()
 
    FUNCTION = "set_swt"
 
    OUTPUT_NODE = True
 
    CATEGORY = "Crewai/tools"
 
    def set_swt(self):
        swt = ScrapeWebsiteTool()
        return (swt,)

#SerperDevTool    
class SDTNode:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {

        }
 
    RETURN_TYPES = ("TOOL",)
    RETURN_NAMES = ()
 
    FUNCTION = "set_sdt"
 
    OUTPUT_NODE = True
 
    CATEGORY = "Crewai/tools"
 
    def set_sdt(self):
        sdt = SerperDevTool()
        return (sdt,)

#MDXSearchTool    
class MDXSTNode:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "file_path": ("STRING",{"default": "/file_path"}),                
             },
        }
 
    RETURN_TYPES = ("TOOL",)
    RETURN_NAMES = ()
 
    FUNCTION = "set_mdxst"
 
    OUTPUT_NODE = True
 
    CATEGORY = "Crewai/tools"
 
    def set_mdxst(self,file_path):
        mdxst = MDXSearchTool(mdx=file_path)
        return (mdxst,)
    
#FileReadTool    
class FRTNode:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "file_path": ("STRING",{"default": "/file_path"}),                
             },
        }
 
    RETURN_TYPES = ("TOOL",)
    RETURN_NAMES = ()
 
    FUNCTION = "set_frt"
 
    OUTPUT_NODE = True
 
    CATEGORY = "Crewai/tools"
 
    def set_frt(self,file_path):
        frt = FileReadTool(file_path=file_path)
        return (frt,)
    
# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Crew": CrewNode,
    "Agent": AgentNode,
    "Task": TaskNode,
    "OpenAILLM": OpenAILLMNode,
    "ClaudeLLM": ClaudeLLMNode,
    "GroqLLM": GroqLLMNode,
    "AgentList": AgentListNode,
    "TaskList": TaskListNode,
    "ToolsList": ToolsListNode,
    "SWT": SWTNode,
    "SDT": SDTNode,
    "FRT": FRTNode,
    "MDXST": MDXSTNode,
    "ContextList": ContextListNode,

}
 
# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "CrewtNode": "Node of Crew",
    "AgentNode": "Node of Agent",
    "TaskNode": "Node of Task",
    "OpenAILLMNode": "OpenAI LLM",
    "ClaudeLLMNode": "Claude LLM",
    "GroqLLMNode": "Groq LLM",
    "AgentListNode": "Node of AgentList",
    "TaskListNode": "Node of TaskList",
    "ToolsListNode": "Node of ToolsList",
    "SWTNode": "Node of SWT", 
    "SDTNode": "Node of SDT",
    "FRTNode": "Node of FRT",
    "MDSXSTNode": "Node of MDXST",
    "ContextListNode": "Node of ContextList",
}
 
