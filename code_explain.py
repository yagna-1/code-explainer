#!/usr/bin/env python3
# Code Explainer - An agentic system for explaining code snippets
# Based on Retrieval-Augmented Generation with open-source LLMs

import os
import argparse
import logging
import time
import json
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, asdict, field
import threading
from enum import Enum
import re
from datetime import datetime

import torch
import numpy as np
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel, Field
import uvicorn
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
import requests
from pygments.lexers import get_lexer_by_name, guess_lexer
from pygments.formatters import HtmlFormatter
from pygments.util import ClassNotFound  # Import ClassNotFound for specific exception handling

# LangChain imports
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage
)
from langchain.callbacks.base import BaseCallbackManager
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool
from langchain.callbacks.base import BaseCallbackHandler
from langchain.agents import AgentExecutor, create_react_agent
from langchain.schema import AgentAction, AgentFinish
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManagerForLLMRun  # Import for type hint

# OpenTelemetry
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("code_explainer.log")
    ]
)
logger = logging.getLogger(__name__)

# Configure OpenTelemetry
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
trace.get_tracer_provider().add_span_processor(
    SimpleSpanProcessor(ConsoleSpanExporter())
)

# Constants
DEFAULT_MODEL = "mchochlov/codebert-base-cd-ft"  # Code-aware embedding model
DEFAULT_LLM_API_URL = "http://localhost:8080/v1"  # Default local LLM server endpoint
DEFAULT_SEARCH_URL = "http://localhost:8888"      # Default SearxNG endpoint
COLLECTION_NAME = "code_snippets"
MAX_CODE_CHARS_FOR_AGENT_EXPLANATION = 5000  # Max characters for code input to agent
DEFAULT_LANGUAGE = "auto"  # Default language for code explanation

class LanguageEnum(str, Enum):
    AUTO = "auto"
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CPP = "cpp"
    CSHARP = "csharp"
    GO = "go"
    RUST = "rust"

@dataclass
class ModelConfig:
    """Configuration for models used in the system."""
    embedding_model: str = DEFAULT_MODEL
    llm_api_url: str = DEFAULT_LLM_API_URL
    search_url: str = DEFAULT_SEARCH_URL
    temperature: float = 0.2
    max_new_tokens: int = 1024
    streaming: bool = False
    enable_web_search: bool = True
    agent_verbose: bool = False
    
    def __post_init__(self):
        # Override with environment variables if available
        if os.environ.get("EMBEDDING_MODEL"):
            self.embedding_model = os.environ.get("EMBEDDING_MODEL")
        if os.environ.get("LLM_API_URL"):
            self.llm_api_url = os.environ.get("LLM_API_URL")
        if os.environ.get("SEARCH_URL"):
            self.search_url = os.environ.get("SEARCH_URL")
        if os.environ.get("LLM_TEMPERATURE"):
            self.temperature = float(os.environ.get("LLM_TEMPERATURE"))
        if os.environ.get("ENABLE_WEB_SEARCH"):
            self.enable_web_search = os.environ.get("ENABLE_WEB_SEARCH").lower() == 'true'

class LoggingCallback(BaseCallbackHandler):
    """Callback handler for logging LLM, agent and tool usage."""
    
    def __init__(self):
        """Initialize with empty lists for all event types."""
        self.llm_starts = []
        self.llm_ends = []
        self.llm_errors = []
        self.tool_starts = []
        self.tool_ends = []
        self.tool_errors = []
        self.agent_actions = []
        self.agent_ends = []
        self.chain_starts = []
        self.chain_ends = []
        self.chain_errors = []
        self.reasoning_steps = []
        
    def on_llm_start(self, serialized, prompts, **kwargs):
        """Handle LLM start event."""
        self.llm_starts.append({"prompt": prompts[0]})
        
    def on_llm_end(self, response, **kwargs):
        """Handle LLM end event."""
        self.llm_ends.append({"output": response.generations[0][0].text})
        
    def on_llm_error(self, error, **kwargs):
        """Handle LLM error event."""
        self.llm_errors.append({"error": str(error)})
        
    def on_tool_start(self, serialized, input_str, **kwargs):
        """Handle tool start event."""
        self.tool_starts.append({"tool": serialized["name"], "input": input_str})
        self.reasoning_steps.append({
            "type": "action",
            "content": f"Using {serialized['name']} to search for: {input_str}"
        })
        
    def on_tool_end(self, output, **kwargs):
        """Handle tool end event."""
        self.tool_ends.append({"output": output})
        self.reasoning_steps.append({
            "type": "observation",
            "content": f"Found information: {output[:100]}..." if len(output) > 100 else f"Found information: {output}"
        })
        
    def on_tool_error(self, error, **kwargs):
        """Handle tool error event."""
        self.tool_errors.append({"error": str(error)})
        self.reasoning_steps.append({
            "type": "error",
            "content": f"Error while retrieving information: {str(error)}"
        })
    
    def on_agent_action(self, action, **kwargs):
        """Run on agent action."""
        self.agent_actions.append({"action": action})
        thought = action.log.split("Action:")[0].replace("Thought:", "").strip()
        if thought:
            self.reasoning_steps.append({
                "type": "thought",
                "content": thought
            })
        tool = action.tool.strip()
        tool_input = action.tool_input.strip() if isinstance(action.tool_input, str) else json.dumps(action.tool_input)
        self.reasoning_steps.append({
            "type": "action", 
            "content": f"Using {tool} with input: {tool_input}"
        })
        
    def on_agent_finish(self, finish, **kwargs):
        """Run on agent end."""
        self.agent_ends.append({"output": finish.return_values})
        final_thought = finish.log.replace("Final Answer:", "").strip()
        if final_thought and not any(step["type"] == "result" and step["content"] == final_thought for step in self.reasoning_steps):
            self.reasoning_steps.append({
                "type": "result",
                "content": f"Final Answer: {final_thought}"
            })
        elif not any(step["type"] == "result" for step in self.reasoning_steps):
            self.reasoning_steps.append({
                "type": "result",
                "content": "Generated final explanation"
            })

class LLMInterface:
    """Interface for interaction with Large Language Models."""
    
    def __init__(self, config: ModelConfig):
        """Initialize LLM interface."""
        self.config = config
        self.api_url = config.llm_api_url
        self.temperature = config.temperature
        self.max_new_tokens = config.max_new_tokens
    
    def call_llm(self, prompt: str, callback: Optional[LoggingCallback] = None) -> str:
        """Call LLM API to generate a response to the prompt."""
        headers = {'Content-Type': 'application/json'}
        data = {
            'model': 'phi-2-dpo.Q4_K_M.gguf',  # or the model ID of the locally running LLM
            'prompt': prompt,
            'temperature': self.temperature,
            'max_tokens': self.max_new_tokens,
            'stop': ["Observation:", "Human:"]
        }

        start_time = time.time()
        try:
            with tracer.start_as_current_span("llm_api_call") as span:
                span.set_attribute("prompt_length", len(prompt))
                
                response = requests.post(
                    f"{self.api_url}/completions", 
                    headers=headers, 
                    json=data
                )
                response.raise_for_status()
                result = response.json()
                
                span.set_attribute("response_time", time.time() - start_time)
                
                return result.get('choices', [{}])[0].get('text', '')
        except Exception as e:
            logger.error(f"LLM API error: {str(e)}")
            if callback:
                callback.llm_errors.append({"error": str(e)})
            return f"Error: Failed to generate explanation. {str(e)}"
    
    def langchain_llm(self):
        """Create a LangChain-compatible LLM wrapper for our API."""
        from langchain.llms.base import LLM
        
        class CodeLLM(LLM):
            """LangChain wrapper for our LLM API."""
            
            llm_interface: LLMInterface = self
            
            @property
            def _llm_type(self):
                return "code-explainer-llm"
                
            def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> str:
                return self.llm_interface.call_llm(prompt)
                
            @property
            def _identifying_params(self):
                return {"api_url": self.llm_interface.api_url}
        
        return CodeLLM()

    def _determine_language(self, code: str, specified_language: Optional[str]) -> str:
        """Determines the language, guessing if 'auto' or not specified."""
        if specified_language and specified_language.lower() != "auto":
            logger.info(f"Language specified by user: {specified_language.lower()}")
            return specified_language.lower()

        if not code.strip():
            logger.info("Code input is empty, defaulting to 'text' for language detection.")
            return "text"

        # Improved: Try to match using Pygments' get_lexer_by_name and analyse_text, but also check for common language keywords
        priority_languages = [
            lang.value for lang in LanguageEnum if lang.value != LanguageEnum.AUTO.value
        ]

        best_guess_lang = None
        highest_score = 0.0

        # Additional: Keyword-based quick check for common languages
        keyword_map = {
            "python": ["def ", "import ", "self", "print(", "lambda", "yield", "None", "True", "False"],
            "javascript": ["function ", "var ", "let ", "const ", "=>", "console.log", "document.", "window."],
            "typescript": ["interface ", "implements ", "extends ", ": number", ": string", "as ", "readonly"],
            "java": ["public class", "static void main", "System.out.println", "implements ", "extends "],
            "cpp": ["#include", "std::", "cout<<", "cin>>", "->", "::", "template<", "using namespace"],
            "csharp": ["using System", "namespace ", "public class", "void Main(", "Console.WriteLine"],
            "go": ["func ", "package main", "import \"", "fmt.", "go func", "chan ", ":="],
            "rust": ["fn main()", "let mut", "println!", "use std::", "impl ", "pub struct", "match ", "enum "],
        }
        for lang, keywords in keyword_map.items():
            for kw in keywords:
                if kw in code:
                    logger.info(f"Keyword-based detection: matched '{kw}' for language '{lang}'")
                    return lang

        # Pygments scoring
        for lang_name in priority_languages:
            try:
                lexer = get_lexer_by_name(lang_name)
                score = lexer.analyse_text(code)
                logger.debug(f"Analyzed code with {lang_name} lexer, score: {score}")
                if score > highest_score:
                    highest_score = score
                    best_guess_lang = lang_name
            except ClassNotFound:
                logger.debug(f"Lexer for {lang_name} not found during priority check.")
            except Exception as e:
                logger.warning(f"Error analyzing with lexer for {lang_name}: {e}")
                continue

        if best_guess_lang and highest_score > 0.01:
            logger.info(f"Determined language using priority list: {best_guess_lang} (score: {highest_score:.2f})")
            return best_guess_lang

        logger.info("Falling back to guess_lexer as priority language analysis did not yield a confident match (highest score <= 0.01).")
        try:
            lexer = guess_lexer(code)
            guessed_alias = lexer.aliases[0] if lexer.aliases else "text"
            logger.info(f"Guessed language via guess_lexer (first alias): {guessed_alias.lower()} (Lexer: {lexer.name})")
            return guessed_alias.lower()
        except ClassNotFound:
            logger.warning(f"guess_lexer could not identify the language. Defaulting to 'text'. Code (first 100 chars): '{code[:100]}'")
            return "text"
        except Exception as e:
            logger.error(f"Unexpected error in guess_lexer: {e}. Defaulting to 'text'. Code (first 100 chars): '{code[:100]}'")
            return "text"

# API Models
class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class CodeExplainRequest(BaseModel):
    code: str
    language: Optional[str] = None  # Added language field
    enable_web_search_override: Optional[bool] = Field(None, alias="enable_web_search")
    conversation_history: Optional[List[Message]] = None

class CodeExplainResponse(BaseModel):
    explanation: str
    updated_conversation_history: List[Message]

class AgentExplainRequest(BaseModel):
    code: str
    language: Optional[str] = None  # Added language field
    enable_web_search_override: Optional[bool] = Field(None, alias="enable_web_search")
    conversation_history: Optional[List[Message]] = None

class AgentExplainResponse(BaseModel):
    explanation: str
    reasoning_steps: List[Dict]
    updated_conversation_history: List[Message]

class RetrievalTool(BaseTool):
    """Tool for retrieving similar code snippets."""
    
    name: str = "code_retrieval"
    description: str = "Retrieve similar code snippets to help explain the current code. Input should be a section of the code you want to find references for."
    code_explainer: Any = Field(None, exclude=True)  # Add the code_explainer field with exclude=True to prevent serialization issues
    
    def __init__(self, code_explainer):
        """Initialize retrieval tool."""
        super().__init__(code_explainer=code_explainer) # Pass code_explainer to super

    def _run(self, query: str) -> str:
        """Use the tool."""
        # Retrieve from local knowledge base only, as per original intent
        results = self.code_explainer.hybrid_retrieve(query, use_web=False) 
        # Return results as a JSON string for the agent
        return json.dumps([res.get('content', '') for res in results])

class WebSearchTool(BaseTool):
    """Tool for searching external resources."""
    
    name: str = "web_search"
    description: str = "Search external resources like Stack Overflow and GitHub for code explanations. Use when you need specific information about libraries, functions or patterns."
    code_explainer: Any = Field(None, exclude=True)  # Add the code_explainer field with exclude=True to prevent serialization issues
    
    def __init__(self, code_explainer: "CodeExplainer"):
        """Initialize web search tool."""
        super().__init__(code_explainer=code_explainer) # Pass code_explainer to super

    def _run(self, query: str) -> str:
        """Use the tool."""
        results = self.code_explainer.search_web(query)
        # Format results concisely for the agent
        return json.dumps([f"{res.get('title', '')}: {res.get('content', '')[:200]}... ({res.get('url')})" for res in results])

class CodeExplainer:
    """Main class handling code explanation using RAG and LLMs."""
    
    def __init__(self, config: ModelConfig = None):
        """Initialize the code explainer with models and vector store."""
        self.config = config or ModelConfig()
        
        logger.info(f"Loading embedding model: {self.config.embedding_model}")
        with tracer.start_as_current_span("load_embedding_model"):
            self.embedding_model = SentenceTransformer(self.config.embedding_model)
        
        # Initialize cross-encoder for reranking
        with tracer.start_as_current_span("load_reranker"):
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Initialize LLM interface
        self.llm_interface = LLMInterface(self.config)
        
        # Initialize vector store
        with tracer.start_as_current_span("initialize_vector_store"):
            self.chroma_client = chromadb.Client()
            try:
                self.collection = self.chroma_client.get_collection(COLLECTION_NAME)
                logger.info(f"Using existing collection: {COLLECTION_NAME}")
            except:
                logger.info(f"Creating new collection: {COLLECTION_NAME}")
                self.collection = self.chroma_client.create_collection(COLLECTION_NAME)
        
        # Initialize tools first
        self.retrieval_tool = RetrievalTool(code_explainer=self)
        self.web_search_tool = WebSearchTool(code_explainer=self)
        
        # Initialize ReAct agent (default one)
        self.agent_executor = self._create_agent_executor(use_web_search_for_this_agent=self.config.enable_web_search)
    
    def _create_agent_executor(self, use_web_search_for_this_agent: bool) -> AgentExecutor:
        """Helper to create an AgentExecutor instance with specific web search config."""
        tools = [self.retrieval_tool]
        if use_web_search_for_this_agent:
            tools.append(self.web_search_tool)
        
        from langchain import hub
        prompt = hub.pull("hwchase17/react")
        
        llm = self.llm_interface.langchain_llm()
        agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
        
        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            verbose=self.config.agent_verbose,
            handle_parsing_errors=True,
            max_iterations=300,           
            max_execution_time=300 
        )

    async def explain_code_agent_async(self, code: str, language: Optional[str] = None, enable_web_search_override: Optional[bool] = None, conversation_history: Optional[List['Message']] = None) -> Tuple[str, List[Dict], List['Message']]:
        """Explain code using the agent asynchronously, with conversation history and improvement suggestions."""
        if len(code) > MAX_CODE_CHARS_FOR_AGENT_EXPLANATION:
            explanation = "The provided code is too large for the agent to process effectively in detail. Please try explaining smaller parts of the code, or use the 'Explain Code' button for a general overview (which might also be limited by code size)."
            logger.warning(f"Code input for agent is too large: {len(code)} chars. Max allowed: {MAX_CODE_CHARS_FOR_AGENT_EXPLANATION}")
            
            updated_history = conversation_history or []
            user_code_summary = code[:100] + "..." if len(code) > 100 else code
            user_prompt_content = f"Explain this code and suggest potential improvements if any: {user_code_summary}"
            updated_history.append(Message(role="user", content=user_prompt_content))
            updated_history.append(Message(role="assistant", content=explanation))
            return explanation, [], updated_history

        callback = LoggingCallback()
        
        history_str = ""
        if conversation_history:
            for msg in conversation_history:
                history_str += f"{msg.role.capitalize()}: {msg.content}\n"
        
        determined_language = self.llm_interface._determine_language(code, language)
        language_hint = f" (Detected language: {determined_language})" if determined_language != "text" else ""

        prompt_input = f"""
{history_str}
User: Explain this code{language_hint} and suggest potential improvements:

```
{code}
```
"""
        input_data = {"input": prompt_input.strip()}
        
        agent_to_use = self.agent_executor # Start with default
        if enable_web_search_override is not None:
            logger.info(f"Using web search override for agent request: {enable_web_search_override}")
            agent_to_use = self._create_agent_executor(use_web_search_for_this_agent=enable_web_search_override)
            
        try:
            result = await agent_to_use.ainvoke(input_data, callbacks=[callback])
            explanation = result.get("output", "Agent did not produce an output.")
        except Exception as e:
            logger.error(f"Agent execution error: {e}", exc_info=True)
            explanation = f"Error during agent execution: {e}"
            callback.reasoning_steps.append({"type": "error", "content": explanation})
            
        updated_history = conversation_history or []
        updated_history.append(Message(role="user", content=f"Explain this code and suggest improvements: {code}"))
        updated_history.append(Message(role="assistant", content=explanation))
        
        return explanation, callback.reasoning_steps, updated_history
    
    def explain_code(self, code: str, language: Optional[str] = None, enable_web_search_override: Optional[bool] = None, conversation_history: Optional[List[Message]] = None) -> Tuple[str, List['Message']]:
        """Main method to explain code using RAG pipeline, with conversation history and improvement suggestions."""
        input_conv_history = conversation_history or []

        determined_language = self.llm_interface._determine_language(code, language)
        language_info_for_prompt = f"The code's detected language is {determined_language}." if determined_language != "text" \
                               else "The programming language of the code is unknown or not specified."

        # Build history string from the input_conv_history for the prompt.
        prompt_history_str = ""
        for msg in input_conv_history:
            prompt_history_str += f"{msg.role.capitalize()}: {msg.content}\n"

        is_follow_up = False
        last_user_message_content = ""
        if input_conv_history and input_conv_history[-1].role == "user":
            last_user_message_content = input_conv_history[-1].content.strip()
            
            normalized_code_explain_request_1 = f"explain and improve: {code.strip()}".lower()
            normalized_code_explain_request_2 = f"explain this code and suggest potential improvements: {code.strip()}".lower() # Matches agent's phrasing
            normalized_code_explain_request_3 = f"Explain the following code, its purpose, how it works, notable techniques, patterns, and suggest potential improvements if applicable.\nCODE TO EXPLAIN:\n```\n{code.strip()}\n```".lower() # Matches conceptual_user_request_text
            normalized_last_user_message = last_user_message_content.lower()

            # A message is a follow-up if it's not the code itself or a generic explain command for this code.
            if normalized_last_user_message != code.strip().lower() and \
               normalized_last_user_message != normalized_code_explain_request_1 and \
               normalized_last_user_message != normalized_code_explain_request_2 and \
               normalized_last_user_message != normalized_code_explain_request_3:
                is_follow_up = True
                logger.info(f"Identified as follow-up. Last user message: '{last_user_message_content}'")

        retrieved_context = self.hybrid_retrieve(code, use_web=enable_web_search_override)
        context_str_for_rag = "\n\n".join([f"REFERENCE CODE {i+1}:\n{item['content']}"
                                           for i, item in enumerate(retrieved_context)])

        if is_follow_up:
            # Prompt for answering a follow-up question.
            prompt = f"""You are a helpful AI assistant in a conversation about the following code:
```
{code}
```
{language_info_for_prompt}

Conversation History (most recent messages last):
{prompt_history_str}
Your task is to respond to the User's *very last message*: "{last_user_message_content}"

Follow these guidelines for your response:
- If the user's last message ("{last_user_message_content}") is a simple social phrase like "thank you", "ok", "sounds good", "got it", etc., then give a short, polite reply (e.g., "You're welcome!", "Great! Is there anything else about the code I can help with?"). Do not explain the code again or provide improvement suggestions unless explicitly asked in this last message.
- If the user's last message ("{last_user_message_content}") asks a specific question or makes a request about the "Code Under Discussion" (e.g., its purpose, how a part works, potential improvements, real-life applications, or a modification request), then answer that specific question or address that request directly and concisely.
  Use the "Potentially Relevant Information" below only if it directly helps answer this specific question about the code.
  Potentially Relevant Information: {context_str_for_rag if context_str_for_rag.strip() else "None available."}
  Only explain parts of the code or suggest improvements if the user's *last message specifically asks for that*.
- If the user's last message ("{last_user_message_content}") is a general knowledge question not directly related to the code (e.g., "What is the capital of France?"), answer it normally as a helpful assistant.

IMPORTANT:
- Avoid repeating prior explanations unless the user's last message specifically asks for a repetition or clarification.
- Keep your response concise and relevant to the user's last message. Do not include unnecessary details or verbose outputs.

Assistant:"""

        else:
            # Prompt for an initial explanation.
            conceptual_user_request_text_for_prompt = f"Explain the following code, its purpose, how it works, notable techniques, patterns, and suggest potential improvements if applicable.\nCODE TO EXPLAIN:\n```\n{code}\n```"
            
            current_prompt_history_for_initial = prompt_history_str
            # Ensure the conceptual request for this code is the last user turn in the prompt history
            if not current_prompt_history_for_initial.strip().endswith(f"User: {conceptual_user_request_text_for_prompt.strip()}"):
                 current_prompt_history_for_initial += f"User: {conceptual_user_request_text_for_prompt}\n"

            prompt = f"""{current_prompt_history_for_initial}Assistant: ({language_info_for_prompt})
"""
            if context_str_for_rag:
                prompt += f"I have found these relevant code references from my knowledge base:\n{context_str_for_rag}\n"
            prompt += "\nEXPLANATION AND IMPROVEMENTS:"
        
        explanation = self.invoke_llm(prompt)

        # Clean the explanation if it's a follow-up and starts with the user's question.
        if is_follow_up:
            # Regex to find "User: <question text, possibly quoted> \n Assistant:" at the beginning of the string.
            # It handles optional quotes around the question and variations in newlines/spacing.
            # re.escape is used for last_user_message_content to handle special characters in the question.
            user_question_prefix_pattern = re.compile(
                r"^\s*User:\s*\"?" + re.escape(last_user_message_content) + r"\"?\s*\n+\s*Assistant:\s*",
                re.IGNORECASE | re.DOTALL
            )
            match = user_question_prefix_pattern.match(explanation)
            if match:
                logger.info("LLM response for follow-up started with user question prefix. Stripping it.")
                explanation = explanation[match.end():].strip()


        # Construct the final updated history to return.
        updated_conv_history = list(input_conv_history) 

        if not is_follow_up:
            # For initial explanations, ensure the user's request to explain this code is logged.
            # This is mainly for when "Explain Code" button is pressed, which might not send this specific request in history.
            conceptual_user_request_for_history = f"Explain and improve: {code}" # Simplified for logging
            
            add_conceptual_request_to_history = True
            if updated_conv_history:
                last_msg = updated_conv_history[-1]
                if last_msg.role == "user" and last_msg.content.strip().lower() == conceptual_user_request_for_history.lower():
                    add_conceptual_request_to_history = False
            
            if add_conceptual_request_to_history:
                # Avoid adding if the immediate previous message is already this conceptual request
                # This can happen if the frontend sends the "Explain and improve: code" as the last user message
                # and then calls this endpoint.
                if not (updated_conv_history and \
                        updated_conv_history[-1].role == "user" and \
                        updated_conv_history[-1].content.strip().lower() == conceptual_user_request_for_history.lower()):
                    updated_conv_history.append(Message(role="user", content=conceptual_user_request_for_history))
        
        # Add assistant's (cleaned) response to the history.
        # Avoid adding duplicate assistant messages.
        if not updated_conv_history or \
           updated_conv_history[-1].role != "assistant" or \
           updated_conv_history[-1].content != explanation:
            updated_conv_history.append(Message(role="assistant", content=explanation))
        
        return explanation, updated_conv_history

    def add_to_knowledge_base(self, code_snippets: List[str], metadata: List[Dict] = None):
        """Add code snippets to the vector store for future retrieval."""
        if not metadata:
            metadata = [{}] * len(code_snippets)
            
        # Generate embeddings
        embeddings = self.embedding_model.encode(code_snippets, convert_to_tensor=False)
        
        # Add to Chroma collection
        self.collection.add(
            embeddings=embeddings,
            documents=code_snippets,
            metadatas=metadata,
            ids=[f"doc_{i}" for i in range(len(code_snippets))]
        )
        logger.info(f"Added {len(code_snippets)} code snippets to knowledge base")
    
    def search_web(self, query: str, num_results: int = 5) -> List[Dict]:
        """Search external resources using SearxNG."""
        try:
            response = requests.get(
                f"{self.config.search_url}/search",
                params={
                    "q": f"code explanation {query}",
                    "format": "json",
                    "engines": "github,stackoverflow,gitxplore,codeberg",
                    "language": "en-US"
                }
            )
            response.raise_for_status()
            results = response.json().get("results", [])
            return results[:num_results]
        except Exception as e:
            logger.error(f"Web search error: {str(e)}")
            return []
    
    def retrieve_similar_code(self, code_query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve similar code snippets from vector store."""
        try:
            query_embedding = self.embedding_model.encode(code_query, convert_to_tensor=False)
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            docs = []
            for i, doc in enumerate(results['documents'][0]):
                docs.append({
                    'content': doc,
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'score': results['distances'][0][i] if 'distances' in results else 0
                })
            return docs
        except Exception as e:
            logger.error(f"Vector retrieval error: {str(e)}")
            return []
    
    def hybrid_retrieve(self, code_query: str, top_k: int = 5, use_web: Optional[bool] = None) -> List[Dict]:
        """Combine vector retrieval and web search for comprehensive results."""
        effective_use_web = self.config.enable_web_search # Default
        if use_web is not None: # If an override is passed, it takes precedence
            effective_use_web = use_web
            logger.info(f"Web search for hybrid_retrieve set to: {effective_use_web} (override)")
        else:
            logger.info(f"Web search for hybrid_retrieve set to: {effective_use_web} (config default)")

        # Get vector results
        vector_results = self.retrieve_similar_code(code_query, top_k)
        
        # Perform web search if needed
        web_results = []
        if effective_use_web and len(vector_results) < top_k:
            web_results = self.search_web(code_query, top_k - len(vector_results))
            web_results = [{'content': r.get('content', r.get('snippet', '')), 
                           'metadata': {'url': r.get('url'), 'source': 'web'}, 
                           'score': 0} for r in web_results]
        
        # Combine results
        combined_results = vector_results + web_results
        
        # Rerank if we have more than one result
        if len(combined_results) > 1:
            combined_results = self.rerank_results(code_query, combined_results)
            
        return combined_results[:top_k]
    
    def rerank_results(self, query: str, results: List[Dict]) -> List[Dict]:
        """Rerank results using cross-encoder."""
        texts = [result['content'] for result in results]
        pairs = [[query, text] for text in texts]
        
        scores = self.reranker.predict(pairs)
        
        # Update scores and sort
        for i, score in enumerate(scores):
            results[i]['score'] = float(score)
        
        return sorted(results, key=lambda x: x['score'], reverse=True)
    
    def invoke_llm(self, prompt: str) -> str:
        """Call the LLM API to generate code explanations."""
        headers = {'Content-Type': 'application/json'}
        data = {
            'model': 'phi-2-dpo.Q4_K_M.gguf',  # or the model ID of your locally running LLM
            'prompt': prompt,
            'temperature': self.config.temperature,
            'max_tokens': self.config.max_new_tokens,
        }
        
        try:
            response = requests.post(
                f"{self.config.llm_api_url}/completions", 
                headers=headers, 
                json=data
            )
            response.raise_for_status()
            return response.json().get('choices', [{}])[0].get('text', '')
        except Exception as e:
            logger.error(f"LLM API error: {str(e)}")
            return f"Error: Failed to generate explanation. {str(e)}"

# Initialize FastAPI
app = FastAPI(title="Code Explainer API")
code_explainer = CodeExplainer()

# Mount static files (frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the frontend HTML page."""
    with open(os.path.join("static", "index.html"), "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/explain", response_model=CodeExplainResponse)
async def explain_code_endpoint(request: CodeExplainRequest):
    """API endpoint to explain code."""
    try:
        explanation, updated_history = code_explainer.explain_code(
            request.code,
            language=request.language,  # Pass language
            enable_web_search_override=request.enable_web_search_override,
            conversation_history=request.conversation_history
        )
        return {"explanation": explanation, "updated_conversation_history": updated_history}
    except Exception as e:
        logger.error(f"Error processing explanation request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate explanation: {str(e)}")

@app.post("/agent_explain", response_model=AgentExplainResponse)
async def agent_explain_code_endpoint(request: AgentExplainRequest, background_tasks: BackgroundTasks):
    """API endpoint to explain code using agent asynchronously."""
    try:
        explanation, reasoning_steps, updated_history = await code_explainer.explain_code_agent_async(
            request.code,
            language=request.language,  # Pass language
            enable_web_search_override=request.enable_web_search_override,
            conversation_history=request.conversation_history
        )
        return {"explanation": explanation, "reasoning_steps": reasoning_steps, "updated_conversation_history": updated_history}
    except Exception as e:
        logger.error(f"Error processing agent explanation request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate explanation: {str(e)}")

# CLI interface
def main():
    """Command line interface for code explanation."""
    parser = argparse.ArgumentParser(description="Code Explainer")
    parser.add_argument("-c", "--code", help="Code to explain")
    parser.add_argument("-f", "--file", help="File containing code to explain")
    parser.add_argument("--server", action="store_true", help="Run as API server")
    parser.add_argument("--port", type=int, default=8000, help="API server port")
    args = parser.parse_args()
    
    if args.server:
        # Run as API server
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    else:
        # Run as CLI tool
        code = ""
        if args.file:
            with open(args.file, "r") as f:
                code = f.read()
        elif args.code:
            code = args.code
        else:
            parser.print_help()
            return
        
        explainer = CodeExplainer()
        explanation = explainer.explain_code(code)
        print(explanation)

if __name__ == "__main__":
    main()