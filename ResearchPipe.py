"""
title: arXiv Research Pipe
description: Function Pipe made to create summary of searches using arXiv.org for relevant papers on a topic and web scrape for more contextual information.
author: Haervwe
author_url: https://github.com/Haervwe/open-webui-tools/
funding_url: https://github.com/Haervwe/open-webui-tools
git: https://github.com/Haervwe/open-webui-tools  
version: 0.4.3
"""

import logging
import random
import math
import json
import aiohttp
import asyncio
from typing import List, Dict, Union, Optional, AsyncGenerator, Callable, Awaitable
from dataclasses import dataclass
from pydantic import BaseModel, Field
from open_webui.constants import TASKS
from bs4 import BeautifulSoup
import re
from open_webui.main import generate_chat_completions
from open_webui.models.users import User

name = "Research Pipe"


def setup_logger():
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.set_name(name)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger


logger = setup_logger()


class Pipe:
    __current_event_emitter__: Callable[[dict], Awaitable[None]]
    __question__: str
    __model__: str

    class Valves(BaseModel):
        MODEL: str = Field(
            default="llama3.1:latest",
            description="Model to use (model id from ollama)",
        )
        TAVILY_API_KEY: str = Field(
            default="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",
            description="API key for Tavily search service",
        )
        MAX_SEARCH_RESULTS: int = Field(
            default=3, description="Maximum number of search results to fetch per query"
        )
        ARXIV_MAX_RESULTS: int = Field(
            default=3, description="Maximum number of arXiv papers to fetch"
        )
        TREE_DEPTH: int = Field(
            default=4, description="Maximum depth of the research tree"
        )
        TREE_BREADTH: int = Field(
            default=3, description="Number of research paths to explore at each node"
        )
        EXPLORATION_WEIGHT: float = Field(
            default=1.414, description="Controls exploration vs exploitation"
        )
        TEMPERATURE_DECAY: bool = Field(
            default=True,
            description="Activates Temperature, lowers the Temperature in each subsequent step",
        )
        DINAMYC_TEMPERATURE_DECAY: bool = Field(
            default=True,
            description="Activates Temperature Dynamic mapping, giving higher creativity for lower scored parent nodes",
        )
        TEMPERATURE_MAX: float = Field(
            default=1.4,
            description="Temperature for starting the research process with Temperature decay if active",
        )
        TEMPERATURE_MIN: float = Field(
            default=0.5,
            description="Temperature the research process will attempt to converge to with Temperature decay",
        )

    def __init__(self):
        self.valves = self.Valves()

    def pipes(self) -> list[dict[str, str]]:
        return [{"id": f"{name}-{self.valves.MODEL}", "name": f"{name}"}]

    def resolve_model(self, body: dict) -> str:
        model_id = body.get("model")
        without_pipe = ".".join(model_id.split(".")[1:])
        return without_pipe.replace(f"{name}-", "")

    def resolve_question(self, body: dict) -> str:
        return body.get("messages")[-1].get("content").strip()

    async def search_arxiv(self, query: str) -> List[Dict]:
        await self.emit_status("tool", f"Fetching arXiv papers for: {query}...", False)
        try:
            arxiv_url = "http://export.arxiv.org/api/query"
            params = {
                "search_query": f"{query}",
                "max_results": self.valves.ARXIV_MAX_RESULTS,
                "sortBy": "relevance",
            }
            async with aiohttp.ClientSession() as session:
                async with session.get(arxiv_url, params=params) as response:
                    logger.debug(f"arXiv API response status: {response.status}")
                    if response.status == 200:
                        data = await response.text()
                        soup = BeautifulSoup(data, "xml")
                        entries = soup.find_all("entry")
                        return [
                            {
                                "title": entry.find("title").text,
                                "url": entry.find("link")["href"],
                                "content": entry.find("summary").text,
                            }
                            for entry in entries
                        ]
        except Exception as e:
            logger.error(f"arXiv search error: {e}")
        return []

    async def search_web(self, query: str) -> List[Dict]:
        if not self.valves.TAVILY_API_KEY:
            return []
        async with aiohttp.ClientSession() as session:
            try:
                url = "https://api.tavily.com/search"
                headers = {"Content-Type": "application/json"}
                data = {
                    "api_key": self.valves.TAVILY_API_KEY,
                    "query": query,
                    "max_results": self.valves.MAX_SEARCH_RESULTS,
                    "search_depth": "advanced",
                }
                async with session.post(url, headers=headers, json=data) as response:
                    logger.debug(f"Tavily API response status: {response.status}")
                    if response.status == 200:
                        result = await response.json()
                        results = result.get("results", [])
                        return [
                            {
                                "title": result["title"],
                                "url": result["url"],
                                "content": result["content"],
                                "score": result["score"],
                            }
                            for result in results
                        ]
                    else:
                        logger.error(f"Tavily API error: {response.status}")
                        return []
            except Exception as e:
                logger.error(f"Search error: {e}")
                return []

    async def gather_research(self, topic: str) -> List[Dict]:
        await self.emit_status("tool", "Researching...", False)
        web_query, arxiv_query = await self.preprocess_query(topic)
        logger.debug(f"Enhanced queries - arXiv: {arxiv_query}")

        await self.emit_status(
            "tool",
            f"Enhanced queries - Web: {web_query} | arXiv: {arxiv_query}",
            False,
        )
        # web_research = []  # Web search disabled in this version
        web_research = await self.search_web(web_query)

        await self.emit_status("tool", f"Web sources found: {len(web_research)}", False)
        arxiv_research = await self.search_arxiv(arxiv_query)
        logger.debug(f"arxiv_research: {arxiv_research}")
        await self.emit_status(
            "tool", f"ArXiv papers found: {len(arxiv_research)}", False
        )
        research = web_research + arxiv_research
        logger.debug(
            f"Research Result: ArXiv papers: {len(arxiv_research)}, Web sources: {len(web_research)}"
        )
        await self.emit_status(
            "user",
            f"Research gathered: ArXiv papers: {len(arxiv_research)}, Web sources: {len(web_research)}",
            True,
        )
        return research

    async def preprocess_query(self, query: str) -> tuple[str, str]:
        prompt_web = f"""
        Enhance the following query to improve the relevance of web search results:
        - Focus on adding relevant keywords, synonyms, or contextual phrases
        - Only output the enhanced query without explanations

        Initial query: "{query}"

        Enhanced web search query:
        """
        web_query = await self.get_completion(prompt_web)

        prompt_arxiv = f"""
        Format an optimized query for the arXiv API based on the following input:
        - Use arXiv's query syntax (AND, OR, NOT) and search fields (ti, au, abs, cat)
        - Only output the formatted arXiv API query without explanations

        Initial query: "{query}"

        arXiv categories:
        - cs.AI: Artificial Intelligence
        - cs.LG: Machine Learning 
        - cs.CV: Computer Vision
        - cs.CL: Computation and Language (NLP)
        - cs.RO: Robotics
        - stat.ML: Machine Learning (Statistics)
        - math.OC: Optimization and Control
        - physics: Physics
        - q-bio: Quantitative Biology
        - q-fin: Quantitative Finance
        - econ: Economics

        Enhanced arXiv search query (API format):
        """
        arxiv_query = await self.get_completion(prompt_arxiv)

        return web_query, arxiv_query

    async def get_streaming_completion(
        self, messages, temperature: float = 1
    ) -> AsyncGenerator[str, None]:
        try:
            form_data = {
                "model": self.__model__,
                "messages": messages,
                "stream": True,
                "temperature": temperature,
            }
            response = await generate_chat_completions(
                self.__request__,
                form_data,
                user=self.__user__,
            )
            if not hasattr(response, "body_iterator"):
                raise ValueError("Response does not support streaming")
            async for chunk in response.body_iterator:
                for part in self.get_chunk_content(chunk):
                    yield part
        except Exception as e:
            raise RuntimeError(f"Streaming completion failed: {e}")

    async def get_completion(self, messages) -> str:
        response = await generate_chat_completions(
            self.__request__,
            {
                "model": self.__model__,
                "messages": [{"role": "user", "content": messages}],
            },
            user=self.__user__,
        )
        return response["choices"][0]["message"]["content"]

    async def get_improvement(self, content: str, topic: str) -> str:
        prompt = f"""
        How can this research synthesis be improved?
        Topic: {topic}

        Current synthesis:
        {content}

        Suggest ONE specific improvement in a single sentence.
        """
        return await self.get_completion(prompt)

    async def synthesize_research(
        self, research: List[Dict], topic: str, temperature
    ) -> str:
        research_text = "\n\n".join(
            f"Title: {r['title']}\nContent: {r['content']}\nURL: {r['url']}"
            for r in research
        )
        prompt = f"""
        Create a research synthesis on the topic: {topic}

        Available research:
        {research_text}

        Create a comprehensive synthesis that:
        1. Integrates the sources with links
        2. Highlights key findings
        3. Maintains academic rigor while being accessible
        """
        complete = ""
        async for chunk in self.get_streaming_completion(
            [{"role": "user", "content": prompt}], temperature
        ):
            complete += chunk
            await self.emit_message(chunk)
        return complete

    async def evaluate_content(self, content: str, topic: str) -> float:
        logger.debug(f"Evaluating content for topic: {topic[:50]}...")
        prompt = f"""
        Evaluate the quality of the research synthesis provided below:

        Content: "{content}"
        Topic: "{topic}"

        Consider the following criteria:
        1. Integration of sources.
        2. Depth of analysis.
        3. Clarity and coherence.
        4. Relevance to the topic.

        Provide a single numeric score between 1 and 10.
        """
        try:
            result = await self.get_completion(prompt)
            match = re.search(r"\b(10|\d(\.\d+)?)\b", result.strip())
            if match:
                score = float(match.group())
                if 1.0 <= score <= 10.0:
                    return score
                else:
                    logger.debug(f"Score out of range: {score}. Result was: {result}")
                    return 0.0
            else:
                logger.debug(f"No valid number in response: {result}")
                return 0.0
        except Exception as e:
            logger.debug(f"Error during evaluation: {e}")
            return 0.0
        finally:
            logger.debug("Evaluation complete.")
            return 0.0

    def get_chunk_content(self, chunk):
        chunk_str = chunk
        if chunk_str.startswith("data: "):
            chunk_str = chunk_str[6:]
        chunk_str = chunk_str.strip()
        if chunk_str == "[DONE]" or not chunk_str:
            return
        try:
            chunk_data = json.loads(chunk_str)
            if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                delta = chunk_data["choices"][0].get("delta", {})
                if "content" in delta:
                    yield delta["content"]
        except json.JSONDecodeError:
            logger.error(f'ChunkDecodeError: unable to parse "{chunk_str[:100]}"')

    async def get_message_completion(self, model: str, content):
        async for chunk in self.get_streaming_completion(
            [{"role": "user", "content": content}]
        ):
            yield chunk

    async def stream_prompt_completion(self, prompt, **format_args):
        complete = ""
        async for chunk in self.get_message_completion(
            self.__model__,
            prompt.format(**format_args),
        ):
            complete += chunk
            await self.emit_message(chunk)
        return complete

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __event_emitter__=None,
        __task__=None,
        __model__=None,
        __request__=None,
    ) -> str:
        model = self.valves.MODEL
        logger.debug(f"Model {model}")
        logger.debug(f"User: {__user__}")
        self.__user__ = User(**__user__)
        self.__request__ = __request__
        if __task__ and __task__ != TASKS.DEFAULT:
            logger.debug(f"Model {TASKS}")
            response = await generate_chat_completions(
                self.__request__,
                {"model": model, "messages": body.get("messages"), "stream": False},
                user=self.__user__,
            )
            content = response["choices"][0]["message"]["content"]
            return f"{name}: {content}"
        logger.debug(f"Pipe {name} received: {body}"[:70])
        self.__current_event_emitter__ = __event_emitter__
        self.__model__ = model

        topic = body.get("messages", [])[-1].get("content", "").strip()

        await self.progress("Initializing research process...")
        initial_temperature = (
            self.valves.TEMPERATURE_MAX if self.valves.TEMPERATURE_DECAY else 1
        )
        # Gather initial research and synthesize content directly
        # initial_research = await self.gather_research(topic)
        #         # Initial research
        initial_research_1 = await self.gather_research(topic)
        initial_research_2 = await self.gather_research(topic)
        initial_research_3 = await self.gather_research(topic)
        initial_research_4 = await self.gather_research(topic)
        initial_research_5 = await self.gather_research(topic)

        initial_research = (
            initial_research_1
            + initial_research_2
            + initial_research_3
            + initial_research_4
            + initial_research_5
        )

        initial_content = await self.synthesize_research(
            initial_research, topic, initial_temperature
        )

        await self.emit_message(initial_content)
        await self.done()
        return ""

    async def progress(self, message: str):
        await self.emit_status("info", message, False)

    async def done(self):
        await self.emit_status("info", "Research complete", True)

    async def emit_message(self, message: str):
        await self.__current_event_emitter__(
            {"type": "message", "data": {"content": message}}
        )

    async def emit_replace(self, message: str):
        await self.__current_event_emitter__(
            {"type": "replace", "data": {"content": message}}
        )

    async def emit_status(self, level: str, message: str, done: bool):
        await self.__current_event_emitter__(
            {
                "type": "status",
                "data": {
                    "status": "complete" if done else "in_progress",
                    "level": level,
                    "description": message,
                    "done": done,
                },
            }
        )
