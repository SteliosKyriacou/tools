"""
title: arXiv Research Pipe
description: Function Pipe made to create summary of searches using arXiv.org for relevant papers on a topic, along with PubMed, Zenodo, OpenAIRE, Europe PMC, and web searches for more contextual scientific information.
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
            default=5, description="Maximum number of search results to fetch per query"
        )
        ARXIV_MAX_RESULTS: int = Field(
            default=5, description="Maximum number of arXiv papers to fetch"
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
            default=0.5,
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

    # ---------------------------
    # Repository Search Functions
    # ---------------------------
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

    async def search_pubmed(self, query: str) -> List[Dict]:
        """Gather research from PubMed using NCBI's E-utilities."""
        await self.emit_status(
            "tool", f"Fetching PubMed articles for: {query}...", False
        )
        try:
            base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
            search_url = base_url + "esearch.fcgi"
            fetch_url = base_url + "efetch.fcgi"
            params = {
                "db": "pubmed",
                "term": query,
                "retmode": "json",
                "retmax": self.valves.MAX_SEARCH_RESULTS,
            }
            logger.debug(f"pub_med_params: {params}")
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        id_list = data.get("esearchresult", {}).get("idlist", [])
                        if not id_list:
                            await self.emit_status(
                                "tool", "No PubMed articles found.", False
                            )
                            return []
                        fetch_params = {
                            "db": "pubmed",
                            "id": ",".join(id_list),
                            "retmode": "xml",
                        }
                        async with session.get(
                            fetch_url, params=fetch_params
                        ) as fetch_response:
                            if fetch_response.status == 200:
                                text = await fetch_response.text()
                                soup = BeautifulSoup(text, "xml")
                                articles = []
                                for article in soup.find_all("PubmedArticle"):
                                    title_tag = article.find("ArticleTitle")
                                    abstract_tag = article.find("AbstractText")
                                    article_id = article.find("PMID")
                                    title = title_tag.text if title_tag else "No title"
                                    abstract = (
                                        abstract_tag.text
                                        if abstract_tag
                                        else "No abstract"
                                    )
                                    pmid = article_id.text if article_id else "N/A"
                                    articles.append(
                                        {
                                            "title": title,
                                            "content": abstract,
                                            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                                        }
                                    )
                                await self.emit_status(
                                    "tool",
                                    f"PubMed articles found: {len(articles)}",
                                    False,
                                )
                                logger.debug(f"pub_med_articles: {articles}")
                                return articles
                    else:
                        await self.emit_status(
                            "tool",
                            f"PubMed search error: HTTP {response.status}",
                            False,
                        )
        except Exception as e:
            logger.error(f"PubMed search error: {e}")
        return []

    async def search_zenodo(self, query: str) -> List[Dict]:
        """Search Zenodo for open access records."""
        await self.emit_status(
            "tool", f"Fetching Zenodo records for: {query}...", False
        )
        try:
            zenodo_url = "https://zenodo.org/api/records/"
            params = {
                "q": query,
                "size": self.valves.MAX_SEARCH_RESULTS,
            }
            async with aiohttp.ClientSession() as session:
                async with session.get(zenodo_url, params=params) as response:
                    logger.debug(f"Zenodo API response status: {response.status}")
                    if response.status == 200:
                        data = await response.json()
                        records = data.get("hits", {}).get("hits", [])
                        results = []
                        for rec in records:
                            metadata = rec.get("metadata", {})
                            results.append(
                                {
                                    "title": metadata.get("title", "No title"),
                                    "content": metadata.get(
                                        "description", "No description"
                                    ),
                                    "url": rec.get("links", {}).get("html", ""),
                                }
                            )
                        return results
        except Exception as e:
            logger.error(f"Zenodo search error: {e}")
        return []

    async def search_openaire(self, query: str) -> List[Dict]:
        """Search OpenAIRE for publications."""
        await self.emit_status(
            "tool", f"Fetching OpenAIRE records for: {query}...", False
        )
        try:
            openaire_url = "https://api.openaire.eu/search/publications"
            params = {"keywords": query, "size": self.valves.MAX_SEARCH_RESULTS}
            async with aiohttp.ClientSession() as session:
                async with session.get(openaire_url, params=params) as response:
                    logger.debug(f"OpenAIRE API response status: {response.status}")
                    if response.status == 200:
                        data = await response.json()
                        results = []
                        for rec in data.get("response", {}).get("results", []):
                            results.append(
                                {
                                    "title": rec.get("title", "No title"),
                                    "content": rec.get("description", "No description"),
                                    "url": rec.get("uri", ""),
                                }
                            )
                        return results
        except Exception as e:
            logger.error(f"OpenAIRE search error: {e}")
        return []

    async def search_europepmc(self, query: str) -> List[Dict]:
        """Search Europe PMC for open access articles."""
        await self.emit_status(
            "tool", f"Fetching Europe PMC articles for: {query}...", False
        )
        try:
            europepmc_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
            params = {
                "query": query,
                "format": "json",
                "pageSize": self.valves.MAX_SEARCH_RESULTS,
            }
            async with aiohttp.ClientSession() as session:
                async with session.get(europepmc_url, params=params) as response:
                    logger.debug(f"Europe PMC response status: {response.status}")
                    if response.status == 200:
                        data = await response.json()
                        results = []
                        for rec in data.get("resultList", {}).get("result", []):
                            results.append(
                                {
                                    "title": rec.get("title", "No title"),
                                    "content": rec.get("abstractText", "No abstract"),
                                    "url": (
                                        rec.get("fullTextUrlList", {})
                                        .get("fullTextUrl", [{}])[0]
                                        .get("url", "")
                                        if rec.get("fullTextUrlList")
                                        else ""
                                    ),
                                }
                            )
                        return results
        except Exception as e:
            logger.error(f"Europe PMC search error: {e}")
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
                                "title": res["title"],
                                "url": res["url"],
                                "content": res["content"],
                                "score": res["score"],
                            }
                            for res in results
                        ]
                    else:
                        logger.error(f"Tavily API error: {response.status}")
                        return []
            except Exception as e:
                logger.error(f"Search error: {e}")
                return []

    # ---------------------------
    # Research Aggregation
    # ---------------------------
    async def preprocess_query(self, query: str) -> tuple[str, str, str, str, str, str]:
        prompt_web = f"""
        Enhance the following query to improve the relevance of web search results:
        - Focus on adding relevant keywords, synonyms, or contextual phrases.
        - Only output the enhanced query without explanations.
    
        Initial query: "{query}"
    
        Enhanced web search query:
        """
        web_query = await self.get_completion(prompt_web)

        prompt_arxiv = f"""
        Format an optimized query for the arXiv API based on the following input:
        - Use arXiv's query syntax (AND, OR, NOT) and search fields (ti, au, abs, cat).
        - Only output the formatted arXiv API query without explanations.
    
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

        prompt_pubmed = f"""
        Format an optimized query for PubMed based on the following input:
        - Use PubMed's search syntax including MeSH terms, boolean operators (AND, OR, NOT), and field tags (e.g., [Mesh], [Title/Abstract]) if applicable.
        - For example, you might use:
          • "Neurogenesis"[Mesh] OR neurogenesis[Title/Abstract]
          • "Aged, Mice"[Mesh] OR "aged mice"[Title/Abstract]
          • treatment[Title/Abstract] OR therapeutics[Title/Abstract]
        - Combine terms using AND/OR as needed.
        - Only output the formatted PubMed query without explanations.
    
        Initial query: "{query}"
    
        Enhanced PubMed search query:
        """
        pubmed_query = await self.get_completion(prompt_pubmed)

        prompt_zenodo = f"""
        Format an optimized query for Zenodo based on the following input:
        - Focus on adding relevant keywords that align with Zenodo's record metadata.
        - Only output the formatted query without explanations.
    
        Initial query: "{query}"
    
        Enhanced Zenodo search query:
        """
        zenodo_query = await self.get_completion(prompt_zenodo)

        prompt_openaire = f"""
        Format an optimized query for OpenAIRE based on the following input:
        - Use OpenAIRE's search syntax and focus on keywords that match their publication records.
        - Only output the formatted query without explanations.
    
        Initial query: "{query}"
    
        Enhanced OpenAIRE search query:
        """
        openaire_query = await self.get_completion(prompt_openaire)

        prompt_europepmc = f"""
        Format an optimized query for Europe PMC based on the following input:
        - Use Europe PMC's search syntax, focusing on relevant keywords.
        - Only output the formatted query without explanations.
    
        Initial query: "{query}"
    
        Enhanced Europe PMC search query:
        """
        europepmc_query = await self.get_completion(prompt_europepmc)

        return (
            web_query,
            arxiv_query,
            pubmed_query,
            zenodo_query,
            openaire_query,
            europepmc_query,
        )

    async def gather_research(self, topic: str) -> List[Dict]:
        await self.emit_status("tool", "Researching...", False)
        (
            web_query,
            arxiv_query,
            pubmed_query,
            zenodo_query,
            openaire_query,
            europepmc_query,
        ) = await self.preprocess_query(topic)
        await self.emit_status(
            "tool", f"Enhanced queries - Web: {web_query} | arXiv: {arxiv_query}", False
        )

        # Perform searches across multiple repositories using optimized queries:
        web_research = await self.search_web(web_query)
        arxiv_research = await self.search_arxiv(arxiv_query)
        pubmed_research = await self.search_pubmed(pubmed_query)
        zenodo_research = await self.search_zenodo(zenodo_query)
        openaire_research = await self.search_openaire(openaire_query)
        europepmc_research = await self.search_europepmc(europepmc_query)

        research = (
            web_research
            + arxiv_research
            + pubmed_research
            + zenodo_research
            + openaire_research
            + europepmc_research
        )
        await self.emit_status(
            "user",
            f"Research gathered: ArXiv: {len(arxiv_research)}, PubMed: {len(pubmed_research)}, Zenodo: {len(zenodo_research)}, OpenAIRE: {len(openaire_research)}, EuropePMC: {len(europepmc_research)}, Web: {len(web_research)}",
            True,
        )
        return research

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
                self.__request__, form_data, user=self.__user__
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

    # async def synthesize_research(
    #     self, research: List[Dict], topic: str, temperature
    # ) -> str:
    #     research_text = "\n\n".join(
    #         f"Title: {r['title']}\nContent: {r['content']}\nURL: {r['url']}"
    #         for r in research
    #     )
    #     prompt = f"""
    #     Create a research synthesis on the topic: {topic}

    #     Available research:
    #     {research_text}

    #     Create a comprehensive synthesis that:
    #     1. Integrates the sources with links.
    #     2. Highlights key findings.
    #     3. Maintains academic rigor while being accessible.
    #     """
    #     complete = ""
    #     async for chunk in self.get_streaming_completion(
    #         [{"role": "user", "content": prompt}], temperature
    #     ):
    #         complete += chunk
    #         await self.emit_message(chunk)
    #     return complete

    async def synthesize_research(
        self, research: List[Dict], topic: str, temperature
    ) -> str:
        # Create the main research synthesis text from the gathered research
        research_text = "\n\n".join(
            f"Title: {r['title']}\nContent: {r['content']}\nURL: {r['url']}"
            for r in research
        )
        # Create a complete list of resources (titles and links) at the end
        resource_list = "\n".join(
            f"- {r['title']}: {r['url']}" for r in research if r.get("url")
        )
        prompt = f"""
            Create a research synthesis on the topic: {topic}
    
            Available research:
            {research_text}
    
            Create a comprehensive synthesis that:
            1. Integrates the sources with links.
            2. Highlights key findings.
            3. Maintains academic rigor while being accessible.
            
            At the end of the synthesis, include a section titled "Complete Resource List" that lists all resources with their titles and URLs, one per line.
            """
        complete = ""
        async for chunk in self.get_streaming_completion(
            [{"role": "user", "content": prompt}], temperature
        ):
            complete += chunk
            await self.emit_message(chunk)

        # Ensure the complete resource list is appended if not already included.
        if "Complete Resource List" not in complete:
            complete += "\n\nComplete Resource List:\n" + resource_list

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
            self.__model__, prompt.format(**format_args)
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

        # Gather initial research from all sources
        initial_research = await self.gather_research(topic)
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
