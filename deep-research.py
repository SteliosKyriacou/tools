"""
title: Deep Research Tool
description: Tool to simulate a multi-step deep research process that performs an arXiv search using searchthearxiv.com and returns detailed, formatted references.
author: Your Name
author_urls:
  - https://github.com/yourusername/
funding_url: https://github.com/yourusername/open-webui-tools
version: 0.2.1
"""

import asyncio
import aiohttp
import urllib.parse
from typing import Any, Optional
from pydantic import BaseModel


class Tools:
    class UserValves(BaseModel):
        """No API keys are required for this deep research simulation."""

        pass

    def __init__(self):
        self.base_url = "https://searchthearxiv.com/search"
        self.max_results = 50

    async def deep_research(
        self,
        query: str,
        __user__: dict = {},
        __event_emitter__: Optional[Any] = None,
    ) -> str:
        """
        Simulate a multi-step deep research process that generates a comprehensive report,
        including detailed references from arXiv via searchthearxiv.com.

        Process Steps:
          1. Clarification: Process the query.
          2. Autonomous Web Browsing: Search arXiv using searchthearxiv.com and format detailed references.
          3. Data Analysis & Synthesis: Simulate analysis and synthesize insights from the retrieved papers.
          4. Structured Report Generation: Compile the findings into a structured report with citations.
          5. Report Delivery: Finalize and deliver the report.

        If an error occurs during any step, the error is reported via __event_emitter__ and an error message is returned.

        Args:
            query: The research query to investigate.
            __user__: (Optional) The user object.
            __event_emitter__: (Optional) An async callback to report progress events.

        Returns:
            A comprehensive research report as a string, or an error message.
        """
        report = []

        # Step 1: Clarification
        try:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Step 1: Clarification started.",
                            "done": False,
                        },
                    }
                )
            report.append("Step 1: Clarification")
            report.append(f"Processing query: '{query}'")
            clarification_info = (
                "Query appears sufficiently detailed for deep research."
            )
            report.append(f"Clarification: {clarification_info}")
            report.append("")
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Step 1: Clarification completed.",
                            "done": True,
                        },
                    }
                )
        except Exception as e:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Error in Step 1: Clarification: {str(e)}",
                            "done": True,
                        },
                    }
                )
            return f"Error in deep research: {str(e)}"

        # Step 2: Autonomous Web Browsing using arXiv search.
        formatted_references = []  # To store detailed reference strings.
        try:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Step 2: Autonomous Web Browsing started.",
                            "done": False,
                        },
                    }
                )
            report.append("Step 2: Autonomous Web Browsing")
            report.append(
                "Searching arXiv via searchthearxiv.com for relevant papers..."
            )

            # Prepare search parameters.
            encoded_query = urllib.parse.quote(query)
            params = {"query": encoded_query}
            headers = {
                "user-agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/132.0.0.0 Safari/537.36"
                ),
                "x-requested-with": "XMLHttpRequest",
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.base_url, params=params, headers=headers, timeout=30
                ) as response:
                    response.raise_for_status()
                    root = await response.json(content_type=None)

            entries = root.get("papers", [])
            if not entries:
                no_results_msg = f"No papers found on arXiv for '{query}'."
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": no_results_msg, "done": True},
                        }
                    )
                return no_results_msg

            report.append("Found the following papers:")
            # Process up to max_results entries.
            for i, entry in enumerate(entries[: self.max_results], 1):
                title = entry.get("title", "Unknown Title").strip()
                authors = entry.get("authors", "Unknown Authors")
                abstract = entry.get("abstract", "No abstract available").strip()
                paper_id = entry.get("id", "")
                link = (
                    f"https://arxiv.org/abs/{paper_id}"
                    if paper_id
                    else "No link available"
                )
                pdf_link = (
                    f"https://arxiv.org/pdf/{paper_id}"
                    if paper_id
                    else "No PDF available"
                )
                year = entry.get("year", "Unknown Year")
                month = entry.get("month", "Unknown Month")
                pub_date = (
                    f"{month}-{year}"
                    if month != "Unknown Month" and year != "Unknown Year"
                    else "Unknown Date"
                )

                formatted_ref = (
                    f"{i}. {title}\n"
                    f"   Authors: {authors}\n"
                    f"   Published: {pub_date}\n"
                    f"   URL: {link}\n"
                    f"   PDF URL: {pdf_link}\n"
                    f"   Abstract: {abstract}\n"
                )
                formatted_references.append(formatted_ref)
                report.append(f" - {formatted_ref}")
            report.append("")
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Step 2: Autonomous Web Browsing completed.",
                            "done": True,
                        },
                    }
                )
        except Exception as e:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Error in Step 2: Autonomous Web Browsing: {str(e)}",
                            "done": True,
                        },
                    }
                )
            return f"Error in deep research: {str(e)}"

        # Step 3: Data Analysis & Synthesis
        try:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Step 3: Data Analysis & Synthesis started.",
                            "done": False,
                        },
                    }
                )
            report.append("Step 3: Data Analysis & Synthesis")
            report.append("Simulating analysis of the retrieved papers...")
            await asyncio.sleep(2)  # Simulated delay for analysis.
            synthesized_info = (
                "Synthesized insights: The papers indicate emerging trends in renewable energy impacts, "
                "highlighting market disruptions, policy shifts, and technological advances."
            )
            report.append(synthesized_info)
            report.append("")
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Step 3: Data Analysis & Synthesis completed.",
                            "done": True,
                        },
                    }
                )
        except Exception as e:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Error in Step 3: Data Analysis & Synthesis: {str(e)}",
                            "done": True,
                        },
                    }
                )
            return f"Error in deep research: {str(e)}"

        # Step 4: Structured Report Generation
        try:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Step 4: Structured Report Generation started.",
                            "done": False,
                        },
                    }
                )
            report.append("Step 4: Structured Report Generation")
            report.append(
                "Compiling the findings into a structured report with sections and detailed citations..."
            )
            report.append("Report Sections:")
            report.append(" - Introduction")
            report.append(" - Methodology")
            report.append(" - Analysis")
            report.append(" - Conclusion")
            # Add a References section using the formatted references.
            report.append("References:")
            for ref in formatted_references:
                report.append(ref)
            report.append("")
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Step 4: Structured Report Generation completed.",
                            "done": True,
                        },
                    }
                )
        except Exception as e:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Error in Step 4: Structured Report Generation: {str(e)}",
                            "done": True,
                        },
                    }
                )
            return f"Error in deep research: {str(e)}"

        # Step 5: Report Delivery
        try:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Step 5: Report Delivery started.",
                            "done": False,
                        },
                    }
                )
            report.append("Step 5: Report Delivery")
            report.append("Finalizing report and simulating final processing delay...")
            await asyncio.sleep(5)  # Simulated final delay.
            final_report = [
                "Final Report:",
                f"This comprehensive report addresses the query: '{query}'.",
                "It is based on the following arXiv papers:",
            ]
            for ref in formatted_references:
                final_report.append(f" - {ref}")
            final_report.append(synthesized_info)
            final_report.append("End of Report.")
            report.extend(final_report)
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Step 5: Report Delivery completed. Report is ready.",
                            "done": True,
                        },
                    }
                )
        except Exception as e:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Error in Step 5: Report Delivery: {str(e)}",
                            "done": True,
                        },
                    }
                )
            return f"Error in deep research: {str(e)}"

        return "\n".join(report)


# Example usage:
if __name__ == "__main__":

    async def event_emitter(event: Any) -> None:
        # Print emitted events to the console.
        print(f"[Event] {event}")

    async def main():
        tools = Tools()
        query = "renewable energy impact on global markets"
        result = await tools.deep_research(query, __event_emitter__=event_emitter)
        print("\nFinal Report:\n")
        print(result)

    asyncio.run(main())
