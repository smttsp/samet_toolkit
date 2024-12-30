import logging

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables.base import RunnableSequence
from pydantic import BaseModel, Field

from samet_toolkit.utils.llm_batch_processing import BatchProcessingUtility
from samet_toolkit.utils.llm_finder import LLMFinder
from samet_toolkit.utils.file_parser import SegmentMergerProcessor


class Segment(BaseModel):
    seg_idx: int = Field(..., description="The index of the segment, starts with 0")
    content: str = Field(
        ..., description="A segment of text that is coherent and complete."
    )


class PageSegment(BaseModel):
    segments: list[Segment] = Field(
        default_factory=list,
        description=(
            "List of text segments that are grouped based on semantic and structural "
            "coherence. A segment can include more than one topic or question if they "
            "are short or contextually related. Segments are not split unless "
            "absolutely necessary, such as when a topic becomes very long or unrelated "
            "content appears."
        ),
    )


class DocumentChunker:
    def __init__(self, config, llm_name: str, page_texts: list[str]):
        self.config = config
        self.page_texts = page_texts
        self.llm_name = llm_name
        self.llm = LLMFinder(config).get_llm(llm_name)

    def semantic_parse_file(self) -> list[str]:
        chain, all_inputs = self._agentic_split_inputs(self.page_texts)
        all_page_segments = BatchProcessingUtility.batch_process_with_retry(
            chain, all_inputs, PageSegment, self.llm_name
        )

        all_page_segments = self._filter_empty_content(all_page_segments)
        all_segments = self._flatten_segments(all_page_segments)
        smp_obj = SegmentMergerProcessor(self.config, all_segments)
        all_combined_segments = smp_obj.process()
        return all_combined_segments

    @staticmethod
    def _filter_empty_content(raw_segments: dict[int, PageSegment]):
        for idx, page_segment in raw_segments.items():
            for idx2, seg in enumerate(page_segment.segments):
                if not seg.content.strip():
                    print()
                    page_segment.segments.remove(seg)
        return raw_segments

    def _agentic_split_inputs(
        self, texts: list[str], num_token: int = 1000
    ) -> tuple[RunnableSequence, dict[int, dict[str, str]]]:
        """Uses LLM to split text into coherent segments (topics)"""
        system_prompt = (
            "You are a world-class text analyzer. Your task is to divide the given text"
            " into coherent segments based on semantic structure. A segment must NOT "
            "break in the middle of a topic, question, or answer. Return each segment "
            f"as a separate string. Ensure segments are concise and up to {num_token} "
            "tokens. If there is a Q&A section, both the question and answer should be"
            " in the same segment. The document might start or end in the middle of a "
            "sentence. Return the sentences as they are, DO NOT add or remove "
            "sentences. You can only change the punctuation if necessary. But you "
            "SHOULD NOT change the words in the sentence. AGAIN, one of the most "
            "important things is Question AND its Answer SHOULD BE in the same segment."
            "This is EXTREMELY important. If say the question is not complete in a"
            "page or the answer is on the other page, still try to keep them together."
        )
        output_parser = JsonOutputParser(pydantic_object=PageSegment)

        prompt = PromptTemplate(
            template="{system_prompt}\n"
            "document_content: {document_content}\n"
            "format_instructions: {format_instructions} \n",
            input_variables=["document_content"],
            partial_variables={
                "system_prompt": system_prompt,
                "format_instructions": output_parser.get_format_instructions(),
            },
        )

        chain = prompt | self.llm | output_parser

        all_inputs = {idx: {"document_content": text} for idx, text in enumerate(texts)}
        return chain, all_inputs

    @staticmethod
    def _flatten_segments(segments: dict[int, PageSegment]) -> list[str]:
        keys = sorted(segments.keys())

        all_segments = []
        for k in keys:
            text_seg = segments[k]
            sorted_text_seg = sorted(text_seg.segments, key=lambda x: x.seg_idx)

            if not sorted_text_seg:
                logging.warning(f"No segments found for page {k}")
                continue

            texts = [seg.content for seg in sorted_text_seg]

            if all_segments:
                all_segments[-1] += " " + texts[0]
                all_segments.extend(texts[1:])
            else:
                all_segments.extend(texts)
        return all_segments
