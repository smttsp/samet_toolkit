import contextvars
import logging
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from functools import partial
from typing import Any, Type, TypeVar

from langchain_core.exceptions import OutputParserException
from langchain_core.runnables.base import RunnableSequence
from pydantic import BaseModel
from vertexai.generative_models import ResponseValidationError

from samet_toolkit.utils import Providers
from samet_toolkit.utils.llm_finder import LLMFinder


class ContextExecutor(ThreadPoolExecutor):
    def submit(self, fn: Callable, *args, **kwargs) -> Future:
        ctx = contextvars.copy_context()

        return super().submit(partial(ctx.run, partial(fn, *args, **kwargs)))


# Define a generic type for the Pydantic model
T = TypeVar("T", bound=BaseModel)
MAX_RETRY = 3


class BatchProcessingUtility:
    @staticmethod
    def batch_process_with_retry(
        chain: RunnableSequence,
        all_inputs: dict[Any, dict],
        output_model: Type[T] | None,
        llm_name: str,
        max_retry: int = MAX_RETRY,
    ) -> dict[Any, T]:
        """Processes input data in batches, with retries for failed operations,
        and deserializes results into instances of a specified Pydantic model.

        Args:
            chain (RunnableSequence): The processing chain to execute.
            all_inputs (Dict[int, Dict]): A dictionary mapping input identifiers to input data.
            output_model (Type[T]): The Pydantic model class used for output deserialization.
            llm_name (str): the name of the llm model to use for processing.
            max_retry (int, optional): The maximum number of retries for failed operations. Defaults to MAX_RETRY.

        Returns:
            Dict[int, T]: A dictionary mapping input identifiers to deserialized output objects of the specified model.

        Raises:
            Exception: Propagates exceptions for individual input processing failures.

        Example:
            > class EdgeList(BaseModel):
            > ...     edges: list
            > ...
            > utility = BatchProcessingUtility()
            > result = utility.batch_process_with_retry(chain, all_inputs, EdgeList)
        """
        provider = LLMFinder._find_provider_from_name(llm_name)
        bp = BatchProcessing(provider=provider)

        all_res_dict = {}
        while all_inputs:
            all_raw_res = bp.get_results_via_batch_processing(chain, all_inputs)
            remaining = {}
            for idx, res in all_raw_res.items():
                try:
                    if not res:
                        remaining[idx] = all_inputs[idx]
                    all_res_dict[idx] = output_model(**res) if output_model else res
                except Exception:
                    remaining[idx] = all_inputs[idx]
                    print("Error in", idx, res)
            all_inputs = remaining
            if max_retry == 0:
                break
            max_retry -= 1
            if len(remaining):
                logging.warning(
                    f"Retrying {len(remaining)} docs with {max_retry=} left"
                )
        return all_res_dict


class BatchProcessing:
    # FIXME: Retry mechanism is not implemented

    _executor = ContextExecutor()

    def __init__(self, provider: Providers):
        self.llm_name = provider

    def get_results_via_batch_processing(
        self, chain, inputs: list[dict] | dict[int, dict], max_workers: int = 50
    ):
        if self.llm_name == Providers.OPENAI:
            all_res = self._openai_batch_processing(chain, inputs)
        elif self.llm_name == Providers.VERTEXAI:
            all_res = self._vertexai_batch_processing(
                chain, inputs, max_workers=max_workers
            )
        else:
            raise ValueError(f"Unknown LLM provider: {self.llm_name}")
        return all_res

    @staticmethod
    def _openai_batch_processing(
        chain: RunnableSequence,
        all_inputs: list[dict],
        batch_size: int = 50,
        break_time: float = 60,
    ):
        all_res: list[dict] = []
        for i in range(0, len(all_inputs), batch_size):
            batch = all_inputs[i : min(i + batch_size, len(all_inputs))]
            batch_results = chain.batch(batch)
            all_res.extend(batch_results)

            if i + batch_size < len(all_inputs):
                print(f"Processed {i + batch_size}/{len(all_inputs)} documents")
                time.sleep(break_time)
        return all_res

    def _vertexai_batch_processing(
        self,
        chain: RunnableSequence,
        remaining_inst_dict: dict[int, dict],
        max_workers: int = 50,
    ):
        # idx = next(iter(remaining_inst_dict))
        # self._fetch_with_invoke(chain, {idx: remaining_inst_dict[idx]})
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(self._fetch_with_invoke, chain, inst): idx
                for idx, inst in remaining_inst_dict.items()
            }

            results = {}
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    logging.warning(f"Generated an exception for idx {idx}: %s", exc)
        return results

    @staticmethod
    def _fetch_with_invoke(chain: RunnableSequence, run_dict: dict) -> dict:
        try:
            return chain.invoke(run_dict)
        except (OutputParserException, ResponseValidationError) as e:
            logging.debug(
                "OutputParserException or ResponseValidationError occurred %s", e
            )
            return {}
        except Exception as e:
            logging.warning("Unexpected error: %s", e)
            return {}
